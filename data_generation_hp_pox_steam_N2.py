# %%
#!/usr/bin/env python3
"""
Cantera dataset generation for the chemical-kinetics Neural ODE.

Generates 0-D, constant-pressure, adiabatic methane auto-ignition trajectories
across a Latin-hypercube design in (initial temperature, O2/CH4 ratio,
pressure), samples each trajectory on a non-uniform grid concentrated where the
chemistry actually moves, and saves states, exact derivatives and times.

Two choices are worth knowing about up front.

EXACT DERIVATIVES, NOT FINITE DIFFERENCES. The derivative at every saved sample
comes from calling the same Cantera rhs() used for the reference solve,
evaluated directly at that state. The alternative -- finite-differencing the
saved samples -- would be unreliable here, because the arc-length sampler places
many adjacent samples sub-nanosecond apart while chasing fast early radical
chemistry, and differencing across a near-duplicate timestamp produces a huge
spurious derivative that has nothing to do with the chemistry. Evaluating the
rhs directly decouples derivative quality from sample timing entirely.

SANITY CHECKS AT GENERATION TIME. run_dataset_sanity_checks() runs
automatically and prints a PASS/FAIL report before anything is written. Catching
a malformed dataset here costs seconds; catching it later means debugging a
training run that was never going to work.

Outputs land in EXPERIMENT_ROOT (see PATHS below): states, derivatives, times,
phase coordinates, case metadata, a JSON record of the generation settings, and
diagnostic plots.

Install:
  pip install cantera scipy numpy matplotlib
"""

from pathlib import Path
import heapq
import json
import time
import warnings

import cantera as ct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3-D axes)
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.stats import qmc


warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Richter et al. report 50, 60, and 70 bar(g). Cantera requires absolute
# pressure, so the simplified benchmark uses approximately 51, 61, 71 bar(a).
PRESSURE_LOWER_BAR = 51.0
PRESSURE_UPPER_BAR = 71.0

# Initial temperature of the premixed 0-D reactor.
# NOTE: Fig. 8 of Richter et al. (2015) is the CALIBRATION curve of the optical
# probe (grey value vs. oven temperature, measured at atmospheric pressure in a
# separate rig); its 1000-1750 degC axis is not a process temperature.
#   "benchmark"      reactor temperatures of the four test cases, 1200 / 1400
#                    degC (Table 1 target, Table 6 control thermocouple G12:
#                    1201 / 1201 / 1198 / 1401 degC)          -> 1473-1673 K
#   "measured_wide"  lowest measured reactor gas temperature (Table 6, 1089
#                    degC) to the outer-flame temperature (Fig. 9, ~1650 degC),
#                    rounded to 1100-1650 degC                -> 1373-1923 K
# Either way T0 is the temperature of a premixed reaction-zone state, not a feed
# inlet temperature (feeds enter at 240-365 degC, Table 1, and would not
# autoignite in a closed 0-D reactor); Zhou et al. (2010) use the same kind of
# assumption (T0 = 1250-1550 K).
T_RANGE_OPTION = "benchmark"
if T_RANGE_OPTION == "benchmark":
    T_LOWER = 1473.15   # 1200 degC
    T_UPPER = 1673.15   # 1400 degC
elif T_RANGE_OPTION == "measured_wide":
    T_LOWER = 1373.15   # 1100 degC
    T_UPPER = 1923.15   # 1650 degC
else:
    raise ValueError(f"unknown T_RANGE_OPTION {T_RANGE_OPTION!r}")

# Fuel = pure CH4 (agreed simplification). Treating each reported natural-gas
# flow as pure CH4 by mass gives the molar ratio (m_O2/32)/(m_NG/16)
# = 0.5*m_O2/m_NG (the "equivalent CH4" of Zhou et al. 2010, Table 2 footnote).
# Cantera receives CH4:1, O2:ratio, H2O:steam ratio, N2:N2 ratio.
REFERENCE_O2_CH4_RATIOS = (
    0.592158700,
    0.583816434,
    0.565822946,
    0.715288412,
)
O2_CH4_LOWER = min(REFERENCE_O2_CH4_RATIOS)
O2_CH4_UPPER = max(REFERENCE_O2_CH4_RATIOS)

# Reproducible 3-D Latin hypercube design. Extend each paper-derived span by
# 10% on both sides; this is a moderate extrapolation margin, not 10% of the
# absolute temperature/pressure values. The four exact simplified paper cases
# are appended as anchors after the LHS points.
RANGE_EXTENSION_FRACTION = 0.10
# Same 10% margin as T and P: 0.551-0.730 mol/mol.  (A 0.60 margin reached
# 0.476, below the partial-oxidation stoichiometry CH4 + 1/2 O2 -> CO + 2 H2
# (0.5), far outside the benchmark (0.566-0.715), in the fuel-rich, sooting
# regime that GRI-Mech 3.0 has no chemistry for.)
O2_CH4_RANGE_EXTENSION_FRACTION = RANGE_EXTENSION_FRACTION

# -----------------------------------------------------------------------------
# STEAM AND DILUENT
# -----------------------------------------------------------------------------
# The HP-POX feed is roughly 20 mol% steam, computed from the benchmark mass
# flows (H2O/CH4 ~ 0.40 mol/mol across all four reference cases). Steam is not
# ballast here: it drives steam reforming (CH4 + H2O -> CO + 3H2) and water-gas
# shift (CO + H2O <-> CO2 + H2), which together set the syngas H2/CO ratio that
# the reactor exists to produce. Omitting it models methane-oxygen ignition
# rather than HP-POX.
#
# Nitrogen is ~1 mol% of the feed: the Optisos purge (Table 1) plus the N2 of
# the natural gas (Table 3, ~0.86 vol%).  Per case, (m_N2,purge/28.014 +
# n_NG * x_N2,NG) / (m_NG/16) = 0.0203 / 0.0199 / 0.0191 / 0.0223, mean 0.0204;
# this reproduces the measured outlet N2 of ~0.65 vol% (Table 4).  Chemically
# inert (the mechanism has no N chemistry), but it carries mass and heat
# capacity.  It is constant in time within a case but differs slightly between
# cases, so the training code (v7, INERT_SPECIES) carries it per case from the
# initial state instead of learning it.
#
# Set INCLUDE_STEAM = False to reproduce the earlier CH4/O2-only dataset.
INCLUDE_STEAM = True
H2O_CH4_LOWER = 0.393          # from the benchmark mass flows
H2O_CH4_UPPER = 0.402
H2O_CH4_RANGE_EXTENSION_FRACTION = 4.0   # the four reference values are nearly
                                         # identical, so a fraction of their
                                         # spread would give no variation at all
H2O_CH4_MARGIN = H2O_CH4_RANGE_EXTENSION_FRACTION * (H2O_CH4_UPPER - H2O_CH4_LOWER)
H2O_CH4_SAMPLE_LOWER = max(H2O_CH4_LOWER - H2O_CH4_MARGIN, 0.0)
H2O_CH4_SAMPLE_UPPER = H2O_CH4_UPPER + H2O_CH4_MARGIN
N2_CH4_RATIO = 0.0204          # benchmark mean (purge + natural-gas N2), held fixed
N_LHS_CASES = 400
LHS_SEED = 20250814

T_MARGIN = RANGE_EXTENSION_FRACTION * (T_UPPER - T_LOWER)
O2_CH4_MARGIN = O2_CH4_RANGE_EXTENSION_FRACTION * (O2_CH4_UPPER - O2_CH4_LOWER)
PRESSURE_MARGIN_BAR = RANGE_EXTENSION_FRACTION * (
    PRESSURE_UPPER_BAR - PRESSURE_LOWER_BAR
)

T_SAMPLE_LOWER = T_LOWER - T_MARGIN
T_SAMPLE_UPPER = T_UPPER + T_MARGIN
O2_CH4_SAMPLE_LOWER = O2_CH4_LOWER - O2_CH4_MARGIN
O2_CH4_SAMPLE_UPPER = O2_CH4_UPPER + O2_CH4_MARGIN
PRESSURE_SAMPLE_LOWER_BAR = PRESSURE_LOWER_BAR - PRESSURE_MARGIN_BAR
PRESSURE_SAMPLE_UPPER_BAR = PRESSURE_UPPER_BAR + PRESSURE_MARGIN_BAR

# NEW: targeted oversampling of the low-O2/CH4 (most fuel-rich) corner.
# Motivated directly by a training-time finding: 4 windowed-shooting
# training cases persistently failed to converge (zero gradient signal for
# hundreds of epochs, since a rejected window contributes no backward()
# pass), and all 4 sat in the 1st-7th percentile of O2/CH4 -- the region
# closest to the theoretical minimum oxygen needed for the reaction to
# proceed at all (stoichiometric partial oxidation, CH4 + 0.5 O2 -> CO +
# 2H2, is 0.5; the main LHS design's extended sample floor is only ~10%
# above that). 4 examples out of 300 is too few for the model to learn a
# genuinely distinct chemical regime, regardless of loss weighting.
#
# This adds a SEPARATE, smaller LHS design (own seed, so it doesn't disturb
# the reproducibility of the main 300-case design), jointly stratifying T
# and pressure exactly as the main design does, but with O2/CH4 restricted
# to [O2_CH4_SAMPLE_LOWER, O2_CH4_LOWER] -- the extended sample floor up to
# the reference cases' own minimum, which is exactly the sub-range all 4
# failing cases fell into.
N_EXTRA_LOW_O2CH4_CASES = 0     # disabled: the design follows the benchmark, not
                                # the cases a model found hard (set > 0 to restore)
EXTRA_LOW_O2CH4_SEED = 20260827  # distinct from LHS_SEED, so this addition is independently reproducible
O2_CH4_EXTRA_LOWER = O2_CH4_SAMPLE_LOWER
O2_CH4_EXTRA_UPPER = O2_CH4_LOWER

REFERENCE_CASES = (
    (1473.15, REFERENCE_O2_CH4_RATIOS[0], 51.0),
    (1473.15, REFERENCE_O2_CH4_RATIOS[1], 61.0),
    (1473.15, REFERENCE_O2_CH4_RATIOS[2], 71.0),
    (1673.15, REFERENCE_O2_CH4_RATIOS[3], 51.0),
)

FUEL = "CH4"
OXIDIZER = "O2:1.0"  # O2 stream; steam and N2 are added via H2O/CH4 and N2/CH4

# Integration horizon.
# The script can automatically extend it if ignition is too close to the end.
T_START = 0.0
# Start with a 1 ms horizon so the retained trajectories visibly include the
# flat steady-state tail. The automatic checks below still extend this value if
# any case has not equilibrated by 90% of the horizon.
INITIAL_T_END = 1.0e-3
MAX_T_END = 1.0e5
HORIZON_GROWTH = 10.0
MAX_HORIZON_ATTEMPTS = 15

# High-accuracy reference solver.
SOLVER_METHOD = "LSODA"
RTOL = 1e-10
ATOL = 1e-12
# Let LSODA choose large steps after the fast chemistry has settled. A fixed
# 1e-5 s cap is useful near ignition only if the solver cannot adapt, but LSODA
# already takes much smaller steps there; retaining the cap creates millions
# of unnecessary points when a rich case needs seconds to equilibrate.
MAX_STEP = np.inf

# Case-specific hybrid sampling. These quotas add up to 200 points. The middle
# region is not split at fixed event phases: its points are selected from
# continuous, species-aware arc length, so sharp or multi-stage chemistry gets
# resolution wherever the actual trajectory needs it.
# The induction period spans several decades of log-time; the ignition event
# spans well under one. Giving the arc-length sampler 350 of 475 points put
# roughly a hundred times more resolution per decade on the event than on the
# induction, which is where a log-time surrogate has the least supervision and
# where chain branching amplifies any error. Rebalanced so the induction period
# gets usable resolution; the total is unchanged.
N_IDLE = 145
N_DYNAMIC_ARC = 255
N_STEADY = 150
N_POINTS_PER_TRAJECTORY = N_IDLE + N_DYNAMIC_ARC + N_STEADY

# Of the 25 idle points, keep these on the genuinely flat initial baseline.
# The remaining idle points resolve the departure from that baseline.
N_FLAT_IDLE = 6
IDLE_BASELINE_END_FRACTION_OF_ONSET = 0.25
# Sets where the first positive sample lands, and therefore how large the jump
# in log10(Y) across the very first interval is. Smaller means the first stored
# sample is closer to t=0, so radicals have grown less and the jump is smaller.
# HO2 grows almost exactly linearly in time through early induction (log-log
# slope ~0.99), so each decade earlier removes a full decade from the jump:
# 1e-4 gives ~5.1 decades, 1e-5 ~4.4, 1e-6 ~3.4. The cost is that fewer species
# are above the representation floor on the earliest samples -- around 1e-6 only
# the chain-branching pool (HO2, H, CH3, OH) still carries signal, which is the
# point at which going further stops being worthwhile. plot_floor_diagnostic and
# plot_first_interval_jump report both sides of this.
IDLE_FIRST_POSITIVE_FRACTION_OF_ONSET = 1.0e-6

# Place the idle samples uniformly in log-time rather than warping them toward
# onset. Log spacing already concentrates points near onset in linear time, and
# it is uniform in the coordinate the surrogate actually integrates in, so every
# decade of radical growth gets equal resolution. Set False for the previous
# onset-warped behaviour.
IDLE_LOG_UNIFORM = True

# Dense candidate curve used only to calculate continuous state-space arc
# length. The final stored dataset still contains N_POINTS_PER_TRAJECTORY.
N_ARC_CANDIDATES = 4000
ARC_TEMPERATURE_WEIGHT = 1.0
ARC_SPECIES_WEIGHT = 1.0
ARC_LOG_TIME_WEIGHT = 0.03
# CHANGED: was 1e-8. That threshold let essentially any radical-pool
# species (H, O, OH, HO2 -- typical peak mass fractions ~1e-5 to 1e-7 in
# this mechanism/regime) participate in arc-length sampling. 1e-4 keeps
# real minor species (a few tenths of a percent or more) in the metric
# while excluding species whose total excursion is too small to matter
# for a combustion-relevant surrogate. This does NOT remove those
# species from the dataset -- every species is still saved with its
# exact analytic derivative at every sample point -- it only stops them
# from driving WHERE points get placed.
MIN_SPECIES_EXCURSION_FOR_ARC = 1e-4

# NEW: extra forced checkpoint times strictly inside the
# transition_end -> equilibrium segment. This segment is the one most
# likely to be too long/stiff for a single training-time dopri5 call --
# it's exactly the span that made MAX_T_END need to grow to 100 s (slow
# CO/H2/H2O relaxation toward the true HP-equilibrium state). These
# checkpoints are forced into the arc-length selection (so they are
# guaranteed to be real saved sample points, at exact solver-verified
# states, not interpolated), and their indices are saved to
# case_metadata.npz as tail_subwindow_sample_indices. They do NOT change
# any existing anchor, window, or array shape your training script
# already reads -- this is purely additive. Your training script's
# window builder can optionally use these as extra re-anchor boundaries
# inside that one segment (splitting it into
# N_TAIL_SUBWINDOW_CHECKPOINTS + 1 shorter windows instead of one), or
# ignore them entirely with zero effect on the existing 6-anchor scheme.
N_TAIL_SUBWINDOW_CHECKPOINTS = 3

# Event thresholds and window padding. Ignition is still defined by max(dT/dt).
# The onset is the first pre-peak point reaching 1% of peak dT/dt. The end of
# the rapid transition and equilibrium use normalized full-state activity, so
# important species must also have settled rather than temperature alone.
IGNITION_ONSET_RATE_FRACTION = 0.01
IGNITION_TRANSITION_ACTIVITY_FRACTION = 0.05
EQUILIBRIUM_ACTIVITY_FRACTION = 0.001
# A state is accepted as equilibrated only when every active component is this
# close to its final value, relative to that component's full excursion.
EQUILIBRIUM_STATE_FRACTION = 0.002
MIN_SPECIES_EXCURSION_FOR_ACTIVITY = 1e-8
PRE_IGNITION_PAD_FRACTION = 0.20
PRE_IGNITION_MIN_FRACTION_OF_TIGN = 0.02
POST_EQUILIBRIUM_PAD_FRACTION = 0.15
# Keep t=0 plus N_FLAT_IDLE sparse baseline points. The two-zone idle sampler
# prevents this from recreating the old, overlong induction allocation.
KEEP_IDLE_BASELINE_FROM_T0 = True
# Values above 1 bias the idle samples toward ignition onset, where the curve
# first leaves its flat baseline. 2.5 is deliberately moderate: it retains a
# few genuinely idle samples instead of collapsing all 25 points onto onset.
IDLE_TIME_WARP_STRENGTH = 2.5
# True keeps the complete verified tail through the common t_end. Only
# N_STEADY points are placed there, so this does not steal resolution from the
# reaction/relaxation arc-length budget.
KEEP_FULL_STEADY_TAIL_TO_T_END = True
LOG_SPACE_STEADY_TAIL = True

# Sanity criteria used for automatic horizon extension.
MIN_TEMPERATURE_RISE_FOR_IGNITION = 100.0
END_GUARD_FRACTION = 0.90
EQUILIBRIUM_END_GUARD_FRACTION = 0.80

# =============================================================================
# NEW (v2): DERIVATIVE STORAGE / SANITY-CHECK / ODEINT-VERIFICATION CONFIG
# =============================================================================

# Mass-fraction physical bounds and conservation tolerances. Species mass
# fractions must lie in [0, 1] and sum to 1 at every sampled point; dY/dt
# summed across every species must be ~0 at every point (Cantera's
# net_production_rates * MW / density formula conserves mass by
# construction if computed correctly -- a nonzero sum here is a strong,
# cheap signal something in the derivative pipeline is broken).
SPECIES_SUM_TOLERANCE = 1e-6
MASS_CONSERVATION_TOLERANCE = 1e-8  # on sum(dY/dt), physical units (1/s)

# Timestamp-gap sanity check (the check that surfaced the near-duplicate-
# timestamp / oversampling-near-t0 issue this session). A gap this small is
# almost certainly not resolving real dynamics -- it is far below any
# plausible collision-limited chemical timescale for this system -- and is
# flagged as suspect rather than silently trusted.
DEGENERATE_GAP_SECONDS = 1e-8
# If more than this fraction of a case's adjacent sample gaps are below
# DEGENERATE_GAP_SECONDS, that case is flagged in the sanity report as an
# oversampling-near-t0 candidate (see run_dataset_sanity_checks).
DEGENERATE_GAP_FRACTION_WARN_THRESHOLD = 0.10

# Finite-difference cross-check: for points whose adjacent gap is safely
# above the degenerate threshold, compare the exact rhs()-computed
# derivative against a simple central finite difference of the saved
# states. This is a complementary check to mass conservation -- it can
# catch a sign error or unit mismatch that a conservation check alone would
# not (conservation only checks that species sum to zero net, not that the
# individual signs/magnitudes are right).
FINITE_DIFF_CHECK_MIN_GAP_SECONDS = 1e-6
FINITE_DIFF_RELATIVE_TOLERANCE = 0.25  # generous -- this is a coarse cross-check, not a precision claim
# A relative-error ratio is only meaningful when BOTH sides are comparing
# genuinely resolved, non-negligible derivative estimates. Many species sit
# at EXACTLY 0.0 (bit-identical, not just small) until they first turn on --
# common for the ~20 species that start the mixture at zero -- and the
# central-difference window straddling that exact turn-on instant produces
# a nonzero finite-difference estimate against an exact-zero rhs() value at
# the midpoint, which is a 0-vs-nonzero comparison, not a disagreement about
# the derivative's magnitude. That comparison is ill-posed (relative error
# saturates at exactly 1.0) and is NOT evidence of a derivative bug -- it is
# an artifact of comparing a genuinely-zero rate to a proxy for it. Only
# compare points where the SMALLER of the two magnitudes still clears this
# floor, which excludes exactly that artifact along with pure numerical
# noise, leaving only comparisons between two real, resolved estimates.
FINITE_DIFF_MIN_ABSOLUTE_MAGNITUDE = 1e-6

# =============================================================================
# PATHS
# =============================================================================

MECHANISM_CANDIDATES = [
    Path("/workspace/grimech30_113.yaml"),
    Path("grimech30_113.yaml"),
]

MECHANISM = next((p for p in MECHANISM_CANDIDATES if p.exists()), None)

if MECHANISM is None:
    raise FileNotFoundError(
        "Could not find grimech30_113.yaml. Put it in /workspace or next "
        "to this script."
    )

if Path("/workspace").exists():
    EXPERIMENT_ROOT = Path("/workspace/hp_pox_steam_N2_data")
else:
    EXPERIMENT_ROOT = Path("hp_pox_steam_N2_data")   # new folder: does not overwrite older data

print(f"Experiment path: {EXPERIMENT_ROOT}")

DATA_DIR = EXPERIMENT_ROOT / "data"
PLOTS_DIR = EXPERIMENT_ROOT / "data_plots"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CHEMISTRY RHS
# =============================================================================

class UnphysicalStateError(RuntimeError):
    """
    Raised when rhs() is asked to evaluate the derivative field at a state
    that is not physically valid (non-finite, or non-positive temperature).

    This is deliberately NOT allowed to surface as Cantera's raw
    CanteraError -- catching it explicitly at the call site lets us report
    *which* solver/case produced the invalid trial state instead of an
    opaque C++ exception, and lets a caller decide whether to abort or to
    record this as a failed verification case and continue.
    """


def make_rhs(gas, pressure_pa):
    """
    Return RHS for state [T, Y_1, ..., Y_N].

    We integrate all mechanism species directly. Cantera receives the full
    composition. Tiny negative solver overshoots are clipped only before
    querying Cantera; the ODE state itself is not otherwise modified.
    """

    def rhs(t, y):
        T = float(y[0])
        Y = np.asarray(y[1:], dtype=np.float64)

        # Explicit RK steppers (e.g. torchdiffeq's dopri5) evaluate the RHS
        # at intermediate/trial stage states that are not guaranteed to lie
        # in the physically valid region, especially for stiff chemistry
        # like ignition. Detect that here with a clear, catchable error
        # instead of letting a NaN/non-positive T crash inside Cantera with
        # an opaque CanteraError.
        if not np.isfinite(T) or T <= 0.0:
            raise UnphysicalStateError(
                f"rhs() called with invalid temperature T={T} at t={t}"
            )
        if not np.all(np.isfinite(Y)):
            raise UnphysicalStateError(
                f"rhs() called with non-finite species state at t={t}"
            )

        # Protect the Cantera query from tiny numerical overshoots.
        Y_safe = np.clip(Y, 0.0, None)
        ysum = Y_safe.sum()

        if ysum <= 0.0:
            raise UnphysicalStateError(
                f"Species mass-fraction sum became non-positive at t={t}."
            )

        # Renormalize only for the Cantera property query.
        Y_safe = Y_safe / ysum

        gas.TPY = T, pressure_pa, Y_safe

        dTdt = -(
            np.dot(
                gas.net_production_rates,
                gas.partial_molar_enthalpies,
            )
            / (gas.density * gas.cp_mass)
        )

        dYdt = (
            gas.net_production_rates
            * gas.molecular_weights
            / gas.density
        )

        return np.concatenate(([dTdt], dYdt))

    return rhs


def initial_state(mechanism_path, T0, o2_ch4_ratio, pressure_pa,
                  h2o_ch4_ratio=0.0, n2_ch4_ratio=0.0):
    """Initial mixture on a per-mole-of-CH4 basis.

    Cantera normalises the mole-fraction dict, so the ratios are all that
    matter. Steam and nitrogen are omitted entirely when their ratio is zero,
    which keeps the CH4/O2-only dataset byte-reproducible.
    """
    gas = ct.Solution(str(mechanism_path))
    if o2_ch4_ratio <= 0.0:
        raise ValueError("O2/CH4 molar ratio must be positive.")
    if h2o_ch4_ratio < 0.0 or n2_ch4_ratio < 0.0:
        raise ValueError("H2O/CH4 and N2/CH4 ratios must be non-negative.")
    composition = {"CH4": 1.0, "O2": float(o2_ch4_ratio)}
    if h2o_ch4_ratio > 0.0:
        composition["H2O"] = float(h2o_ch4_ratio)
    if n2_ch4_ratio > 0.0:
        composition["N2"] = float(n2_ch4_ratio)
    gas.TPX = (T0, pressure_pa, composition)

    return gas, np.concatenate(([gas.T], gas.Y.copy()))


# =============================================================================
# REFERENCE SOLVE / IGNITION DETECTION
# =============================================================================

def _interpolated_crossing_time(t, signal, left_index, threshold):
    """Linearly interpolate a threshold crossing between i and i+1."""
    i = int(left_index)
    if i < 0:
        return float(t[0])
    if i >= len(t) - 1:
        return float(t[-1])

    x0 = float(signal[i])
    x1 = float(signal[i + 1])
    if not np.isfinite(x0 + x1) or abs(x1 - x0) <= np.finfo(float).eps:
        return float(t[i + 1])

    weight = np.clip((threshold - x0) / (x1 - x0), 0.0, 1.0)
    return float(t[i] + weight * (t[i + 1] - t[i]))


def detect_case_events(sol, rhs, T0, equilibrium_state):
    """
    Detect the physically useful sampling window from the adaptive solve.

    Full-state activity locates the rapid transition. Equilibrium is verified
    against Cantera's thermodynamic HP-equilibrium state for the same initial
    mixture, rather than against the arbitrary endpoint of a finite solve.
    """
    t = np.asarray(sol.t, dtype=np.float64)
    y = np.asarray(sol.y, dtype=np.float64).T

    # Temperature always participates. Species participate only when their
    # total excursion is large enough to matter physically.
    equilibrium_state = np.asarray(equilibrium_state, dtype=np.float64)
    if equilibrium_state.shape != (y.shape[1],):
        raise ValueError("Equilibrium state has the wrong dimension.")
    excursions = np.maximum(np.ptp(y, axis=0), np.abs(equilibrium_state - y[0]))
    active_components = np.zeros(y.shape[1], dtype=bool)
    active_components[0] = excursions[0] >= MIN_TEMPERATURE_RISE_FOR_IGNITION
    active_components[1:] = excursions[1:] >= MIN_SPECIES_EXCURSION_FOR_ACTIVITY

    # Evaluate derivatives without materializing a (n_steps, n_states) dydt
    # matrix. The previous expression np.max(np.abs(dydt), axis=0) briefly
    # required two additional full-size arrays and could fail late in a large
    # pressure/temperature/composition sweep. These streaming passes produce the same
    # dT/dt and normalized activity signals with bounded temporary memory.
    dTdt = np.empty(t.size, dtype=np.float64)
    peak_component_rates = np.zeros(y.shape[1], dtype=np.float64)
    for i, (ti, yi) in enumerate(zip(t, y)):
        rate = np.asarray(rhs(ti, yi), dtype=np.float64)
        dTdt[i] = rate[0]
        np.maximum(peak_component_rates, np.abs(rate), out=peak_component_rates)

    ign_idx = int(np.argmax(dTdt))
    t_ign = float(t[ign_idx])
    max_dTdt = float(dTdt[ign_idx])
    if not np.isfinite(max_dTdt) or max_dTdt <= 0.0:
        raise RuntimeError("Could not detect a positive ignition dT/dt peak.")

    active_components &= peak_component_rates > np.finfo(float).tiny
    if not np.any(active_components):
        active_components[0] = True

    active_peaks = peak_component_rates[active_components]
    activity = np.empty(t.size, dtype=np.float64)
    for i, (ti, yi) in enumerate(zip(t, y)):
        active_rate = np.abs(
            np.asarray(rhs(ti, yi), dtype=np.float64)[active_components]
        )
        activity[i] = np.max(active_rate / active_peaks)
    temperature_peak_idx = int(np.argmax(y[:, 0]))
    t_temperature_peak = float(t[temperature_peak_idx])

    # First pre-peak crossing of the temperature-rate onset threshold.
    onset_threshold = IGNITION_ONSET_RATE_FRACTION * max_dTdt
    onset_candidates = np.flatnonzero(dTdt[:ign_idx + 1] >= onset_threshold)
    if onset_candidates.size == 0:
        onset_idx = ign_idx
        t_onset = t_ign
    else:
        onset_idx = int(onset_candidates[0])
        t_onset = _interpolated_crossing_time(
            t,
            dTdt,
            onset_idx - 1,
            onset_threshold,
        )

    # Use the last above-threshold point so a later secondary kinetic feature
    # cannot be accidentally cut off.
    transition_candidates = np.flatnonzero(
        (np.arange(t.size) >= ign_idx)
        & (activity > IGNITION_TRANSITION_ACTIVITY_FRACTION)
    )
    transition_last = int(transition_candidates[-1])
    if transition_last >= t.size - 1:
        t_transition_end = float(t[-1])
        transition_reached = False
    else:
        t_transition_end = _interpolated_crossing_time(
            t,
            activity,
            transition_last,
            IGNITION_TRANSITION_ACTIVITY_FRACTION,
        )
        transition_reached = True

    # Measure each active component's remaining distance to the independently
    # calculated HP-equilibrium state. This stays memory-bounded and detects
    # slow CO/H2/CO2/H2O relaxation without treating the current endpoint as
    # equilibrium by definition.
    active_excursions = excursions[active_components]
    target_active_state = equilibrium_state[active_components]
    state_distance = np.empty(t.size, dtype=np.float64)
    for i, yi in enumerate(y):
        state_distance[i] = np.max(
            np.abs(yi[active_components] - target_active_state)
            / active_excursions
        )

    equilibrium_candidates = np.flatnonzero(
        (np.arange(t.size) >= ign_idx)
        & (state_distance > EQUILIBRIUM_STATE_FRACTION)
    )
    if equilibrium_candidates.size == 0:
        t_equilibrium = t_ign
        equilibrium_reached = True
    else:
        equilibrium_last = int(equilibrium_candidates[-1])
        if equilibrium_last >= t.size - 1:
            t_equilibrium = float(t[-1])
            equilibrium_reached = False
        else:
            t_equilibrium = _interpolated_crossing_time(
                t,
                state_distance,
                equilibrium_last,
                EQUILIBRIUM_STATE_FRACTION,
            )
            equilibrium_reached = True

    # Refine the temperature maximum on the continuous LSODA interpolant. The
    # adaptive solver history brackets the peak well but does not necessarily
    # contain the maximum itself.
    if t_equilibrium > t_ign:
        peak_search = minimize_scalar(
            lambda ti: -float(sol.sol(ti)[0]),
            bounds=(t_ign, t_equilibrium),
            method="bounded",
            options={
                "xatol": max(
                    np.finfo(float).eps,
                    1e-8 * (t_equilibrium - t_ign),
                )
            },
        )
        peak_candidates = np.array(
            [t_temperature_peak, t_ign, t_equilibrium, peak_search.x],
            dtype=np.float64,
        )
        peak_temperatures = np.asarray(sol.sol(peak_candidates)[0], dtype=np.float64)
        t_temperature_peak = float(peak_candidates[int(np.argmax(peak_temperatures))])

    # Preserve only a short, genuinely idle lead-in and a short steady tail.
    induction_width = max(
        t_ign - t_onset,
        PRE_IGNITION_MIN_FRACTION_OF_TIGN * max(t_ign, np.finfo(float).eps),
    )
    if KEEP_IDLE_BASELINE_FROM_T0:
        window_start = T_START
    else:
        window_start = max(
            T_START,
            t_onset - PRE_IGNITION_PAD_FRACTION * induction_width,
        )

    relaxation_width = max(
        t_equilibrium - t_transition_end,
        t_transition_end - t_ign,
        np.finfo(float).eps,
    )
    if KEEP_FULL_STEADY_TAIL_TO_T_END:
        window_end = float(t[-1])
    else:
        window_end = min(
            float(t[-1]),
            t_equilibrium + POST_EQUILIBRIUM_PAD_FRACTION * relaxation_width,
        )

    return {
        "ign_idx": ign_idx,
        "t_ign": t_ign,
        "t_temperature_peak": t_temperature_peak,
        "max_dTdt": max_dTdt,
        "t_onset": float(t_onset),
        "t_transition_end": float(t_transition_end),
        "t_equilibrium": float(t_equilibrium),
        "window_start": float(window_start),
        "window_end": float(window_end),
        "transition_reached": bool(transition_reached),
        "equilibrium_reached": bool(equilibrium_reached),
        "activity_final": float(activity[-1]),
        "equilibrium_error_final": float(state_distance[-1]),
        "n_active_components": int(np.sum(active_components)),
    }

def solve_case(T0, o2_ch4_ratio, pressure_bar, t_end, h2o_ch4_ratio=0.0):
    # CHANGED: pressure_bar is now an explicit argument instead of reading the
    # global PRESSURE_BAR -- every other line here is unchanged.
    pressure_pa = pressure_bar * 1e5

    gas, y0 = initial_state(
        MECHANISM,
        T0,
        o2_ch4_ratio,
        pressure_pa,
        h2o_ch4_ratio,
        N2_CH4_RATIO if INCLUDE_STEAM else 0.0,
    )

    # Compute the physically defined constant-enthalpy, constant-pressure
    # equilibrium target before the RHS starts mutating its working gas object.
    equilibrium_gas = ct.Solution(str(MECHANISM))
    equilibrium_gas.TPY = gas.TPY
    equilibrium_gas.equilibrate("HP")
    equilibrium_state = np.concatenate(
        (np.array([equilibrium_gas.T], dtype=np.float64), equilibrium_gas.Y)
    )

    rhs = make_rhs(gas, pressure_pa)

    sol = solve_ivp(
        rhs,
        t_span=(T_START, t_end),
        y0=y0,
        method=SOLVER_METHOD,
        rtol=RTOL,
        atol=ATOL,
        max_step=MAX_STEP,
        dense_output=True,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    events = detect_case_events(sol, rhs, T0, equilibrium_state)
    T_final = float(sol.y[0, -1])
    delta_T = T_final - T0

    return {
        "T0": float(T0),
        "o2_ch4_ratio": float(o2_ch4_ratio),
        "h2o_ch4_ratio": float(h2o_ch4_ratio),
        "pressure_bar": float(pressure_bar),
        "solution": sol,
        **events,
        "T_final": T_final,
        "delta_T": float(delta_T),
        # NEW (v2): kept for the equilibrium-consistency sanity check --
        # v1 computed this locally but never returned it.
        "equilibrium_state": equilibrium_state,
        "n_internal_steps": int(sol.t.size),
    }


def horizon_is_adequate(case_results, t_end):
    """
    Require:
      - meaningful temperature rise in every case
      - every detected ignition occurs before 90% of the time horizon
      - every case reaches full-state equilibrium with room for a steady tail
    """

    if not case_results:
        return False

    for result in case_results:
        if result["delta_T"] < MIN_TEMPERATURE_RISE_FOR_IGNITION:
            return False

        if result["t_ign"] >= END_GUARD_FRACTION * t_end:
            return False

        if not result["transition_reached"] or not result["equilibrium_reached"]:
            return False

        if result["activity_final"] > EQUILIBRIUM_ACTIVITY_FRACTION:
            return False

        if result["equilibrium_error_final"] > EQUILIBRIUM_STATE_FRACTION:
            return False

        if result["t_equilibrium"] >= EQUILIBRIUM_END_GUARD_FRACTION * t_end:
            return False

    return True


def solve_case_until_adequate(T0, o2_ch4_ratio, pressure_bar,
                              h2o_ch4_ratio=0.0):
    """Solve one case on its own automatically extended physical horizon."""
    t_end = INITIAL_T_END
    last_result = None

    for attempt in range(1, MAX_HORIZON_ATTEMPTS + 1):
        result = solve_case(T0, o2_ch4_ratio, pressure_bar, t_end,
                            h2o_ch4_ratio)
        result["horizon_attempts"] = attempt
        result["case_t_end"] = float(t_end)

        if horizon_is_adequate([result], t_end):
            return result

        last_result = result
        next_t_end = min(t_end * HORIZON_GROWTH, MAX_T_END)
        if next_t_end <= t_end:
            break
        t_end = next_t_end

    details = "unknown"
    if last_result is not None:
        details = (
            f"t_ign={last_result['t_ign']:.4e} s, "
            f"t_equilibrium={last_result['t_equilibrium']:.4e} s, "
            f"equilibrium_error={last_result['equilibrium_error_final']:.3e}, "
            f"activity_final={last_result['activity_final']:.3e}"
        )
    raise RuntimeError(
        "Case did not reach verified HP equilibrium by MAX_T_END: "
        f"T0={T0}, O2/CH4={o2_ch4_ratio}, P={pressure_bar} bar; {details}."
    )


# =============================================================================
# CASE-SPECIFIC HYBRID SAMPLING
# =============================================================================

def case_time_anchors(result):
    """Return strictly increasing physical-time anchors for one trajectory."""
    anchors = np.array(
        [
            result["window_start"],
            result["t_onset"],
            result["t_ign"],
            result["t_transition_end"],
            result["t_equilibrium"],
            result["window_end"],
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(anchors)):
        raise RuntimeError(
            "Detected event times are not finite for "
            f"T0={result['T0']}, O2/CH4={result['o2_ch4_ratio']}, "
            f"P={result['pressure_bar']}. "
            f"Anchors: {anchors}"
        )

    # Exact equality can occur when two thresholds are crossed within one very
    # small adaptive-solver step. Repair only such numerical degeneracy; a
    # genuinely reversed event order remains an error.
    span = anchors[-1] - anchors[0]
    tolerance = max(1e-12 * max(abs(anchors[-1]), 1.0), np.finfo(float).eps)
    if span <= 0.0 or np.any(np.diff(anchors) < -tolerance):
        raise RuntimeError(
            "Detected events are out of physical order for "
            f"T0={result['T0']}, O2/CH4={result['o2_ch4_ratio']}, "
            f"P={result['pressure_bar']}. "
            f"Anchors: {anchors}"
        )

    minimum_gap = max(np.spacing(max(abs(anchors[-1]), 1.0)), 1e-12 * span)
    for i in range(1, len(anchors) - 1):
        anchors[i] = max(anchors[i], anchors[i - 1] + minimum_gap)
    for i in range(len(anchors) - 2, 0, -1):
        anchors[i] = min(anchors[i], anchors[i + 1] - minimum_gap)

    if not np.all(np.diff(anchors) > 0.0):
        raise RuntimeError("Could not make event anchors strictly increasing.")
    return anchors


def exponential_idle_times(t_start, t_onset, n_points):
    """Idle-phase sample times, from the initial condition to ignition onset.

    With IDLE_LOG_UNIFORM the samples are geometric in time, which is uniform in
    the log-time coordinate the surrogate integrates in. The radical pool grows
    like a power law here, so equal resolution per decade is what the model
    needs; concentrating points near onset instead leaves the early decades
    almost unsupervised.
    """
    if n_points < 1 or not t_start < t_onset:
        raise ValueError("Idle interval and point count must be positive.")

    first_positive = max(
        np.nextafter(max(t_start, 0.0), t_onset),
        t_start + IDLE_FIRST_POSITIVE_FRACTION_OF_ONSET * (t_onset - t_start),
    )

    if IDLE_LOG_UNIFORM:
        # t_start (the exact initial condition) plus a geometric sweep up to,
        # but not including, onset -- onset itself belongs to the arc region.
        return np.concatenate((
            np.array([t_start], dtype=np.float64),
            np.geomspace(first_positive, t_onset, n_points - 1,
                         endpoint=False, dtype=np.float64),
        ))

    n_flat = int(N_FLAT_IDLE)
    if not 2 <= n_flat <= n_points - 2:
        raise ValueError("N_FLAT_IDLE must leave at least two onset-focused points.")
    baseline_fraction = float(IDLE_BASELINE_END_FRACTION_OF_ONSET)
    if not 0.0 < baseline_fraction < 1.0:
        raise ValueError("Idle baseline-end fraction must lie between zero and one.")
    baseline_end = t_start + baseline_fraction * (t_onset - t_start)
    baseline = np.concatenate((
        np.array([t_start], dtype=np.float64),
        np.geomspace(first_positive, baseline_end, n_flat - 1,
                     endpoint=False, dtype=np.float64),
    ))
    strength = float(IDLE_TIME_WARP_STRENGTH)
    if strength <= 0.0:
        raise ValueError("IDLE_TIME_WARP_STRENGTH must be positive.")
    u = np.linspace(0.0, 1.0, n_points - n_flat, endpoint=False, dtype=np.float64)
    onset_focused = 1.0 - np.expm1(strength * (1.0 - u)) / np.expm1(strength)
    departure = baseline_end + (t_onset - baseline_end) * onset_focused
    return np.concatenate((baseline, departure))


def _largest_arc_intervals(arc_end, mandatory_arc, n_points):
    """Fill the largest 1-D arc-length gaps while preserving exact events."""
    if arc_end <= 0.0:
        raise ValueError("Arc length must be positive.")

    tolerance = 1e-12 * arc_end
    selected = []
    for value in sorted(float(v) for v in mandatory_arc):
        value = min(max(value, 0.0), np.nextafter(arc_end, 0.0))
        if not selected or value - selected[-1] > tolerance:
            selected.append(value)

    if not selected or selected[0] > tolerance:
        selected.insert(0, 0.0)
    if len(selected) > n_points:
        raise RuntimeError("More mandatory arc points than the dynamic point budget.")

    # The right boundary is a sentinel: it closes the final gap but is not
    # selected because equilibrium belongs to the steady-tail allocation.
    boundaries = selected + [arc_end]
    heap = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        heapq.heappush(heap, (-(right - left), left, right))

    while len(selected) < n_points:
        neg_width, left, right = heapq.heappop(heap)
        midpoint = 0.5 * (left + right)
        selected.append(midpoint)
        heapq.heappush(heap, (-(midpoint - left), left, midpoint))
        heapq.heappush(heap, (-(right - midpoint), midpoint, right))

    return np.sort(np.asarray(selected, dtype=np.float64))


def species_aware_arc_times(result, t_start, t_end, n_points, extra_forced_times=None):
    """
    Select reaction/relaxation times from continuous normalized state arc.

    Unlike the notebook reference, this includes every species with a
    meaningful excursion, does not snap targets to pre-existing rows, and
    forces the detected ignition peak, temperature peak, and transition end
    into the selected set. extra_forced_times (optional) are unioned into
    the same mandatory-point mechanism -- used to guarantee tail
    sub-window checkpoints land on exact, arc-length-consistent samples.
    """
    if n_points < 1 or not t_start < t_end:
        raise ValueError("Dynamic interval and point count must be positive.")

    sol = result["solution"]
    adaptive = np.asarray(sol.t, dtype=np.float64)
    adaptive = adaptive[(adaptive > t_start) & (adaptive < t_end)]
    forced_times = np.array(
        [
            t_start,
            result["t_ign"],
            result["t_temperature_peak"],
            result["t_transition_end"],
            t_end,
        ],
        dtype=np.float64,
    )
    if extra_forced_times is not None and len(extra_forced_times) > 0:
        forced_times = np.concatenate(
            (forced_times, np.asarray(extra_forced_times, dtype=np.float64))
        )
    forced_times = forced_times[
        (forced_times >= t_start) & (forced_times <= t_end)
    ]
    log_candidates = np.geomspace(t_start, t_end, N_ARC_CANDIDATES)
    candidate_t = np.unique(
        np.concatenate((forced_times, adaptive, log_candidates))
    )
    candidate_y = np.asarray(sol.sol(candidate_t), dtype=np.float64).T

    T = candidate_y[:, 0]
    T_scaled = (T - T.min()) / max(np.ptp(T), np.finfo(float).eps)
    dT = np.diff(T_scaled)

    Y = candidate_y[:, 1:]
    species_excursion = np.ptp(Y, axis=0)
    active_species = species_excursion >= MIN_SPECIES_EXCURSION_FOR_ARC
    if np.any(active_species):
        # CHANGED: was min-max normalization PER SPECIES
        # (Y - Y.min()) / species_excursion, which scales every active
        # species into its own private [0, 1] range regardless of how
        # large that species' actual excursion is. A trace radical
        # (H/O/OH/HO2) with excursion ~1e-7 got the exact same [0, 1]
        # weight in step_length as CH4 swinging ~0.4 -- so a fast but
        # physically tiny radical-pool wiggle could dominate the arc
        # length and pull most of N_DYNAMIC_ARC into a sub-nanosecond
        # cluster around it. This is the direct cause of the
        # near-duplicate-timestamp warning (sanity check #7).
        # Now every active species shares ONE reference scale (the
        # largest excursion among the active species, i.e. whichever
        # major species swings the most for this case). A species with
        # a much smaller -- but still above-threshold -- excursion
        # contributes proportionally less arc length instead of being
        # blown back up to fill [0, 1] on its own. Combined with raising
        # MIN_SPECIES_EXCURSION_FOR_ARC below, genuinely negligible trace
        # species are excluded entirely; species that are real but minor
        # (e.g. a few-percent intermediate) still participate, just at
        # their true relative weight instead of an inflated one.
        shared_scale = max(
            float(np.max(species_excursion[active_species])),
            np.finfo(float).eps,
        )
        Y_active = Y[:, active_species]
        Y_scaled = (Y_active - Y_active.min(axis=0)) / shared_scale
        species_step_sq = np.mean(np.diff(Y_scaled, axis=0) ** 2, axis=1)
    else:
        species_step_sq = np.zeros(candidate_t.size - 1, dtype=np.float64)

    log_t = np.log(candidate_t)
    log_t_scaled = (log_t - log_t[0]) / max(log_t[-1] - log_t[0], np.finfo(float).eps)
    dlog_t = np.diff(log_t_scaled)

    step_length = np.sqrt(
        ARC_TEMPERATURE_WEIGHT * dT**2
        + ARC_SPECIES_WEIGHT * species_step_sq
        + ARC_LOG_TIME_WEIGHT * dlog_t**2
    )
    cumulative_arc = np.concatenate(([0.0], np.cumsum(step_length)))
    arc_end = float(cumulative_arc[-1])
    if not np.isfinite(arc_end) or arc_end <= 0.0:
        raise RuntimeError("Could not construct a positive state-space arc length.")

    mandatory_times = forced_times[forced_times < t_end]
    mandatory_arc = np.interp(mandatory_times, candidate_t, cumulative_arc)
    selected_arc = _largest_arc_intervals(arc_end, mandatory_arc, n_points)
    selected_t = np.interp(selected_arc, cumulative_arc, candidate_t)

    if selected_t.size != n_points or not np.all(np.diff(selected_t) > 0.0):
        raise RuntimeError("Dynamic arc-length selection is not strictly increasing.")
    return selected_t, int(np.sum(active_species))


def logarithmic_steady_times(t_equilibrium, t_end, n_points):
    """Verified steady-state samples including equilibrium and common t_end."""
    if n_points < 2 or not 0.0 < t_equilibrium < t_end:
        raise ValueError("Steady interval must be positive and contain at least two points.")
    if LOG_SPACE_STEADY_TAIL:
        return np.geomspace(t_equilibrium, t_end, n_points, dtype=np.float64)
    return np.linspace(t_equilibrium, t_end, n_points, dtype=np.float64)


def tail_subwindow_checkpoint_times(t_transition_end, t_equilibrium, n_checkpoints):
    """Strictly-interior log-spaced checkpoints for re-anchoring the tail."""
    if n_checkpoints <= 0:
        return np.array([], dtype=np.float64)
    if not t_transition_end < t_equilibrium:
        # Degenerate ordering for this case (e.g. equilibrium detected
        # essentially at transition_end) -- no room for interior
        # checkpoints, and that's fine, the segment is already short.
        return np.array([], dtype=np.float64)
    # n_checkpoints + 2 points from geomspace, keep only the interior ones
    # so the two endpoints (already anchors) are not duplicated.
    full = np.geomspace(t_transition_end, t_equilibrium, n_checkpoints + 2)
    return full[1:-1]


def select_hybrid_case_times(result):
    """Return exactly N_POINTS_PER_TRAJECTORY case-specific physical times."""
    if N_IDLE + N_DYNAMIC_ARC + N_STEADY != N_POINTS_PER_TRAJECTORY:
        raise ValueError("Hybrid sampling quotas must sum to N_POINTS_PER_TRAJECTORY.")

    anchors = case_time_anchors(result)
    idle = exponential_idle_times(anchors[0], anchors[1], N_IDLE)
    tail_checkpoints = tail_subwindow_checkpoint_times(
        anchors[3], anchors[4], N_TAIL_SUBWINDOW_CHECKPOINTS
    )
    dynamic, n_arc_species = species_aware_arc_times(
        result,
        anchors[1],
        anchors[4],
        N_DYNAMIC_ARC,
        extra_forced_times=tail_checkpoints,
    )
    steady = logarithmic_steady_times(anchors[4], anchors[5], N_STEADY)
    selected = np.concatenate((idle, dynamic, steady))

    if selected.size != N_POINTS_PER_TRAJECTORY:
        raise RuntimeError("Hybrid selector returned the wrong number of points.")
    if not np.all(np.isfinite(selected)) or not np.all(np.diff(selected) > 0.0):
        raise RuntimeError("Hybrid physical-time samples are not finite and increasing.")

    # Locate each checkpoint's index in the FINAL concatenated array.
    # _largest_arc_intervals forces every extra_forced_times value to be
    # an exact member of `dynamic` (mandatory points bypass the greedy
    # bisection), so this is an exact match, not a nearest-neighbor
    # approximation -- searchsorted is safe here.
    if tail_checkpoints.size > 0:
        tail_checkpoint_indices = np.searchsorted(selected, tail_checkpoints)
        if not np.array_equal(selected[tail_checkpoint_indices], tail_checkpoints):
            raise RuntimeError(
                "Tail sub-window checkpoint times were not preserved exactly "
                "in the final sample array -- extra_forced_times was not "
                "honored by species_aware_arc_times."
            )
    else:
        tail_checkpoint_indices = np.array([], dtype=np.int64)

    return selected, n_arc_species, tail_checkpoint_indices


def plot_case_diagnostics(
    result,
    sampled_y,
    sampled_t,
    species_names,
    trajectory_label,
    output_filename,
):
    """Plot temperature and key species for one fully solved trajectory."""
    # Pinned to the actual minimum sampled time (matches the odeint
    # verification plots below) rather than an independent t_onset-based
    # guess. The old t_onset*1e-4 formula could sit ABOVE the smallest
    # positive sampled idle time whenever the idle warp pushed points very
    # close to t=0, making that point render to the left of where dense_t
    # (and therefore the drawn reference line) starts -- i.e. a training
    # sample appearing to fall outside the plotted reference curve, even
    # though the underlying sample and solution were both fine.
    positive_floor = (
        max(float(sampled_t[sampled_t > 0].min()) * 1e-2, 1e-15)
        if np.any(sampled_t > 0)
        else 1e-15
    )
    dense_t = np.unique(
        np.concatenate(
            [
                np.array([positive_floor]),
                np.asarray(result["solution"].t)[
                    np.asarray(result["solution"].t) > 0.0
                ],
                np.geomspace(positive_floor, result["window_end"], 5000),
            ]
        )
    )
    dense_y = np.asarray(result["solution"].sol(dense_t)).T
    plotted_sample_t = np.where(sampled_t > 0.0, sampled_t, positive_floor)
    key_species = ["CH4", "O2", "CO", "H2", "CO2", "H2O"]
    items = [("T", 0)] + [
        (name, 1 + species_names.index(name))
        for name in key_species
        if name in species_names
    ]

    fig, axes = plt.subplots(
        len(items),
        1,
        figsize=(12, 3.2 * len(items)),
        sharex=True,
    )
    if len(items) == 1:
        axes = [axes]
    for ax, (name, column) in zip(axes, items):
        ax.plot(
            dense_t,
            dense_y[:, column],
            lw=1.5,
            label="Dense LSODA reference",
        )
        ax.scatter(
            plotted_sample_t,
            sampled_y[:, column],
            s=14,
            label="Hybrid selected points",
        )
        ax.axvline(
            result["t_ign"],
            ls="--",
            lw=1.2,
            label="Ignition: max(dT/dt)",
        )
        ax.axvline(
            result["t_temperature_peak"],
            color="tab:cyan",
            ls=":",
            lw=1.0,
            label="Temperature peak",
        )
        ax.axvspan(
            result["t_equilibrium"],
            result["window_end"],
            color="tab:blue",
            alpha=0.08,
            label="Verified steady tail",
        )
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_ylabel("Temperature (K)" if name == "T" else f"Y_{name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Physical time (s)")
    fig.suptitle(
        f"{trajectory_label} trajectory — dense reference and training samples\n"
        f"T0={result['T0']:.1f} K | "
        f"O2/CH4={result['o2_ch4_ratio']:.4f} mol/mol | "
        f"P={result['pressure_bar']:.1f} bar(a) | "
        f"t_ign={result['t_ign']:.3e} s | "
        f"t_Tpeak={result['t_temperature_peak']:.3e} s | "
        f"points={N_POINTS_PER_TRAJECTORY} "
        f"({N_IDLE}/{N_DYNAMIC_ARC}/{N_STEADY})"
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_filename, dpi=180, bbox_inches="tight")
    plt.close(fig)




# =============================================================================
# DATASET CHARACTERISATION PLOTS
# =============================================================================
# These plots describe properties of the dataset that determine how the model
# has to be built. Each one answers a specific design question, and together
# they are the evidence for the choices made in the training script.
# =============================================================================


def plot_stiffness_reduction(sampled, sampled_derivatives, physical_times,
                             species_names, output_filename="stiffness_log_time.png"):
    """Dynamic range of the right-hand side in physical time vs log-time.

    Why this matters: the model learns dz/du rather than dz/dt, where
    u ~ log10(t). Rescaling by dt/du = ln(10) * span * (t + delta) collapses the
    dynamic range of the target by many orders of magnitude, which is what makes
    the problem learnable by a bounded network without any stiff solver.
    """
    y = np.clip(sampled[..., 1:], 0.0, None)
    dT = sampled_derivatives[..., :1]
    dY = sampled_derivatives[..., 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        dlogY = dY / ((y + 1e-12) * np.log(10.0))
    d_dt = np.concatenate([dT, dlogY], axis=-1)

    delta = float(np.median([c[c > 0].min() for c in physical_times]))
    s_min = np.log10(delta)
    s_max = np.log10(physical_times.max() + delta)
    dt_du = np.log(10.0) * (s_max - s_min) * (physical_times + delta)
    d_du = d_dt * dt_du[..., None]

    a = np.abs(d_dt[np.isfinite(d_dt) & (d_dt != 0)])
    b = np.abs(d_du[np.isfinite(d_du) & (d_du != 0)])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.logspace(-12, 20, 130)
    axes[0].hist(a, bins=bins, color="crimson", alpha=0.75, label="physical time")
    axes[0].hist(b, bins=bins, color="steelblue", alpha=0.75, label="log-time")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("|d(state)/d(time variable)|")
    axes[0].set_ylabel("count")
    axes[0].set_title("Right-hand side magnitude")
    axes[0].legend()

    axes[1].loglog(physical_times[:, 1:].ravel(),
                   np.abs(d_dt[:, 1:, 1:]).max(axis=-1).ravel(),
                   ".", ms=1, alpha=0.25, color="crimson", label="physical time")
    axes[1].loglog(physical_times[:, 1:].ravel(),
                   np.abs(d_du[:, 1:, 1:]).max(axis=-1).ravel(),
                   ".", ms=1, alpha=0.25, color="steelblue", label="log-time")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("max |d log10 Y / d(time variable)|")
    axes[1].set_title("Where the stiffness lives")
    axes[1].legend(markerscale=8)

    fig.suptitle(
        f"Change of variable removes ~{np.log10(a.max()/b.max()):.0f} decades of "
        f"dynamic range   (max: {a.max():.2e} -> {b.max():.2e})"
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"max_dt": float(a.max()), "max_du": float(b.max())}


def plot_species_dynamic_range(sampled, species_names,
                               output_filename="species_dynamic_range.png"):
    """Range spanned by every species across the dataset.

    Why this matters: species span roughly twelve decades, and the minor
    radicals that control ignition are the smallest ones. On a linear scale the
    loss would be dominated entirely by CH4/O2/N2. This is the justification for
    learning log10(Y) and for placing the representation floor below the
    smallest physically meaningful mass fraction.
    """
    y = np.clip(sampled[..., 1:], 0.0, None)
    lo = np.array([y[..., j][y[..., j] > 0].min() if (y[..., j] > 0).any() else np.nan
                   for j in range(y.shape[-1])])
    hi = y.reshape(-1, y.shape[-1]).max(axis=0)
    order = np.argsort(hi)
    names = [species_names[j] for j in order]

    fig, ax = plt.subplots(figsize=(11, 0.28 * len(names) + 2.2))
    ypos = np.arange(len(names))
    ax.hlines(ypos, lo[order], hi[order], color="steelblue", lw=3, alpha=0.8)
    ax.plot(hi[order], ypos, "o", ms=4, color="darkblue", label="max")
    ax.plot(lo[order], ypos, "o", ms=4, color="crimson", label="min (non-zero)")
    ax.axvline(1e-12, color="k", ls="--", lw=1, label="representation floor 1e-12")
    ax.set_xscale("log")
    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("mass fraction")
    ax.set_title("Dataset span of every species\n"
                 "(the species controlling ignition are the smallest ones)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_radical_pool(final_results, sampled, physical_times, species_names,
                      case_indices, radicals=("H", "O", "OH", "HO2", "H2O2",
                                              "CH3", "CH2O", "C2H6"),
                      output_filename="radical_pool_growth.png"):
    """Radical build-up during induction, on log-log axes.

    Why this matters: the induction period is where the chain-branching pool
    grows by many decades while temperature is still essentially flat. Nothing
    visible happens in T, but everything that sets the ignition delay happens
    here. It is also where the model's remaining error concentrates, so it is
    worth showing explicitly rather than hiding inside a temperature plot.
    """
    present = [s for s in radicals if s in species_names]
    fig, axes = plt.subplots(1, len(case_indices),
                             figsize=(5.0 * len(case_indices), 4.4), squeeze=False)
    for ax, ci in zip(axes[0], case_indices):
        t = physical_times[ci]
        t_ign = final_results[ci]["t_ign"]
        for s in present:
            j = species_names.index(s)
            ax.loglog(t, np.clip(sampled[ci, :, 1 + j], 1e-16, None), lw=1.3, label=s)
        ax.axvline(t_ign, color="k", ls="--", lw=1.2)
        ax.text(t_ign, ax.get_ylim()[1], " ignition", fontsize=8,
                va="top", rotation=90)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("mass fraction")
        ax.set_title(f"case {ci}: T0={final_results[ci]['T0']:.0f} K, "
                     f"P={final_results[ci]['pressure_bar']:.0f} bar")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.suptitle("Radical pool during induction -- several decades of growth "
                 "before temperature moves at all")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ignition_delay_map(final_results, output_filename="ignition_delay_map.png"):
    """Ignition delay against each design variable.

    Why this matters: the delay varies by orders of magnitude across the design,
    and it is controlled mainly by initial temperature. A single surrogate has to
    cover all of it, which is why (pressure, T0, O2/CH4) are supplied to the
    network as conditioning inputs rather than training one model per condition.
    """
    T0 = np.array([r["T0"] for r in final_results])
    ratio = np.array([r["o2_ch4_ratio"] for r in final_results])
    P = np.array([r["pressure_bar"] for r in final_results])
    t_ign = np.array([r["t_ign"] for r in final_results])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    sc = axes[0].scatter(1000.0 / T0, t_ign, c=P, cmap="viridis", s=16)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("1000 / T0  (1/K)"); axes[0].set_ylabel("ignition delay (s)")
    axes[0].set_title("Arrhenius view (colour = pressure)")
    fig.colorbar(sc, ax=axes[0], label="P (bar)")

    sc = axes[1].scatter(ratio, t_ign, c=T0, cmap="plasma", s=16)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("O2 / CH4"); axes[1].set_ylabel("ignition delay (s)")
    axes[1].set_title("Stoichiometry (colour = T0)")
    fig.colorbar(sc, ax=axes[1], label="T0 (K)")

    axes[2].hist(np.log10(t_ign), bins=40, color="steelblue", alpha=0.85)
    axes[2].set_xlabel("log10 ignition delay (s)"); axes[2].set_ylabel("cases")
    axes[2].set_title(f"Spread: {t_ign.min():.2e} to {t_ign.max():.2e} s "
                      f"({np.log10(t_ign.max()/t_ign.min()):.1f} decades)")

    for a in axes[:2]:
        a.grid(alpha=0.3)
    fig.suptitle("Ignition delay across the design -- the range one surrogate "
                 "has to cover")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sampling_density(physical_times, output_filename="sampling_density.png"):
    """How the samples are distributed in time and in log-time.

    Why this matters: samples are placed logarithmically, so they are roughly
    uniform in log-time and extremely non-uniform in physical time. Any training
    scheme that resamples onto a uniform grid in physical time would throw away
    the early decades entirely. The integrator therefore steps on the original
    sample grid, with per-case non-uniform step sizes.
    """
    delta = float(np.median([c[c > 0].min() for c in physical_times]))
    s_min = np.log10(delta)
    s_max = np.log10(physical_times.max() + delta)
    u = (np.log10(physical_times + delta) - s_min) / (s_max - s_min)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(physical_times.ravel(), bins=60, color="crimson", alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("samples")
    axes[0].set_title("Samples in physical time\n(everything piles up at the end)")

    axes[1].hist(u.ravel(), bins=60, color="steelblue", alpha=0.85)
    axes[1].set_xlabel("u = normalised log-time"); axes[1].set_ylabel("samples")
    axes[1].set_title("Samples in log-time\n(usable resolution everywhere)")

    du = np.diff(u, axis=1).ravel()
    dt = np.diff(physical_times, axis=1).ravel()
    axes[2].hist(np.log10(dt[dt > 0]), bins=60, color="crimson",
                 alpha=0.6, label="log10 dt (s)")
    axes[2].hist(np.log10(du[du > 0]), bins=60, color="steelblue",
                 alpha=0.6, label="log10 du")
    axes[2].set_xlabel("log10 step size"); axes[2].set_ylabel("count")
    axes[2].set_title("Step sizes: %.1f decades in t, %.1f in u"
                      % (np.ptp(np.log10(dt[dt > 0])), np.ptp(np.log10(du[du > 0]))))
    axes[2].legend(fontsize=8)

    fig.suptitle(f"Sampling density   (u spans {s_max - s_min:.2f} decades of time)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_first_interval_jump(sampled, physical_times, species_names,
                             output_filename="first_interval_jump.png"):
    """Size of the jump in log10(Y) across the very first sampled interval.

    Why this matters: at t = 0 the mixture contains no radicals at all, so
    log10(Y) is at the floor for exactly the species that are about to move
    fastest, and Cantera correctly reports their production rate as zero. Across
    the first interval those species climb several decades. No bounded
    right-hand side can cross that, which is why the training script handles this
    one interval with a small learned map and starts the ODE at the second
    sample.
    """
    y = np.clip(sampled[..., 1:], 0.0, None)
    L = np.log10(y + 1e-12)
    jump = np.abs(L[:, 1] - L[:, 0])
    per_case = jump.max(axis=1)
    worst_j = int(np.argmax(jump.max(axis=0)))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    axes[0].hist(per_case, bins=40, color="steelblue", alpha=0.85)
    axes[0].set_xlabel("largest jump in log10(Y) over the first interval (decades)")
    axes[0].set_ylabel("cases")
    axes[0].set_title(f"median {np.median(per_case):.2f}, max {per_case.max():.2f} decades")

    med = np.median(jump, axis=0)
    order = np.argsort(med)[::-1][:15]
    axes[1].barh([species_names[j] for j in order][::-1], med[order][::-1],
                 color="steelblue", alpha=0.85)
    axes[1].set_xlabel("median jump over the first interval (decades)")
    axes[1].set_title(f"worst species: {species_names[worst_j]}")
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle("The first sampled interval is a genuine singularity, "
                 "not a small step")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"median_decades": float(np.median(per_case)),
            "max_decades": float(per_case.max())}


def plot_constant_species(sampled, species_names,
                          output_filename="constant_species.png"):
    """log10 range of every species across the dataset.

    Why this matters: the training script drops species whose log10 range is
    essentially zero, since they carry no signal but would still take a share of
    the loss. This plot shows the separation is unambiguous -- inert diluents sit
    at exactly zero range, and every reacting species spans several decades, so
    the threshold is not a tuned parameter.
    """
    L = np.log10(np.clip(sampled[..., 1:], 0.0, None) + 1e-12)
    rng = np.ptp(L.reshape(-1, L.shape[-1]), axis=0)
    order = np.argsort(rng)

    fig, ax = plt.subplots(figsize=(11, 0.28 * len(species_names) + 2.2))
    colors = ["crimson" if rng[j] < 1e-8 else "steelblue" for j in order]
    ax.barh(np.arange(len(order)), np.maximum(rng[order], 1e-10), color=colors,
            alpha=0.85)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([species_names[j] for j in order], fontsize=7)
    ax.set_xscale("log")
    ax.axvline(1e-8, color="k", ls="--", lw=1, label="drop threshold")
    ax.set_xlabel("range of log10(Y) across the dataset (decades)")
    n_const = int((rng < 1e-8).sum())
    ax.set_title(f"Species variation -- {n_const} inert channel(s) in red, "
                 f"{len(rng) - n_const} reacting")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"n_constant": n_const,
            "constant_species": [species_names[j] for j in range(len(rng))
                                 if rng[j] < 1e-8]}



def plot_lhs_design(final_results, output_filename="lhs_design_cube.png"):
    """The Latin-hypercube design, over however many dimensions it has.

    Why this matters: a Latin hypercube spreads a fixed budget of cases evenly
    across all axes at once, so every case contributes new information -- unlike
    a factorial grid, which spends most of its cases re-measuring a response
    that is smooth in pressure and stoichiometry. Flat marginals are the visible
    evidence that the stratification held; the pairwise panels show there is no
    hidden correlation between factors.

    With steam enabled the design is four-dimensional, so all six pairwise
    projections are shown rather than three.
    """
    T0 = np.array([r["T0"] for r in final_results])
    ratio = np.array([r["o2_ch4_ratio"] for r in final_results])
    P = np.array([r["pressure_bar"] for r in final_results])
    steam = np.array([r.get("h2o_ch4_ratio", 0.0) for r in final_results])
    has_steam = float(np.ptp(steam)) > 0.0

    src = np.array([r.get("source", "lhs") for r in final_results])
    # "reference" are the benchmark anchor conditions; "lhs_low_o2ch4" is extra
    # LHS in a sub-range, not an anchor -- keep them visually distinct.
    is_ref = src == "reference"
    is_lhs = ~is_ref

    axes_def = [(T0, "T0 (K)"), (ratio, "O2/CH4"), (P, "P (bar)")]
    if has_steam:
        axes_def.append((steam, "H2O/CH4"))

    pairs = [(i, j) for i in range(len(axes_def))
             for j in range(i + 1, len(axes_def))]
    n_panel = 2 + len(pairs)                      # 3-D view + pairs + marginals
    ncol = 4 if has_steam else 3
    nrow = int(np.ceil((n_panel + 1) / ncol))
    fig = plt.figure(figsize=(4.6 * ncol, 4.2 * nrow))

    ax3d = fig.add_subplot(nrow, ncol, 1, projection="3d")
    ax3d.scatter(T0[is_lhs], ratio[is_lhs], P[is_lhs], s=10, alpha=0.55,
                 color="steelblue", label="LHS")
    if is_ref.any():
        ax3d.scatter(T0[is_ref], ratio[is_ref], P[is_ref], s=70,
                     color="crimson", marker="D", label="benchmark")
    ax3d.set_xlabel("T0 (K)"); ax3d.set_ylabel("O2/CH4"); ax3d.set_zlabel("P (bar)")
    ax3d.set_title("Design space (first three axes)")
    ax3d.legend(fontsize=7, loc="upper left")

    for k, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(nrow, ncol, k + 2)
        x, xl = axes_def[i]
        y, yl = axes_def[j]
        ax.scatter(x[is_lhs], y[is_lhs], s=12, alpha=0.55, color="steelblue")
        if is_ref.any():
            ax.scatter(x[is_ref], y[is_ref], s=70, color="crimson", marker="D")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=0.3)
        ax.set_title(f"{yl} vs {xl}")

    ax = fig.add_subplot(nrow, ncol, len(pairs) + 2)
    for (v, lab), c in zip(axes_def,
                           ("steelblue", "seagreen", "darkorange", "purple")):
        vn = (v - v.min()) / max(np.ptp(v), 1e-12)
        ax.hist(vn, bins=10, histtype="step", lw=1.8, color=c, label=lab)
    ax.set_xlabel("normalised parameter value"); ax.set_ylabel("cases")
    ax.set_title("Marginals (flat = even coverage)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(nrow, ncol, len(pairs) + 3)
    ax.axis("off")
    lines = [f"cases: {len(T0)}  ({int(is_lhs.sum())} LHS + "
             f"{int(is_ref.sum())} benchmark)", ""]
    for v, lab in axes_def:
        lines.append(f"{lab:<9} {v.min():.4f} - {v.max():.4f}")
    lines += ["", "These ranges bound what the surrogate can be",
              "expected to interpolate. Anything outside them",
              "is extrapolation."]
    if not has_steam:
        lines += ["", "No steam in this dataset (INCLUDE_STEAM=False),",
                  "so the design is three-dimensional."]
    ax.text(0.0, 0.98, "\n".join(lines), va="top", family="monospace",
            fontsize=9)

    fig.suptitle(f"Latin-hypercube design of experiments "
                 f"({len(axes_def)} dimensions)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_phase_regions(result, sampled_y, sampled_t, species_names,
                       trajectory_label, output_filename):
    """Temperature, major and minor species with the three sampling phases shaded.

    Why this matters: it shows what each sampling phase is actually resolving.
    The idle region carries several decades of radical growth at a nearly flat
    temperature, the ignition region is a sub-microsecond turnover, and the
    steady region is a slow relaxation. Sample counts are annotated per region,
    so any imbalance between where the points are and where the chemistry moves
    is visible directly.
    """
    anchors = case_time_anchors(result)
    t_onset, t_trans, t_eq = float(anchors[1]), float(anchors[3]), float(anchors[4])
    t = np.asarray(sampled_t, dtype=float)
    floor = max(t[t > 0].min() * 0.5, 1e-16)
    tp = np.where(t > 0, t, floor)

    majors = [s for s in ("CH4", "O2", "CO", "CO2", "H2", "H2O") if s in species_names]
    minors = [s for s in ("H", "O", "OH", "HO2", "H2O2", "CH3", "CH2O", "C2H6",
                          "CH2", "HCO") if s in species_names]

    n_idle = int(np.sum(t < t_onset))
    n_dyn = int(np.sum((t >= t_onset) & (t < t_eq)))
    n_steady = int(np.sum(t >= t_eq))

    fig, axes = plt.subplots(4, 3, figsize=(17, 13))

    def shade(a):
        a.axvspan(floor, t_onset, color="steelblue", alpha=0.10)
        a.axvspan(t_onset, t_eq, color="crimson", alpha=0.10)
        a.axvspan(t_eq, tp.max(), color="seagreen", alpha=0.10)
        a.axvline(result["t_ign"], color="k", ls="--", lw=1.0, alpha=0.7)
        a.set_xscale("log")
        a.set_xlim(floor, tp.max())
        a.grid(alpha=0.25)

    # temperature, spanning the whole top row
    for c in range(3):
        axes[0, c].remove()
    ax = fig.add_subplot(4, 3, (1, 3))
    ax.plot(tp, sampled_y[:, 0], "k-", lw=1.8)
    ax.plot(tp, sampled_y[:, 0], ".", ms=3, color="darkred", alpha=0.55)
    shade(ax)
    ax.set_ylabel("T (K)")
    ax.set_title(
        f"{trajectory_label}: T0={result['T0']:.0f} K, "
        f"O2/CH4={result['o2_ch4_ratio']:.3f}, P={result['pressure_bar']:.0f} bar   |   "
        f"idle {n_idle} pts (blue) · ignition {n_dyn} pts (red) · "
        f"steady {n_steady} pts (green) · dots = stored samples"
    )

    # Five majors (the syngas products the surrogate is ultimately judged on)
    # and four minors (the radicals that set the ignition delay).
    slots = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
    chosen = ([s for s in ("CH4", "O2", "CO", "H2", "H2O") if s in majors]
              + [s for s in ("OH", "HO2", "CH2O", "CH3") if s in minors])
    chosen += [s for s in majors + minors if s not in chosen]
    groups = [[s] for s in chosen[:9]]
    for (r, c), grp in zip(slots, groups):
        a = axes[r, c]
        for s in grp:
            j = species_names.index(s)
            a.plot(tp, np.clip(sampled_y[:, 1 + j], 1e-16, None), lw=1.5, label=s)
            a.plot(tp, np.clip(sampled_y[:, 1 + j], 1e-16, None), ".", ms=2.5,
                   alpha=0.5, color="darkred")
        a.set_yscale("log")
        shade(a)
        a.set_ylabel("mass fraction")
        a.set_title(", ".join(grp))
        if r == 3:
            a.set_xlabel("time (s)")
    for (r, c) in slots[len(groups):]:
        axes[r, c].axis("off")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_filename, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {"n_idle": n_idle, "n_ignition": n_dyn, "n_steady": n_steady}


def plot_floor_diagnostic(sampled, species_names,
                          candidate_floors=(1e-16, 1e-14, 1e-12, 1e-10, 1e-8),
                          output_filename="representation_floor.png"):
    """How much real signal each candidate representation floor would discard.

    Why this matters: the surrogate learns log10(Y + floor), so anything below
    the floor is flattened rather than removed. Too high a floor erases genuine
    induction chemistry; too low, and the [-1, 1] scaling spends its range on
    values beneath the solver's own absolute tolerance, which are numerical
    residue rather than chemistry. The right floor is the one that sits at ATOL:
    below that, the reference solve is not resolving the value anyway.
    """
    y = np.clip(sampled[..., 1:], 0.0, None)
    total = y[..., 0].size

    fracs = []
    for f in candidate_floors:
        fracs.append([(y[..., j] < f).sum() / total for j in range(y.shape[-1])])
    fracs = np.array(fracs)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for k, f in enumerate(candidate_floors):
        axes[0].plot(range(len(species_names)), np.sort(fracs[k])[::-1],
                     lw=1.6, label=f"floor {f:.0e}")
    axes[0].set_xlabel("species (sorted)")
    axes[0].set_ylabel("fraction of samples below the floor")
    axes[0].set_title("Signal flattened by each candidate floor")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    k_ref = list(candidate_floors).index(1e-12) if 1e-12 in candidate_floors else 2
    order = np.argsort(fracs[k_ref])[::-1][:20]
    axes[1].barh([species_names[j] for j in order][::-1],
                 fracs[k_ref][order][::-1], color="steelblue", alpha=0.85)
    axes[1].axvline(0.5, color="crimson", ls="--", lw=1,
                    label="half the trajectory flattened")
    axes[1].set_xlabel(f"fraction of samples below {candidate_floors[k_ref]:.0e}")
    axes[1].set_title("Species most affected")
    axes[1].legend(fontsize=8); axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(f"Representation floor -- solver ATOL is {ATOL:.0e}, so values "
                 f"below that are not resolved by the reference solve either")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_filename, dpi=180, bbox_inches="tight")
    plt.close(fig)

    at_atol = np.array([(y[..., j] < ATOL).sum() / total
                        for j in range(y.shape[-1])])
    return {"recommended_floor": float(ATOL),
            "median_fraction_below": float(np.median(at_atol)),
            "worst_species": species_names[int(np.argmax(at_atol))],
            "worst_fraction": float(at_atol.max())}


def run_dataset_characterisation(final_results, sampled, sampled_derivatives,
                                 physical_times, species_names, case_indices):
    """All characterisation plots, plus a short printed summary."""
    print()
    print("=" * 88)
    print("DATASET CHARACTERISATION")
    print("=" * 88)

    plot_lhs_design(final_results)

    stiff = plot_stiffness_reduction(sampled, sampled_derivatives,
                                     physical_times, species_names)
    print(f"  RHS dynamic range : {stiff['max_dt']:.3e} (physical time) "
          f"-> {stiff['max_du']:.3e} (log-time)")

    plot_species_dynamic_range(sampled, species_names)
    plot_radical_pool(final_results, sampled, physical_times, species_names,
                      case_indices)
    plot_ignition_delay_map(final_results)
    plot_sampling_density(physical_times)

    jump = plot_first_interval_jump(sampled, physical_times, species_names)
    print(f"  first interval    : median {jump['median_decades']:.2f}, "
          f"max {jump['max_decades']:.2f} decades")

    floor = plot_floor_diagnostic(sampled, species_names)
    print(f"  floor (= ATOL)    : {floor['recommended_floor']:.0e}; median "
          f"{100*floor['median_fraction_below']:.1f}% of samples below it, "
          f"worst {floor['worst_species']} at "
          f"{100*floor['worst_fraction']:.1f}%")

    for ci, lab in zip(case_indices, ("random", "fastest", "slowest")):
        counts = plot_phase_regions(final_results[ci], sampled[ci],
                                    physical_times[ci], species_names,
                                    f"{lab} case", f"phase_regions_{lab}.png")
    print(f"  sample split      : idle {counts['n_idle']}, ignition "
          f"{counts['n_ignition']}, steady {counts['n_steady']} "
          f"(last plotted case)")

    const = plot_constant_species(sampled, species_names)
    print(f"  inert channels    : {const['n_constant']} "
          f"{const['constant_species']}")

    t_ign = np.array([r["t_ign"] for r in final_results])
    print(f"  ignition delay    : {t_ign.min():.3e} to {t_ign.max():.3e} s "
          f"({np.log10(t_ign.max() / t_ign.min()):.1f} decades)")
    print(f"  plots             : {PLOTS_DIR}")


# =============================================================================
# DATASET SANITY CHECKS
# =============================================================================
# Every check below prints its own PASS/FAIL line and appends to `failures`
# on failure. This function raises at the end if ANY check failed, so a
# dataset that doesn't pass never silently reaches the training pipeline --
# the whole point, per this session's discussion, is that checking is much
# cheaper than training on data with an undetected problem.

def run_dataset_sanity_checks(
    sampled,
    sampled_derivatives,
    physical_times,
    final_results,
    species_names,
    case_P,
):
    failures = []
    warnings_list = []

    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
        if not condition:
            failures.append(name)

    def warn(name, condition, detail=""):
        status = "OK" if condition else "WARN"
        print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
        if not condition:
            warnings_list.append(name)

    print()
    print("=" * 88)
    print("DATASET SANITY CHECKS")
    print("=" * 88)

    n_cases, n_time, state_dim = sampled.shape

    # 1. Finiteness of every saved array.
    check(
        "1. States finite",
        bool(np.all(np.isfinite(sampled))),
    )
    check(
        "2. Derivatives finite",
        bool(np.all(np.isfinite(sampled_derivatives))),
    )
    check(
        "3. Sample times finite and strictly increasing per case",
        bool(np.all(np.isfinite(physical_times)))
        and bool(np.all(np.diff(physical_times, axis=1) > 0.0)),
    )

    # 2. Physical bounds on species mass fractions.
    species = sampled[:, :, 1:]
    check(
        "4. Species mass fractions in [0, 1]",
        bool(species.min() >= -1e-12) and bool(species.max() <= 1.0 + 1e-9),
        f"observed range [{species.min():.3e}, {species.max():.3e}]",
    )

    # 3. Species mass fractions sum to 1 at every point.
    species_sum_error = np.abs(species.sum(axis=2) - 1.0)
    check(
        "5. Species mass fractions sum to 1.0 at every sampled point",
        bool(species_sum_error.max() <= SPECIES_SUM_TOLERANCE),
        f"max |sum(Y) - 1| = {species_sum_error.max():.3e} "
        f"(tolerance {SPECIES_SUM_TOLERANCE:.0e})",
    )

    # 4. Mass conservation of the saved derivatives: sum(dY/dt) must be ~0
    # at every point, since Cantera's net_production_rates * MW / density
    # formula conserves total mass by construction if computed correctly.
    dY_sum = sampled_derivatives[:, :, 1:].sum(axis=2)
    check(
        "6. sum(dY/dt) ~ 0 at every point (mass conservation)",
        bool(np.abs(dY_sum).max() <= MASS_CONSERVATION_TOLERANCE),
        f"max |sum(dY/dt)| = {np.abs(dY_sum).max():.3e} "
        f"(tolerance {MASS_CONSERVATION_TOLERANCE:.0e})",
    )

    # 5. Timestamp-gap distribution -- the check that surfaced the
    # near-duplicate-timestamp / oversampling-near-t0 issue this session.
    # Made permanent here so it is caught automatically going forward.
    dt = np.diff(physical_times, axis=1)
    degenerate_fraction_per_case = (dt < DEGENERATE_GAP_SECONDS).mean(axis=1)
    n_flagged_cases = int((degenerate_fraction_per_case > DEGENERATE_GAP_FRACTION_WARN_THRESHOLD).sum())
    warn(
        "7. Fraction of cases with >"
        f"{DEGENERATE_GAP_FRACTION_WARN_THRESHOLD:.0%} of gaps below "
        f"{DEGENERATE_GAP_SECONDS:.0e}s",
        n_flagged_cases == 0,
        f"{n_flagged_cases}/{n_cases} cases flagged | "
        f"median fraction across all cases = {np.median(degenerate_fraction_per_case):.1%} | "
        f"worst case = {degenerate_fraction_per_case.max():.1%}. "
        "This does not fail the run (v1's dataset already exhibits this and "
        "is usable), but a high fraction here means a meaningful share of "
        "the point budget is resolving sub-microsecond dynamics -- worth "
        "reviewing the arc-length sampler's species-excursion normalization "
        "(MIN_SPECIES_EXCURSION_FOR_ARC) if this number looks large."
    )

    # 6. Event-anchor ordering, per case (case_time_anchors already enforces
    # this at generation time and would have raised -- this re-verifies it
    # holds on the FINAL saved anchors as a defense-in-depth check).
    anchor_array = np.array([case_time_anchors(r) for r in final_results])
    check(
        "8. Event anchors strictly increasing for every case",
        bool(np.all(np.diff(anchor_array, axis=1) > 0.0)),
    )

    # 7. Equilibrium consistency: the saved final sampled state should be
    # close to the independently-computed HP-equilibrium target.
    final_state = sampled[:, -1, :]
    equilibrium_targets = np.array([r["equilibrium_state"] for r in final_results])
    # Compare temperature in K directly; compare species relative to each
    # component's own excursion so trace species don't dominate the metric.
    temperature_eq_error = np.abs(final_state[:, 0] - equilibrium_targets[:, 0])
    check(
        "9. Final sampled temperature within 5 K of HP-equilibrium target",
        bool(temperature_eq_error.max() <= 5.0),
        f"max |T_final - T_equilibrium| = {temperature_eq_error.max():.3f} K",
    )
    species_eq_error = np.abs(final_state[:, 1:] - equilibrium_targets[:, 1:])
    check(
        "10. Final sampled species within 1e-3 (absolute mass fraction) of HP-equilibrium target",
        bool(species_eq_error.max() <= 1e-3),
        f"max |Y_final - Y_equilibrium| = {species_eq_error.max():.3e}",
    )

    # 8. Finite-difference cross-check of the exact derivative, restricted
    # to points whose neighbors are far enough apart to give a meaningful
    # comparison (see FINITE_DIFF_CHECK_MIN_GAP_SECONDS).
    central_gap = physical_times[:, 2:] - physical_times[:, :-2]
    central_fd = (sampled[:, 2:, :] - sampled[:, :-2, :]) / central_gap[..., None]
    exact_mid = sampled_derivatives[:, 1:-1, :]
    gap_ok = central_gap >= FINITE_DIFF_CHECK_MIN_GAP_SECONDS
    # Broadcast the (case, time) gap mask against the (case, time, state)
    # arrays, AND require both sides to individually clear the magnitude
    # floor -- see FINITE_DIFF_MIN_ABSOLUTE_MAGNITUDE above for why this
    # excludes the exact-zero-turn-on artifact rather than just noise.
    magnitude_ok = np.minimum(np.abs(exact_mid), np.abs(central_fd)) >= FINITE_DIFF_MIN_ABSOLUTE_MAGNITUDE
    safe_mask = gap_ok[:, :, None] & magnitude_ok
    if np.any(safe_mask):
        denom = np.maximum(np.abs(exact_mid), np.abs(central_fd))
        # Points where BOTH sides are exactly 0 (denom == 0) produce a 0/0
        # divide here -- harmless, since those points always fail
        # magnitude_ok above and are dropped by safe_mask a few lines down,
        # but np.errstate keeps the benign RuntimeWarning from printing and
        # worrying anyone reading the console output.
        with np.errstate(invalid="ignore", divide="ignore"):
            relative_error = np.abs(central_fd - exact_mid) / denom
        relative_error_checked = relative_error[safe_mask]
        # Downgraded to a soft warning, not a hard failure: a central
        # finite difference is only ever a coarse approximation near a
        # genuinely sharp transient (which this dataset has by design), so
        # some real disagreement here is expected and is not by itself
        # evidence the exact derivative is wrong -- checks #6 (mass
        # conservation) and #9/#10 (equilibrium consistency) are the
        # decisive correctness checks; this one is a supplementary signal.
        warn(
            "11. Central finite-difference cross-check of exact derivative "
            f"(on {int(safe_mask.sum())} well-resolved points, "
            f"{int((~magnitude_ok & gap_ok[:, :, None]).sum())} near-zero/turn-on points excluded)",
            bool(np.percentile(relative_error_checked, 99.0) <= FINITE_DIFF_RELATIVE_TOLERANCE),
            f"p99 relative error = {np.percentile(relative_error_checked, 99.0):.3f} "
            f"(tolerance {FINITE_DIFF_RELATIVE_TOLERANCE})",
        )
    else:
        warn("11. Central finite-difference cross-check", False, "no points had both a safely large enough gap and non-negligible magnitude to check")

    # 9. Pressure and temperature within configured sampling ranges (a
    # cheap check that the LHS design and reference cases actually landed
    # where configured).
    check(
        "12. All case pressures within configured LHS range",
        bool(case_P.min() >= PRESSURE_SAMPLE_LOWER_BAR - 1e-6)
        and bool(case_P.max() <= PRESSURE_SAMPLE_UPPER_BAR + 1e-6),
        f"observed [{case_P.min():.2f}, {case_P.max():.2f}] bar vs "
        f"configured [{PRESSURE_SAMPLE_LOWER_BAR:.2f}, {PRESSURE_SAMPLE_UPPER_BAR:.2f}] bar",
    )

    print()
    if failures:
        print(f"SANITY CHECKS: {len(failures)} FAILED -- {failures}")
        raise RuntimeError(
            f"Dataset failed sanity checks: {failures}. Refusing to treat "
            "this as a clean dataset -- fix the underlying issue rather "
            "than silently proceeding."
        )
    if warnings_list:
        print(f"SANITY CHECKS: all critical checks passed, {len(warnings_list)} soft warning(s): {warnings_list}")
    else:
        print("SANITY CHECKS: all checks passed cleanly.")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Steam adds a fourth design dimension. The LHS is built over all of them
    # jointly so the stratification property still holds; with steam disabled
    # the fourth column is simply zero and the design is identical to before.
    n_dim = 4 if INCLUDE_STEAM else 3
    lo = [T_SAMPLE_LOWER, O2_CH4_SAMPLE_LOWER, PRESSURE_SAMPLE_LOWER_BAR]
    hi = [T_SAMPLE_UPPER, O2_CH4_SAMPLE_UPPER, PRESSURE_SAMPLE_UPPER_BAR]
    if INCLUDE_STEAM:
        lo.append(H2O_CH4_SAMPLE_LOWER)
        hi.append(H2O_CH4_SAMPLE_UPPER)
    lhs_sampler = qmc.LatinHypercube(d=n_dim, seed=LHS_SEED)
    lhs_values = qmc.scale(lhs_sampler.random(N_LHS_CASES), lo, hi)
    case_list = [tuple(row) for row in lhs_values]
    case_sources = ["lhs"] * N_LHS_CASES
    ref_cases = [tuple(c) + ((H2O_CH4_LOWER + H2O_CH4_UPPER) / 2.0,)
                 if INCLUDE_STEAM else tuple(c) for c in REFERENCE_CASES]
    case_list.extend(ref_cases)
    case_sources.extend(["reference"] * len(REFERENCE_CASES))

    # NEW: targeted low-O2/CH4 oversampling (see config section above for
    # the full rationale). Same LHS methodology as the main design --
    # jointly stratified over (T, O2/CH4, pressure) -- just with O2/CH4
    # restricted to the narrow, previously under-sampled sub-range.
    extra_lo = [T_SAMPLE_LOWER, O2_CH4_EXTRA_LOWER, PRESSURE_SAMPLE_LOWER_BAR]
    extra_hi = [T_SAMPLE_UPPER, O2_CH4_EXTRA_UPPER, PRESSURE_SAMPLE_UPPER_BAR]
    if INCLUDE_STEAM:
        extra_lo.append(H2O_CH4_SAMPLE_LOWER)
        extra_hi.append(H2O_CH4_SAMPLE_UPPER)
    if N_EXTRA_LOW_O2CH4_CASES > 0:
        extra_lhs_sampler = qmc.LatinHypercube(d=n_dim, seed=EXTRA_LOW_O2CH4_SEED)
        extra_lhs_values = qmc.scale(
            extra_lhs_sampler.random(N_EXTRA_LOW_O2CH4_CASES), extra_lo, extra_hi)
        case_list.extend(tuple(row) for row in extra_lhs_values)
        case_sources.extend(["lhs_low_o2ch4"] * N_EXTRA_LOW_O2CH4_CASES)

    gas_ref = ct.Solution(str(MECHANISM))
    species_names = list(gas_ref.species_names)
    state_columns = ["T"] + species_names

    print("=" * 88)
    print("GRIMECH30_113 — MULTI-PRESSURE FULL-TRAJECTORY DATA GENERATION")
    print("=" * 88)
    print(f"Mechanism : {MECHANISM.resolve()}")
    print(
        f"LHS range : T={T_SAMPLE_LOWER:.2f} to {T_SAMPLE_UPPER:.2f} K | "
        f"O2/CH4={O2_CH4_SAMPLE_LOWER:.6f} to "
        f"{O2_CH4_SAMPLE_UPPER:.6f} mol/mol | "
        f"P={PRESSURE_SAMPLE_LOWER_BAR:.2f} to "
        f"{PRESSURE_SAMPLE_UPPER_BAR:.2f} bar(a)"
    )
    print(f"Fuel      : {FUEL}")
    print(f"Oxidizer  : {OXIDIZER}")
    print(f"Species   : {gas_ref.n_species}")
    print(
        f"Cases     : {N_LHS_CASES} LHS + {len(REFERENCE_CASES)} reference anchors + "
        f"{N_EXTRA_LOW_O2CH4_CASES} targeted low-O2/CH4 LHS = {len(case_list)}"
    )
    print(f"Output    : {EXPERIMENT_ROOT.resolve()}")
    print()

    # Show initial N2/AR concentrations explicitly.
    check_gas, _ = initial_state(
        MECHANISM,
        case_list[0][0],
        case_list[0][1],
        case_list[0][2] * 1e5,
        case_list[0][3] if INCLUDE_STEAM else 0.0,
        N2_CH4_RATIO if INCLUDE_STEAM else 0.0,
    )

    for inert_name in ["N2", "AR"]:
        if inert_name in species_names:
            print(
                f"Initial Y_{inert_name}: "
                f"{check_gas[inert_name].Y[0]:.6e}"
            )

    print()

    # -------------------------------------------------------------------------
    # Solve every trajectory on its own automatically extended horizon.
    # Fast cases therefore keep a millisecond-scale endpoint, while only the
    # chemically slow rich cases extend to seconds. Physical sampling times
    # were already case-specific, so this preserves the dataset contract.
    # -------------------------------------------------------------------------

    final_results = []
    wall_start = time.time()

    print("-" * 88)
    print("Solving cases with individual equilibrium-verified horizons")
    print("-" * 88)

    for case_index, case in enumerate(case_list, start=1):
        T0, o2_ch4_ratio, P = case[0], case[1], case[2]
        h2o_ch4_ratio = float(case[3]) if INCLUDE_STEAM else 0.0
        result = solve_case_until_adequate(T0, o2_ch4_ratio, P, h2o_ch4_ratio)
        final_results.append(result)
        print(
            f"Case {case_index:4d}/{len(case_list)}: "
            f"T0={T0:7.1f} K | O2/CH4={o2_ch4_ratio:.5f} | "
            f"P={P:5.1f} bar | "
            f"t_ign={result['t_ign']:.4e} s | "
            f"t_eq={result['t_equilibrium']:.4e} s | "
            f"t_end={result['case_t_end']:.1e} s | "
            f"attempts={result['horizon_attempts']} | "
            f"steps={result['n_internal_steps']}"
        )

    elapsed = time.time() - wall_start
    case_end_times = np.array(
        [r["case_t_end"] for r in final_results],
        dtype=np.float64,
    )
    print(f"All {len(final_results)} cases solved in {elapsed:.1f} s.")

    # -------------------------------------------------------------------------
    # Build one shared sample-index coordinate; physical times are case-specific.
    # -------------------------------------------------------------------------

    t_ignitions = np.array(
        [r["t_ign"] for r in final_results],
        dtype=np.float64,
    )

    sample_phase = np.linspace(
        0.0,
        1.0,
        N_POINTS_PER_TRAJECTORY,
        dtype=np.float64,
    )

    print()
    print("=" * 88)
    print("CASE-SPECIFIC HYBRID SAMPLING")
    print("=" * 88)
    print(
        f"Case t_end range       : "
        f"[{case_end_times.min():.6e}, {case_end_times.max():.6e}] s"
    )
    print(f"Min ignition time      : {t_ignitions.min():.6e} s")
    print(f"Max ignition time      : {t_ignitions.max():.6e} s")
    print(f"Points per trajectory  : {N_POINTS_PER_TRAJECTORY}")
    print(f"Idle/induction points  : {N_IDLE}")
    print(f"Dynamic arc points     : {N_DYNAMIC_ARC}")
    print(f"Steady-state points    : {N_STEADY}")

    # -------------------------------------------------------------------------
    # Evaluate every dense solve on its own physical-time map.
    # -------------------------------------------------------------------------

    sampled = np.empty(
        (
            len(final_results),
            N_POINTS_PER_TRAJECTORY,
            1 + gas_ref.n_species,
        ),
        dtype=np.float64,
    )
    # NEW (v2): exact derivatives [dT/dt, dY_1/dt, ..., dY_N/dt] at every
    # saved sample point, same shape as `sampled`. Computed by calling the
    # SAME rhs() function used for the reference solve, directly at each
    # saved state -- not a finite difference of the (irregularly spaced,
    # sometimes near-duplicate) saved sample times. See module docstring.
    sampled_derivatives = np.empty_like(sampled)

    case_T0 = np.empty(len(final_results), dtype=np.float64)
    case_o2_ch4 = np.empty(len(final_results), dtype=np.float64)
    case_h2o_ch4 = np.empty(len(final_results), dtype=np.float64)
    case_P = np.empty(len(final_results), dtype=np.float64)  # NEW: per-case pressure
    physical_times = np.empty(
        (len(final_results), N_POINTS_PER_TRAJECTORY),
        dtype=np.float64,
    )
    dt_dphase = np.empty_like(physical_times)
    time_anchors = np.empty((len(final_results), 6), dtype=np.float64)
    n_arc_species = np.empty(len(final_results), dtype=np.int64)
    # NEW: sample-array indices of the extra tail sub-window checkpoints
    # (see N_TAIL_SUBWINDOW_CHECKPOINTS). -1 for a case whose
    # transition_end/equilibrium segment was too short to hold interior
    # checkpoints (tail_subwindow_checkpoint_times returned empty) --
    # filter those out downstream rather than treating -1 as a real index.
    tail_subwindow_indices = np.full(
        (len(final_results), N_TAIL_SUBWINDOW_CHECKPOINTS), -1, dtype=np.int64
    )

    for i, result in enumerate(final_results):
        physical_times[i], n_arc_species[i], case_tail_idx = select_hybrid_case_times(result)
        dt_dphase[i] = np.gradient(physical_times[i], sample_phase, edge_order=1)
        if not np.all(np.isfinite(dt_dphase[i])) or not np.all(dt_dphase[i] > 0.0):
            raise RuntimeError("Invalid dt/dsample_phase produced by hybrid sampling.")
        time_anchors[i] = case_time_anchors(result)
        tail_subwindow_indices[i, : case_tail_idx.size] = case_tail_idx
        sampled[i] = result["solution"].sol(physical_times[i]).T
        case_T0[i] = result["T0"]
        case_o2_ch4[i] = result["o2_ch4_ratio"]
        case_h2o_ch4[i] = result.get("h2o_ch4_ratio", 0.0)
        case_P[i] = result["pressure_bar"]

    # Sanity checks.
    if not np.all(np.isfinite(sampled)):
        raise RuntimeError("Sampled dataset contains NaN/Inf.")

    # FIX: the adaptive solver doesn't strictly enforce positivity, leaving
    # tiny negative species values from floating-point noise (observed:
    # down to -9.4e-08) -- not physical, and left as-is the network would
    # be trained to reproduce negative mass fractions as a real target.
    # Clip only the species columns (index 1: onward) -- never touch T.
    n_negative = int((sampled[:, :, 1:] < 0).sum())
    if n_negative > 0:
        print(f"Clipping {n_negative} tiny negative species values "
              f"(min was {sampled[:, :, 1:].min():.3e}) to zero.")
        sampled[:, :, 1:] = np.clip(sampled[:, :, 1:], 0.0, None)

    # NEW (v2): exact derivative at every saved sample point, computed AFTER
    # the negative-value clip above so the saved state and the saved
    # derivative are evaluated at EXACTLY the same point -- computing this
    # before the clip would evaluate the derivative at a slightly different
    # (pre-clip, floating-point-noise-negative) state than what actually
    # gets saved to sampled_states_physical.npy. A fresh gas/rhs is built
    # per case (cheap -- these are plain function evaluations, no
    # integration) rather than trying to keep the original solve's `gas`
    # object alive, since final_results only retains the dense solution
    # object, not the live Cantera state.
    print()
    print(f"Computing exact derivatives at all {len(final_results)} x {N_POINTS_PER_TRAJECTORY} saved points...")
    _deriv_start = time.time()
    for i, result in enumerate(final_results):
        gas_i = ct.Solution(str(MECHANISM))
        rhs_i = make_rhs(gas_i, result["pressure_bar"] * 1e5)
        for j in range(N_POINTS_PER_TRAJECTORY):
            sampled_derivatives[i, j] = rhs_i(physical_times[i, j], sampled[i, j])
    print(f"  done in {time.time() - _deriv_start:.1f} s")

    species_sums = sampled[:, :, 1:].sum(axis=2)

    print()
    print("Dataset shape           :", sampled.shape)
    print("Temperature range       :", sampled[:, :, 0].min(), "to", sampled[:, :, 0].max())
    print("Pressure range          :", case_P.min(), "to", case_P.max(), "bar")
    print("Species-sum range       :", species_sums.min(), "to", species_sums.max())
    print("Minimum species value   :", sampled[:, :, 1:].min())

    # NEW (v2): comprehensive sanity-check pass. Raises if anything critical
    # fails -- see run_dataset_sanity_checks for the full list of checks.
    run_dataset_sanity_checks(
        sampled,
        sampled_derivatives,
        physical_times,
        final_results,
        species_names,
        case_P,
    )

    # -------------------------------------------------------------------------
    # Save physical dataset and metadata.
    # -------------------------------------------------------------------------

    np.save(DATA_DIR / "sampled_states_physical.npy", sampled)
    # NEW (v2): exact [dT/dt, dY_1/dt, ...] at every saved state, same shape
    # and row/column alignment as sampled_states_physical.npy.
    np.save(DATA_DIR / "sampled_derivatives_physical.npy", sampled_derivatives)
    np.save(DATA_DIR / "time_points_physical.npy", physical_times)
    np.save(DATA_DIR / "time_points_relative_to_ignition.npy", physical_times - t_ignitions[:, None])
    np.save(DATA_DIR / "sample_index_phase.npy", sample_phase)
    np.save(DATA_DIR / "dt_dsample_phase.npy", dt_dphase)
    # Compatibility aliases for training code written against the previous
    # version. The coordinate is now sample-index phase, not ignition phase.
    np.save(DATA_DIR / "ignition_phase_points.npy", sample_phase)
    np.save(DATA_DIR / "dt_dignition_phase.npy", dt_dphase)

    np.savez(
        DATA_DIR / "case_metadata.npz",
        temperatures=case_T0,
        o2_ch4_ratios=case_o2_ch4,
        h2o_ch4_ratios=case_h2o_ch4,
        n2_ch4_ratio=np.array([N2_CH4_RATIO if INCLUDE_STEAM else 0.0]),
        pressures=case_P,  # CHANGED: per-case array instead of a single scalar
        case_sources=np.array(case_sources, dtype=object),
        ignition_times=t_ignitions,
        state_columns=np.array(state_columns, dtype=object),
        species_names=np.array(species_names, dtype=object),
        mechanism=np.array([str(MECHANISM.resolve())], dtype=object),
        case_end_times=case_end_times,
        physical_time_anchors=time_anchors,
        temperature_peak_times=np.array(
            [r["t_temperature_peak"] for r in final_results],
            dtype=np.float64,
        ),
        n_species_used_for_arc=n_arc_species,
        event_names=np.array(
            ["window_start", "ignition_onset", "ignition_peak",
             "transition_end", "equilibrium", "window_end"],
            dtype=object,
        ),
        # NEW: sample-array indices (into the per-case N_POINTS_PER_TRAJECTORY
        # axis of `sampled_states_physical.npy`) of extra checkpoints inside
        # the transition_end -> equilibrium segment, for optionally splitting
        # that one segment into shorter re-anchored training windows. Purely
        # additive -- -1 means "no checkpoint at this slot for this case"
        # (segment too short to hold N_TAIL_SUBWINDOW_CHECKPOINTS interior
        # points); filter those before using. Does not affect
        # physical_time_anchors or anything that already reads it.
        tail_subwindow_sample_indices=tail_subwindow_indices,
    )

    config = {
        "mechanism": str(MECHANISM.resolve()),
        "reference_pressure_absolute_bar": [51.0, 61.0, 71.0],
        "reference_pressure_gauge_bar": [50.0, 60.0, 70.0],
        "pressure_semantics": "absolute_bar_in_solver",
        "lhs_pressure_lower_absolute_bar": PRESSURE_SAMPLE_LOWER_BAR,
        "lhs_pressure_upper_absolute_bar": PRESSURE_SAMPLE_UPPER_BAR,
        "fuel": FUEL,
        "oxidizer": OXIDIZER,
        "reference_T_lower_K": T_LOWER,
        "reference_T_upper_K": T_UPPER,
        "lhs_T_lower_K": T_SAMPLE_LOWER,
        "lhs_T_upper_K": T_SAMPLE_UPPER,
        "mixture_coordinate": "molar_O2_per_molar_CH4",
        "reference_o2_ch4_lower_molar": O2_CH4_LOWER,
        "reference_o2_ch4_upper_molar": O2_CH4_UPPER,
        "lhs_o2_ch4_lower_molar": O2_CH4_SAMPLE_LOWER,
        "lhs_o2_ch4_upper_molar": O2_CH4_SAMPLE_UPPER,
        "range_extension_fraction_each_side": RANGE_EXTENSION_FRACTION,
        "lhs_cases": N_LHS_CASES,
        "lhs_seed": LHS_SEED,
        "reference_anchor_cases": [list(case) for case in REFERENCE_CASES],
        "reference_o2_ch4_ratios_molar": list(REFERENCE_O2_CH4_RATIOS),
        "include_steam": bool(INCLUDE_STEAM),
        "h2o_ch4_sample_range": ([H2O_CH4_SAMPLE_LOWER, H2O_CH4_SAMPLE_UPPER]
                                 if INCLUDE_STEAM else None),
        "n2_ch4_ratio": float(N2_CH4_RATIO) if INCLUDE_STEAM else 0.0,
        "t_range_option": T_RANGE_OPTION,
        "o2_ch4_range_extension_fraction": O2_CH4_RANGE_EXTENSION_FRACTION,
        "n_extra_low_o2ch4_cases": N_EXTRA_LOW_O2CH4_CASES,
        "reference_simplifications": (
            ["pure_CH4_instead_of_natural_gas",
             "O2_CH4_as_O2_per_mass_equivalent_CH4",
             "premixed_0D_reactor_T0_is_reaction_zone_temperature"]
            + ([] if INCLUDE_STEAM else ["no_steam", "no_N2_or_Ar"])),
        "solver_method": SOLVER_METHOD,
        "rtol": RTOL,
        "atol": ATOL,
        "max_step": MAX_STEP,
        "initial_t_end": INITIAL_T_END,
        "max_t_end": MAX_T_END,
        "min_case_t_end": float(case_end_times.min()),
        "max_case_t_end": float(case_end_times.max()),
        "case_specific_end_times": True,
        "sampling_method": "case_specific_species_aware_hybrid_arc_length",
        "n_idle": N_IDLE,
        "n_flat_idle": N_FLAT_IDLE,
        "n_dynamic_arc": N_DYNAMIC_ARC,
        "n_steady": N_STEADY,
        "n_points_per_trajectory": N_POINTS_PER_TRAJECTORY,
        "n_arc_candidates": N_ARC_CANDIDATES,
        "arc_temperature_weight": ARC_TEMPERATURE_WEIGHT,
        "arc_species_weight": ARC_SPECIES_WEIGHT,
        "arc_log_time_weight": ARC_LOG_TIME_WEIGHT,
        "min_species_excursion_for_arc": MIN_SPECIES_EXCURSION_FOR_ARC,
        "ignition_onset_rate_fraction": IGNITION_ONSET_RATE_FRACTION,
        "ignition_transition_activity_fraction": IGNITION_TRANSITION_ACTIVITY_FRACTION,
        "equilibrium_activity_fraction": EQUILIBRIUM_ACTIVITY_FRACTION,
        "equilibrium_state_fraction": EQUILIBRIUM_STATE_FRACTION,
        "pre_ignition_pad_fraction": PRE_IGNITION_PAD_FRACTION,
        "post_equilibrium_pad_fraction": POST_EQUILIBRIUM_PAD_FRACTION,
        "keep_idle_baseline_from_t0": KEEP_IDLE_BASELINE_FROM_T0,
        "idle_time_warp_strength": IDLE_TIME_WARP_STRENGTH,
        "idle_baseline_end_fraction_of_onset": IDLE_BASELINE_END_FRACTION_OF_ONSET,
        "idle_first_positive_fraction_of_onset": IDLE_FIRST_POSITIVE_FRACTION_OF_ONSET,
        "keep_full_steady_tail_to_t_end": KEEP_FULL_STEADY_TAIL_TO_T_END,
        "log_space_steady_tail": LOG_SPACE_STEADY_TAIL,
        "sample_phase_semantics": "normalized_sample_index_only_not_physical_time",
        "min_ignition_time": float(t_ignitions.min()),
        "max_ignition_time": float(t_ignitions.max()),
        # NEW (v2) fields:
        "derivatives_saved": True,
        "derivatives_source": "exact_rhs_evaluation_no_finite_differencing",
        "degenerate_gap_seconds": DEGENERATE_GAP_SECONDS,
        "species_sum_tolerance": SPECIES_SUM_TOLERANCE,
        "mass_conservation_tolerance": MASS_CONSERVATION_TOLERANCE,
        "generator_version": "v2_with_derivatives",
    }

    with open(DATA_DIR / "generation_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # -------------------------------------------------------------------------
    # Diagnostic plots.
    # -------------------------------------------------------------------------

    # CHANGED: color by pressure, so the multi-pressure sweep is visible at a
    # glance -- everything else about this plot is unchanged.
    plt.figure(figsize=(10, 6))

    pressure_norm = (case_P - case_P.min()) / max(case_P.max() - case_P.min(), 1e-12)
    cmap = plt.get_cmap("viridis")

    for i in range(sampled.shape[0]):
        plt.plot(
            sample_phase,
            sampled[i, :, 0],
            linewidth=1.0,
            alpha=0.55,
            color=cmap(pressure_norm[i]),
        )

    plt.axvspan(
        sample_phase[N_IDLE],
        sample_phase[N_IDLE + N_DYNAMIC_ARC - 1],
        alpha=0.15,
        label="Species-aware dynamic arc allocation",
    )

    ignition_sample_phase = np.array(
        [
            np.interp(t_ignitions[i], physical_times[i], sample_phase)
            for i in range(len(final_results))
        ],
        dtype=np.float64,
    )
    plt.scatter(
        ignition_sample_phase,
        np.array(
            [
                result["solution"].sol(result["t_ign"])[0]
                for result in final_results
            ]
        ),
        s=15,
        c="red",
        label="Selected ignition peaks: max(dT/dt)",
        zorder=5,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=case_P.min(), vmax=case_P.max()))
    plt.colorbar(sm, ax=plt.gca(), label="Pressure (bar)")

    plt.xlabel("Normalized sample index (not physical time)")
    plt.ylabel("Temperature (K)")
    plt.title("All trajectories with case-specific hybrid sampling (colored by pressure)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "all_temperature_trajectories_hybrid_sampling.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(10, 5))
    for i in range(len(final_results)):
        positive_time = np.where(physical_times[i] > 0.0, physical_times[i], np.nan)
        plt.plot(
            np.arange(N_POINTS_PER_TRAJECTORY),
            positive_time,
            linewidth=0.7,
            alpha=0.18,
            color=cmap(pressure_norm[i]),
        )
    plt.axvline(N_IDLE, color="tab:orange", ls=":", label="Dynamic arc begins")
    plt.axvline(
        N_IDLE + N_DYNAMIC_ARC,
        color="tab:green",
        ls=":",
        label="Verified steady tail begins",
    )
    plt.yscale("log")
    plt.xlabel("Sample index")
    plt.ylabel("Case-specific physical time (s)")
    plt.title("Physical-time maps created by hybrid sampling")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "hybrid_sampling_physical_time_maps.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    print()
    print("=" * 88)
    print("DATA GENERATION COMPLETE")
    print("=" * 88)
    print(f"Physical data : {DATA_DIR / 'sampled_states_physical.npy'}")
    print(f"Derivatives   : {DATA_DIR / 'sampled_derivatives_physical.npy'} (NEW in v2 -- exact rhs(), no finite differencing)")
    print(f"Physical times: {DATA_DIR / 'time_points_physical.npy'} (one row per case)")
    print(f"Sample phase  : {DATA_DIR / 'sample_index_phase.npy'}")
    print(f"Time scaling  : {DATA_DIR / 'dt_dsample_phase.npy'}")
    print(f"Metadata      : {DATA_DIR / 'case_metadata.npz'}")
    print(f"Plots         : {PLOTS_DIR}")
    print()
    print(
        "Normalization is intentionally NOT done here. "
        "The training script fits normalization on training trajectories only."
    )



    # Individual trajectory plots matching the hybrid arc-length reference
    # style. In addition to one reproducible random case, always show the two
    # ignition-delay extremes so the most difficult temporal cases are easy to
    # inspect without searching the full dataset.
    rng = np.random.default_rng(LHS_SEED)
    random_idx = int(rng.integers(len(final_results)))
    fastest_idx = int(np.argmin(t_ignitions))
    slowest_idx = int(np.argmax(t_ignitions))
    plot_specs = [
        (random_idx, "Random", "random_case_training_samples.png"),
        (fastest_idx, "Fastest ignition", "fastest_ignition_trajectory.png"),
        (slowest_idx, "Slowest ignition", "slowest_ignition_trajectory.png"),
    ]
    for case_index, label, filename in plot_specs:
        plot_case_diagnostics(
            final_results[case_index],
            sampled[case_index],
            physical_times[case_index],
            species_names,
            label,
            filename,
        )

    # Dataset-level characterisation: the properties that determine how the
    # surrogate has to be built (see the plot docstrings for what each shows).
    run_dataset_characterisation(
        final_results,
        sampled,
        sampled_derivatives,
        physical_times,
        species_names,
        [random_idx, fastest_idx, slowest_idx],
    )



if __name__ == "__main__":
    main()


