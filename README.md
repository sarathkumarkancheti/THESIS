# THESIS
# Physics constrained neural network approach for robust learning of chemical kinetics in a high-pressure partial oxidation system
1. We aim to develop a physics constrained neural ODE model by training on physical laws incorporated into loss functions.
2. The reference data is generated using Cantera using different initial conditions oxygen-to-methane ratios(0.45-0.65), temperature(1200-1500K), steam-to-methane ratios(0.25-0.45) and pressures(50-70bar).
3. The first model trained is purely data driven which no physics laws(to evaluate the model whether it's learning dynamics and observe the limitations).
4. Then element conservation is applied as a hard constraint in the model rather than applying it in loss function and fine tune it's weights, this reduces one hyperparameter to tune. Losses of temperature and mass fractions are improved in this model evaluation.
5. In the next step enthalpy conservation is applied in similar way as mass conservation(hard constraint) which further reduced the losses.
6. The remaining step is to train and evaluate the model for a reduced dataset to determine how many data points are needed to accurately represent the dynamics.

#Libraries needed:
1. PyTorch
2. Matplotlib
3. Scipy
4. Cantera
5. pip install cantera torch torchdiffeq matplotlib scipy
