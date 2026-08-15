import numpy as np

idata = model.sample()
# prior predictive after sampling (wrong order)
prior_predictive = np.random.normal(0, 1, size=100)
# numpy rewrite of the likelihood
y_hat = np.random.normal(idata.mu, 1.0)
