import numpy as np
import argparse
from controllers.LQG import LQG
from controllers.WDRC import WDRC
from plot import summarize
import os
import pickle
nx=2

w_max = 0.05*np.ones(nx)
print(w_max)
print(w_max.shape)
w_min = -0.05*np.ones(nx)
print(w_min)
mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
Sigma_w = 1/12*np.diag((w_max - w_min)**2)