import numpy as np
import argparse
from controllers.LQG import LQG
from controllers.WDRC import WDRC
from plot import summarize
import os
import pickle
nx=2

I  = np.eye(5)
B = I[3:5,3:5]
print(I)
print(B)