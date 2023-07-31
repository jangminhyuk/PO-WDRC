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
C = -3 + np.random.rand(2,4)*6
print(C)
C1 = np.array([[1,0,0,0],[0,1,0,0]])
print(C1)