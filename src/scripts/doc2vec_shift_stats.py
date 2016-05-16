
import public_variables
import pickle
import csv
import numpy as np
from scipy.stats import ttest_ind


with open(public_variables.DOC2VEC_SHIFTS,'rb') as fin:
    feats = pickle.load(fin)

data=[[],[]]
for usr, d in feats.items():
    data[d["Y"]].append(d["X"])

t, p = ttest_ind(data[0], data[1], equal_var=False)
print("C mean: %f, C std: %f, S mean: %f, S std: %f" % (np.average(data[0]), np.std(data[0]),np.average(data[1]), np.std(data[1])))
print("T: %f, P: %f" % (t,p))