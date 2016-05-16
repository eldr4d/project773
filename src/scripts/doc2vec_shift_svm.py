
import public_variables
import pickle
import csv
import numpy as np
from scipy.stats import ttest_ind
import util.run_svm


with open(public_variables.DOC2VEC_SHIFTS,'rb') as fin:
    feats = pickle.load(fin)

svm = util.run_svm.RunSvm()
svm.run(feats,"../../../output/doc2vec_shift_svm.csv")