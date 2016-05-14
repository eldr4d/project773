"""
Write example clusters out.
"""

from sklearn import cross_validation
from sklearn import svm
from scipy.stats import ttest_ind
import csv
import public_variables
import util.tweebo_reader
import gensim.models
import numpy as np
import pickle
import re
import itertools

model = gensim.models.Word2Vec.load(public_variables.WORD2VEC)
with open(public_variables.WORD2VEC_KMEANS, "rb") as fin:
    km=pickle.load(fin)

with open("../../../output/w2v_cluster_examples.txt","w", encoding="utf8") as fout:
    for cat in range(public_variables.WORD2VEC_K):
        fout.write("Category %i\n" % cat)
        v=km.cluster_centers_[cat,:]
        print("Searching category %i" % cat)
        top = itertools.islice(sorted(([label, np.sum(np.square(model[label]-v))] for label in model.vocab), key=lambda x:x[1]),0,10)
        print("Sorted")
        for label in top:
            fout.write(label[0]+"\n")
            print(label[0])