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
model = gensim.models.doc2vec.Doc2Vec.load(public_variables.DOC2VEC)
trdr = util.tweebo_reader.TweeboReader(public_variables.TWEEBO_OUTPUT)
with open(public_variables.DOC2VEC_KMEANS, "rb") as fin:
    km=pickle.load(fin)

with open("../../../output/d2v_cluster_examples.txt","w", encoding="utf8") as fout:
    for cat in range(public_variables.DOC2VEC_K):
        fout.write("Category %i\n" % cat)
        v=km.cluster_centers_[cat,:]
        print("Searching category %i" % cat)
        top = itertools.islice(sorted(([label, np.sum(np.square(model.docvecs[label]-v))] for label in model.docvecs.doctags), key=lambda x:x[1]),0,10)
        print("Sorted")
        for label in top:
            fout.write(label[0]+"\n")
            print(label[0])
            m=re.match("(.*)_(0|1)_(\\d+)",label[0])
            usr = m.group(1)
            index = int(m.group(3))
            print("Usr: %s, index: %i" % (usr, index))
            words = list(trdr.parsed_tweets(usr))[index]
            sentence = " ".join(word["clean"] for word in words)
            fout.write(sentence+"\n")
            print(sentence)