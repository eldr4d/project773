"""
Generate clusters from word2vec model.
"""

import gensim.models
import gensim.models.doc2vec
import util.reader
import public_variables
import util.tweebo_reader
import sklearn.cluster
import pickle

model = gensim.models.Word2Vec.load(public_variables.WORD2VEC)
km = sklearn.cluster.KMeans(public_variables.WORD2VEC_K, max_iter=500,n_init=10, verbose=True)
'''
print(dir(model))
help(model)
print(dir(model.vocab))
help(model.vocab)
for v in model.vocab:
    print(v)
'''
mat = model.syn0
print(mat.shape)
km.fit(mat)
with open(public_variables.WORD2VEC_KMEANS, "wb") as fout:
    pickle.dump(km, fout, 2)
