"""
Generate clusters from doc2vec model.
"""

import gensim.models
import gensim.models.doc2vec
import util.reader
import public_variables
import util.tweebo_reader
import sklearn.cluster
import pickle

model = gensim.models.doc2vec.Doc2Vec.load(public_variables.DOC2VEC)
km = sklearn.cluster.KMeans(public_variables.DOC2VEC_K, max_iter=50,n_init=3, verbose=True)
mat = model.docvecs.doctag_syn0
print(mat.shape)
km.fit(mat)
with open(public_variables.DOC2VEC_KMEANS, "wb") as fout:
    pickle.dump(km, fout,2)
