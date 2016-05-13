"""
Calculate frequency in each cluster by user.
Writes out training features in a hash.
"""

import gensim.models
import gensim.models.doc2vec
import public_variables
import util.reader
import util.tweebo_reader
import util.tweet_cleaner
import sklearn.cluster
import pickle
import csv
import util.tweet_filter
import numpy as np

rdr = util.reader.Reader(public_variables.CSV_PATH)
trdr = util.tweebo_reader.TweeboReader(public_variables.TWEEBO_OUTPUT)
model = gensim.models.doc2vec.Doc2Vec.load(public_variables.DOC2VEC)
with open(public_variables.DOC2VEC_KMEANS, "rb") as fin:
    km=pickle.load(fin)

features = {}

for index, usr in enumerate(rdr.read_csv()):
    print("user: %s" % usr["anonymized_name"])
    cats = np.zeros(public_variables.DOC2VEC_K)
    #for each tweet
    cond = 1 if usr["condition"] == "schizophrenia" else 0
    for i, tweet in enumerate(trdr.parsed_tweets(usr['anonymized_name'])):
        if util.tweet_cleaner.TweetCleaner.should_include(tweet):
            #calculate the cluster
            label = "%s_%i_%i" % (usr["anonymized_name"], cond, i)
            vec = model.docvecs[label]
            cat = km.predict(vec.reshape(1,-1))
            cats[cat] += 1
    if cats.sum() > 0:
        cats = cats / cats.sum()
    features[usr["anonymized_name"]] = {"X":cats, "Y":cond}


with open(public_variables.DOC2VEC_FEATURES,"wb") as fout:
    pickle.dump(features, fout)