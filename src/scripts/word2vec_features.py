"""
Calculate frequency of word2vec cluster by user.
Save pickled features.
"""

import gensim.models
import util.reader
import public_variables
import util.tweebo_reader
import sklearn.cluster
import pickle
import numpy as np

rdr = util.reader.Reader(public_variables.CSV_PATH)
trdr = util.tweebo_reader.TweeboReader(public_variables.TWEEBO_OUTPUT)
model = gensim.models.Word2Vec.load(public_variables.WORD2VEC)
with open(public_variables.WORD2VEC_KMEANS, "rb") as fin:
    km=pickle.load(fin)


features = {}

for index, usr in enumerate(rdr.read_csv()):
    print("user: %s" % usr["anonymized_name"])
    cats = np.zeros(public_variables.WORD2VEC_K)
    #for each tweet
    cond = 1 if usr["condition"] == "schizophrenia" else 0
    for i, tweet in enumerate(trdr.parsed_tweets(usr['anonymized_name'])):
        if util.tweet_cleaner.TweetCleaner.should_include(tweet):
            #calculate the cluster for each word
            for word in tweet:
                if word["clean"] in model:
                    vec = model[word["clean"]]
                    cat = km.predict(vec.reshape(1,-1))
                    cats[cat] += 1
    if cats.sum() > 0:
        cats = cats / cats.sum()
    features[usr["anonymized_name"]] = {"X":cats, "Y":cond}

with open(public_variables.WORD2VEC_FEATURES,"wb") as fout:
    pickle.dump(features, fout,2)