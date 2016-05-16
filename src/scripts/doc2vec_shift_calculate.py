


import gensim.models
import gensim.models.doc2vec
import util.reader
import public_variables
import util.tweebo_reader
import util.tweet_cleaner
import sklearn.cluster
import pickle
import numpy as np

model = gensim.models.doc2vec.Doc2Vec.load(public_variables.DOC2VEC)

rdr = util.reader.Reader(public_variables.CSV_PATH)
trdr = util.tweebo_reader.TweeboReader(public_variables.TWEEBO_OUTPUT)
feats={}
for usr in rdr.read_csv():
    print("User: %s" % usr["anonymized_name"])
    last = None
    shifts = []
    cond = 1 if usr["condition"]=="schizophrenia" else 0
    for index, tweet in enumerate(trdr.parsed_tweets(usr["anonymized_name"])):
        if util.tweet_cleaner.TweetCleaner.should_include(tweet):
            label = "%s_%i_%i" % (usr["anonymized_name"], cond, index)
            if label in model.docvecs:
                vec = model.docvecs[label]
                if last is not None:
                    shift = np.sum(np.square(vec-last))
                    shifts.append(shift)
                last=vec
    if(any(shifts)):
        avg = np.average(shifts)
        feats[usr["anonymized_name"]] = {"X":[avg], "Y":cond, "usr":usr}
        print("Avg shift: %f" % avg)

with open(public_variables.DOC2VEC_SHIFTS,'wb') as fout:
    pickle.dump(feats, fout, 2)