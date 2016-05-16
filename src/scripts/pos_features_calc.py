"""
Calculate frequency of parts of speech by user.
Frequency is accross tweets.
Alternatively, we might want to calculate frequency within tweets and average that across tweets.
"""


import gensim.models
import gensim.models.doc2vec
import util.reader
import public_variables
import util.tweebo_reader
import util.tweet_cleaner
import sklearn.cluster
import pickle


rdr = util.reader.Reader(public_variables.CSV_PATH)
trdr = util.tweebo_reader.TweeboReader(public_variables.TWEEBO_OUTPUT)

parts_of_speech={}
all_pos = []
for usr in rdr.read_csv():
    print("usr: %s" % usr["anonymized_name"])
    poss = {}
    total = 0
    for tweet in trdr.parsed_tweets(usr["anonymized_name"]):
        if util.tweet_cleaner.TweetCleaner.should_include(tweet):
            for word in tweet:
                pos = word["pos1"]
                if pos not in poss:
                    poss[pos]=0
                poss[pos]+=1
                total+=1
                if pos not in all_pos:
                    all_pos.append(pos)
    parts_of_speech[usr["anonymized_name"]] = [usr, total, poss]

data = [all_pos, parts_of_speech]

with open(public_variables.POS_FEATURES, "wb") as fout:
    pickle.dump(data, fout, 2)

