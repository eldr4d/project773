
import util.reader
import public_variables
import util.tweebo_reader
import util.tweet_cleaner
import sklearn.cluster
import pickle
import csv

rdr = util.reader.Reader(public_variables.CSV_PATH)
trdr = util.tweebo_reader.TweeboReader(public_variables.TWEEBO_OUTPUT)

with open("../../../output/filter_stats.csv",'w',encoding='utf8',newline='') as fout:
    w=csv.writer(fout)
    w.writerow(["User", "Tweets","Included","Percent"])
    for usr in rdr.read_csv():
        print("User: %s" % usr["anonymized_name"])
        counts=[0,0]
        for tweet in trdr.parsed_tweets(usr["anonymized_name"]):
            counts[0]+=1
            if util.tweet_cleaner.TweetCleaner.should_include(tweet):
                counts[1]+=1
        w.writerow([usr["anonymized_name"],counts[0],counts[1], counts[1]/counts[0]])
