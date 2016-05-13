"""
Reads CSV file and database
"""

import csv
import os.path
import gzip
import json
import pickle
import util.tweet_cleaner

class Reader(object):
    def __init__(self, _file):
        self.file=_file

    def read_csv(self):
        #anonymized_name,condition,age,gender,num_tweets,fold
        with open(self.file,'r') as fin:
            rdr = csv.DictReader(fin)
            for row in rdr:
                yield row

    def user_tweets(self, usr):
        return self.read_tweets(usr['anonymized_name'], usr['condition'])

    def read_tweets(self, anonymized_name, condition):
        dir = "anonymized_%s_tweets" % condition
        filename = "%s.tweets.gz" % anonymized_name
        filepath = os.path.join(os.path.dirname(self.file), dir, filename)
        with gzip.open(filepath, 'rb') as gz:
            for line in gz:
                tweet = json.loads(line.decode("utf-8"))
                for word in tweet:
                    word["clean"] = util.tweet_cleaner.TweetCleaner.clean_word(word["word"])
                yield tweet

    def write_html(self, filepath):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        for usr in self.read_csv():
            filename = "%s-%s.html" % (usr['anonymized_name'], usr['condition'])
            fileout = os.path.join(filepath, filename)
            with open(fileout,'w', encoding="UTF-8") as fout:
                fout.write("<html><body>\n")
                fout.write("<h1>%s (%s)</h1>" % (usr['anonymized_name'], usr['condition']))
                for tweet in self.user_tweets(usr):
                    fout.write("<p><b>%s</b> %s</p>" % (tweet['created_at'], tweet['text']))
                fout.write("</body></html>")

    def all_tweets(self):
        for usr in self.read_csv():
            yield {"usr": usr, "tweets": self.read_tweets(usr['anonymized_name'], usr['condition'])}

    def all_tweet_texts(self):
        for usr in self.read_csv():
            for tweet in self.self.user_tweets(usr):
                yield tweet["text"]

if __name__ == "__main__":
    file = "../../input/anonymized_user_manifest.csv"
    rdr = Reader(file)
    with open("../../../output/tweets.pickle","wb") as fout:
        pickle.dump(list(rdr.all_tweets()),fout)
