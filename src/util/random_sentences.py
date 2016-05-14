import sqlite3
import gensim.models.doc2vec

class RandomSentences(object):
    def __init__(self,file):
        self.file=file
        self.iteration=0

    def __iter__(self):
        print("Iteration %i" % self.iteration)
        self.iteration+=1
        con = sqlite3.connect(self.file)
        cursor = con.execute("SELECT tweet from tweets where filtered=0 order by random()")
        for row in cursor:
            words = row[0].split()
            yield words
        con.close()

class RandomSentencesLabeled(object):
    def __init__(self,file):
        self.file=file
        self.iteration=0

    def __iter__(self):
        print("Iteration %i" % self.iteration)
        self.iteration+=1
        con = sqlite3.connect(self.file)
        cursor = con.execute("SELECT usr, condition, tweet_num, tweet from tweets where filtered=0 order by random()")
        for row in cursor:
            label = "%s_%s_%i" % (row[0],row[1],row[2])
            words = row[3].split()
            yield gensim.models.doc2vec.TaggedDocument(words, (label,))
        con.close()