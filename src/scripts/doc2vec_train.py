"""
Trains doc2vec model
"""

import public_variables
import sqlite3
import gensim.models
import gensim.models.doc2vec
import util.reader
import util.config
import util.tweebo_reader
import util.tweet_filter


class MySentences(object):
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

sentences = MySentences(public_variables.DB_PATH)
model = gensim.models.doc2vec.Doc2Vec(size=public_variables.DOC2VEC_WIDTH, window=10, min_count=5,alpha=0.025, min_alpha=0.025)
print("Building vocab")
model.build_vocab(sentences)
EPOCHS=10
for epoch in range(EPOCHS):
    print("Epoch %d" % epoch)
    model.train(sentences)

model.save(public_variables.DOC2VEC)
