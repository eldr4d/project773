"""
Trains doc2vec model
"""

import public_variables
import sqlite3
import gensim.models
import gensim.models.doc2vec
import util.reader
import util.tweebo_reader
import util.tweet_cleaner
import util.random_sentences

sentences = util.random_sentences.RandomSentencesLabeled(public_variables.DB_PATH)
model = gensim.models.doc2vec.Doc2Vec(size=public_variables.DOC2VEC_WIDTH, window=10, min_count=5,alpha=0.025, min_alpha=0.0025)
print("Building vocab")
model.build_vocab(sentences)
EPOCHS=10
for epoch in range(EPOCHS):
    print("Epoch %d" % epoch)
    model.train(sentences)

model.save(public_variables.DOC2VEC)
