"""
Trains a word2vec model.
"""

import gensim
import public_variables
import util.reader
import util.tweebo_reader
import util.tweet_cleaner
import sqlite3
import util.random_sentences

print("Training")
sentences =  util.random_sentences.RandomSentences(public_variables.DB_PATH)
model = gensim.models.Word2Vec(sentences, size=public_variables.WORD2VEC_WIDTH, window=5, min_count=5)
print("Saving")
model.save(public_variables.WORD2VEC)