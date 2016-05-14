"""
Create a SQLite database with tweets parsed by tweebo.
"""

import sqlite3
import gensim.models
import gensim.models.doc2vec
import public_variables
import util.reader
import util.tweebo_reader
import util.tweet_cleaner

con = sqlite3.connect(public_variables.DB_PATH)
c=con.cursor()

c.execute("drop table tweets")
c.execute("create table tweets(usr TEXT not null, condition int not null, tweet_num int not null, tweet TEXT not null, filtered bit not null)")

rdr = util.reader.Reader(public_variables.CSV_PATH)
trdr = util.tweebo_reader.TweeboReader(public_variables.TWEEBO_OUTPUT)
for usr in rdr.read_csv():
    cond = 0
    if usr["condition"]=="schizophrenia":
        cond=1
    for i, tweet in enumerate(trdr.parsed_tweets(usr['anonymized_name'])):
        text = util.tweet_cleaner.TweetCleaner.word_string(tweet)
        filtered = not util.tweet_cleaner.TweetCleaner.should_include(tweet)
        c.execute("insert into tweets(usr, condition, tweet_num, tweet, filtered) values (?,?,?,?,?)", (usr["anonymized_name"], cond, i, text, filtered))

con.commit()