"""
Writes tweets to text for tweebo to tokenize, parse, etc.
Writes out a shell script to run all of the text files.
"""

count=0
import util.reader
import util.tweet_cleaner
import public_variables
import re
import os

files = []
rdr = util.reader.Reader(public_variables.CSV_PATH)

#Write text files
for usr in rdr.all_tweets():
    print("User: %s" % usr["usr"]["anonymized_name"])
    filename = public_variables.TWEEBO_INPUT % usr["usr"]["anonymized_name"]
    files.append(filename)
    with open(filename,"w", encoding="utf8") as fout:
        for tweet in usr["tweets"]:
            fout.write(util.tweet_cleaner.TweetCleaner.clean_whitespace(tweet["text"]))
            count+=1
print("Wrote %i tweets" % count)

#Write shell script
with open("../../../output/run_tweebo.sh","w") as fout:
    for file in files:
        fout.write("./run.sh %s\n" % os.path.basename(file))
