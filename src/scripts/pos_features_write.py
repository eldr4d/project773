"""
Prints first 10 examples of each part of speech
"""
import public_variables
import pickle
import csv
import numpy as np
import util.reader
import util.tweebo_reader
from scipy.stats import ttest_ind

def get_examples(pos, rdr, trdr):
    examples = []
    for usr in rdr.read_csv():
        if usr["anonymized_name"] not in public_variables.SKIP_USERS:
            for tweet in trdr.parsed_tweets(usr["anonymized_name"]):
                for word in tweet:
                    if(word["pos1"]==pos):
                        examples.append(word)
                        if len(examples) >= 10:
                            return examples
    return examples


with open(public_variables.POS_FEATURES, "rb") as fin:
    data = pickle.load(fin)


all_pos = data[0]
parts_of_speech = data[1]
rdr = util.reader.Reader(public_variables.CSV_PATH)
trdr = util.tweebo_reader.TweeboReader(public_variables.TWEEBO_OUTPUT)

with open(public_variables.POS_EXAMPLES, "w", encoding="utf8") as fout:
    for pos in all_pos:
        print("POS: %s" % pos)
        fout.write("POS: %s\n" % pos)
        count = 0
        for example in get_examples(pos, rdr, trdr):
            print(example["word"])
            fout.write(example["word"]+"\n")
