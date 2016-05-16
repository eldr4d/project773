
import public_variables
import pickle
import csv
import numpy as np
import util.reader
import util.tweebo_reader
import util.tweet_cleaner
from scipy.stats import ttest_ind

with open(public_variables.WORD2VEC_FEATURES,"rb") as fin:
    feats = pickle.load(fin)


rdr = util.reader.Reader(public_variables.CSV_PATH)
with open(public_variables.WORD2VEC_STATS,"w", encoding="utf8", newline="") as fout:
    w = csv.writer(fout)
    w.writerow(["Category","Control mean", "Control stdev", "Schizophrenia mean", "Schizophrenia stdev", "t","p"])
    for cat in range(public_variables.WORD2VEC_K):
        data = [[],[]]
        for usr, feat in feats.items():
            if usr not in public_variables.SKIP_USERS:
                data[feat["Y"]].append(feat["X"][cat])
        t, p = ttest_ind(data[0], data[1], equal_var=False)
        w.writerow([cat, np.average(data[0]), np.std(data[0]), np.average(data[1]), np.std(data[1]), t, p])

