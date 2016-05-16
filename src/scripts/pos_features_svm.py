
import pickle
import public_variables
import numpy as np
import util.run_svm

with open(public_variables.POS_FEATURES, "rb") as fin:
    data = pickle.load(fin)

feats = {}

all_pos = data[0]
parts_of_speech = data[1]
for usr, d in parts_of_speech.items():
    y = 1 if d[0]["condition"] == "schizophrenia" else 0
    x = np.zeros(len(all_pos))
    for i, p in enumerate(all_pos):
        if p in d[2]:
            x[i] = d[2][p]
    if np.sum(x) > 0:
        x = x/np.sum(x)
    feats[usr] = {"X":x, "Y":y}

svm = util.run_svm.RunSvm()
svm.run(feats, "../../../output/pos_features_svm.csv")