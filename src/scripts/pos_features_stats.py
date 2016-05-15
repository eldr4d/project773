
import public_variables
import pickle
import csv
import numpy as np
from scipy.stats import ttest_ind

with open(public_variables.POS_FEATURES, "rb") as fin:
    data = pickle.load(fin)

all_pos = data[0]
parts_of_speech = data[1]

with open(public_variables.POS_STATS,"w", encoding="utf8", newline="") as fout:
    w = csv.writer(fout)
    w.writerow(["POS","Control mean", "Control stdev", "Schizophrenia mean", "Schizophrenia stdev", "t","p"])
    for pos in all_pos:
        print("POS: %s" % pos)
        users = {"control":[],"schizophrenia":[]}
        for usr, datum in parts_of_speech.items():
            freq = 0
            if pos in datum[2]:
                freq = datum[2][pos] / datum[1]
            users[datum[0]["condition"]].append(freq)
        t, p = ttest_ind(users["control"], users["schizophrenia"], equal_var=False)
        p = p * len(all_pos)
        print("T: %f, P: %f" % (t,p))
        w.writerow([pos, np.average(users["control"]), np.std(users["control"]), np.average(users["schizophrenia"]), np.std(users["schizophrenia"]), t, p])