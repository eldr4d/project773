import public_variables
import os
import pickle
import csv

with open("../"+public_variables.__input_path__ + r'liwc_features.dic', 'rb') as liwc_feats_file:
  feats = pickle.load(liwc_feats_file)

print("A")
print(feats.keys())
print("b")
print(feats["control"].items()[0])
print("c")
print(feats["control"].items()[0][0])
print("d")
print(feats["control"].items()[0][1])

#cats = list(x[1] for x in feats["control"].items()[0][1]["liwc_by_week"].keys())
cats=feats["control"].items()[0][1]["liwc_by_week"].keys()
#cats=feats["control"].items()[0][1]["liwc_by_week"].keys()
print(cats)

cats=[x[0] for x in feats["control"].items()[0][1]["liwc_var"]]
print(cats)

with open("../../../output/liwc_stats.csv","wb") as fout:
    w=csv.writer(fout)
    w.writerow(["user", "condition"]+[x[0] for x in feats["control"].items()[0][1]["liwc_var"]])
    for cond, usrs in feats.items():
        for usr, f in usrs.items():
            w.writerow([usr, cond]+[x[1] for x in f["liwc_var"]])