import public_variables
import pickle
import util.run_svm

with open(public_variables.WORD2VEC_FEATURES,"rb") as fin:
    feats = pickle.load(fin)

svm = util.run_svm.RunSvm()
svm.run(feats)