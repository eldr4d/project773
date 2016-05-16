import public_variables
import pickle
import util.run_nn

with open(public_variables.WORD2VEC_FEATURES,"rb") as fin:
    feats = pickle.load(fin)

svm = util.run_nn.RunNn()
svm.run(feats,"../../../output/word2vec_nn.csv")