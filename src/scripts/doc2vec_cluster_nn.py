import public_variables
import util.run_nn
import pickle
with open(public_variables.DOC2VEC_FEATURES,"rb") as fin:
    features = pickle.load(fin)

svm = util.run_nn.RunNn()
svm.run(features,"../../../output/doc2vec_nn.csv")