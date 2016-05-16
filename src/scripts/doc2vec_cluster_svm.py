import public_variables
import util.run_svm
import pickle
with open(public_variables.DOC2VEC_FEATURES,"rb") as fin:
    features = pickle.load(fin)

svm = util.run_svm.RunSvm()
svm.run(features, "../../../output/doc2vec_cluster_svm.csv")