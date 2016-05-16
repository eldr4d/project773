"""
Run an svm using the clusters from word2vec_cluster and word2vec_features
"""

import public_variables
import pickle
import util.reader
import util.tweebo_reader
import util.tweet_cleaner
import gensim.models
import numpy as np
from sklearn import svm
import sklearn.metrics
FOLDS=10

def get_features(folds, feats):
    for index, usr in enumerate(rdr.read_csv()):
        if int(usr["fold"]) in folds:
            yield feats[usr["anonymized_name"]]

def error(clf, X, Y):
    Yp = clf.predict(X)
    for i in range(Yp.shape[0]):
        yp=Yp[i]
        y=Y[i]

def train_test_fold(fold, feats):
    print("Fold %i" % fold)
    folds = list(range(FOLDS))
    folds.remove(fold)
    features = list(get_features(folds, feats))
    X = np.vstack([f["X"] for f in features])
    Y = np.vstack([f["Y"] for f in features]).ravel()

    clf = svm.SVC(kernel='linear')
    clf.fit(X,Y)
    Yp=clf.predict(X)
    precision, recall, f1, support=sklearn.metrics.precision_recall_fscore_support(Y, Yp, pos_label=1, average='binary')
    print("Train Precision: %f, Recall: %f, F1: %f" % (precision, recall, f1))

    testfeatures = list(get_features([fold], feats))
    testX = np.vstack([f["X"] for f in testfeatures])
    testY = np.vstack([f["Y"] for f in testfeatures]).ravel()
    testyp=clf.predict(testX)
    precision, recall, f1, support=sklearn.metrics.precision_recall_fscore_support(testY, testyp, pos_label=1, average='binary')
    print("Test Precision: %f, Recall: %f, F1: %f" % (precision, recall, f1))

    return {"X":testX,"Y":testY,"fold":fold,"Yp":testyp}


if __name__ == "__main__":
    folds = range(FOLDS)

    rdr = util.reader.Reader(public_variables.CSV_PATH)
    trdr = util.tweebo_reader.TweeboReader(public_variables.TWEEBO_OUTPUT)
    model = gensim.models.Word2Vec.load(public_variables.WORD2VEC)
    with open(public_variables.WORD2VEC_KMEANS, "rb") as fin:
        km=pickle.load(fin)
    with open(public_variables.WORD2VEC_FEATURES,"rb") as fin:
        feats = pickle.load(fin)


    for fold in folds:
        train_test_fold(fold, feats)






