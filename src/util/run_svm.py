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
import csv

class RunSvm(object):

    FOLDS=10

    def get_features(self, folds, feats):
        for index, usr in enumerate(self.rdr.read_csv()):
            if int(usr["fold"]) in folds:
                if usr["anonymized_name"] not in public_variables.SKIP_USERS:
                    yield feats[usr["anonymized_name"]]

    def error(self, clf, X, Y):
        Yp = clf.predict(X)
        for i in range(Yp.shape[0]):
            yp=Yp[i]
            y=Y[i]

    def show_errors(self, clf, X, Y, prefix=""):
        Yp=clf.predict(X)
        precision1, recall1, f11, support=sklearn.metrics.precision_recall_fscore_support(Y, Yp, pos_label=1, average='binary')
        print(prefix+"Precision: %f, Recall: %f, F1: %f" % (precision1, recall1, f11))
        return (precision1, recall1, f11, support)

    def train_test_fold(self, fold, feats):
        print("Fold %i" % fold)
        folds = list(range(RunSvm.FOLDS))
        folds.remove(fold)
        features = list(self.get_features(folds, feats))
        X = np.vstack([f["X"] for f in features])
        Y = np.vstack([f["Y"] for f in features]).ravel()
        clf = svm.SVC(kernel='linear', C=5)
        clf.fit(X,Y)
        Yp=clf.predict(X)
        precision1, recall1, f11, support=self.show_errors(clf, X, Y,"Training ")

        testfeatures = list(self.get_features([fold], feats))
        testX = np.vstack([f["X"] for f in testfeatures])
        testY = np.vstack([f["Y"] for f in testfeatures]).ravel()
        #print("Shapes: %s, %s" % (str(testX.shape), str(testY.shape)))
        #print("Distribution: %i, %i" % (np.count_nonzero(testY==0), np.count_nonzero(testY==1)))
        testyp=clf.predict(testX)
        #print("Predicted Distribution: %i, %i" % (np.count_nonzero(testyp==0), np.count_nonzero(testyp==1)))
        precision2, recall2, f12, support=self.show_errors(clf,testX,testY,"Testing ")
        #print("Test Precision: %f, Recall: %f, F1: %f" % (precision2, recall2, f12))
        self.w.writerow([fold, precision1, recall1, f11, precision2, recall2, f12])
        return {"X":testX,"Y":testY,"fold":fold,"Yp":testyp,"TestPrecision":precision2,"TestRecall":recall2,"TestF1":f12}

    def run(self, feats, file):
        self.feats=feats
        with open(file,'w',encoding='utf8',newline='') as fout:
            self.w=csv.writer(fout)
            self.w.writerow(["Fold","Train precision","Train recall","Train F1","Test precision","Test recall", "Test f1"])
            folds = list(range(RunSvm.FOLDS))
            self.rdr = util.reader.Reader(public_variables.CSV_PATH)
            res=[]
            for fold in folds:
                res.append(self.train_test_fold(fold, feats))

            ap = np.average([d["TestPrecision"] for d in res])
            ar = np.average([d["TestRecall"] for d in res])
            af = np.average([d["TestF1"] for d in res])

            print("Average Test Precision: %f, Recall: %f, F1: %f" % (ap, ar, af))









