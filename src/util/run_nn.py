from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
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
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer, SigmoidLayer, TanhLayer, ReluLayer
from pybrain.structure import FullConnection
class RunNn(object):
    FOLDS=10
    def __init__(self):
        self.momentum=0.1
        self.weightdecay=0.00002
        self.hiddensize=50
        self.learningrate=0.01
        self.lrdecay=0.999998
        self.maxEpochs=400
        self.peek=False
    def run(self, feats, file, csvpath=public_variables.CSV_PATH):
        print("Momentum: %f, Decay: %f, Hidden: %i" % (self.momentum, self.weightdecay, self.hiddensize))
        print("LR: %f, lrdecay: %f, maxEpochs: %f" % (self.learningrate, self.lrdecay, self.maxEpochs))
        self.feats=feats
        with open(file,'wb') as fout: #python2
#        with open(file,'w',encoding='utf8',newline="") as fout: #python3
            self.w=csv.writer(fout)
            self.w.writerow(["Fold","Train precision","Train recall","Train F1","Test precision","Test recall", "Test f1"])
            folds = list(range(RunNn.FOLDS))
            self.rdr = util.reader.Reader(csvpath)
            res=[]
            for fold in folds:
                res.append(self.train_test_fold(fold, feats))

            ap = np.average([d["TestPrecision"] for d in res])
            ar = np.average([d["TestRecall"] for d in res])
            af = np.average([d["TestF1"] for d in res])

            print("Average Test Precision: %f, Recall: %f, F1: %f" % (ap, ar, af))

    def build_net(self, alldata):
        net = FeedForwardNetwork()
        inLayer = LinearLayer(alldata.indim)
        hiddenLayer1 = ReluLayer(self.hiddensize)
        hiddenLayer2 = ReluLayer(self.hiddensize)
        outLayer = SoftmaxLayer(alldata.outdim)
        net.addInputModule(inLayer)
        net.addModule(hiddenLayer1)
        net.addModule(hiddenLayer2)
        net.addOutputModule(outLayer)
        in_to_hidden = FullConnection(inLayer, hiddenLayer1)
        hcon = FullConnection(hiddenLayer1, hiddenLayer2)
        hidden_to_out = FullConnection(hiddenLayer2, outLayer)
        if self.peek:
            peek1 = FullConnection(inLayer, outLayer)
            peek2 = FullConnection(hiddenLayer1, outLayer)
            #peek3 = FullConnection(hiddenLayer2, outLayer)
            net.addConnection(peek1)
            net.addConnection(peek2)
            #net.addConnection(peek3)
        net.addConnection(in_to_hidden)
        net.addConnection(hcon)
        net.addConnection(hidden_to_out)
        net.sortModules()
        return net


    def train_test_fold(self, fold, feats):
        print("Fold %i" % fold)
        folds = list(range(RunNn.FOLDS))
        folds.remove(fold)
        features = list(self.get_features(folds, feats))
        X = np.vstack([f["X"] for f in features])
        Y = np.vstack([f["Y"] for f in features]).ravel()
        alldata = ClassificationDataSet(X.shape[1], 1, nb_classes=2)
        for i in range(X.shape[0]):
            alldata.addSample(X[i,:],Y[i])
        alldata._convertToOneOfMany()
        fnn = self.build_net(alldata)
#        fnn = buildNetwork( alldata.indim, self.hiddensize, alldata.outdim, outclass=SoftmaxLayer, hiddenclass=ReluLayer)
        trainer = BackpropTrainer( fnn, dataset=alldata, momentum=self.momentum, verbose=False, weightdecay=self.weightdecay, batchlearning=False, learningrate=self.learningrate, lrdecay=self.lrdecay)
        trainer.trainUntilConvergence(maxEpochs=self.maxEpochs,validationProportion=0.10)
        precision1, recall1, f11, support=self.show_errors(fnn, X, Y,"Training ")

        testfeatures = list(self.get_features([fold], feats))
        testX = np.vstack([f["X"] for f in testfeatures])
        testY = np.vstack([f["Y"] for f in testfeatures]).ravel()
        precision2, recall2, f12, support=self.show_errors(fnn,testX,testY,"Testing ")
        self.w.writerow([fold, precision1, recall1, f11, precision2, recall2, f12])
        return {"X":testX,"Y":testY,"fold":fold,"TestPrecision":precision2,"TestRecall":recall2,"TestF1":f12}

    def get_features(self, folds, feats):
        for index, usr in enumerate(self.rdr.read_csv()):
            if int(usr["fold"]) in folds:
                if usr["anonymized_name"] not in public_variables.SKIP_USERS:
                    yield feats[usr["anonymized_name"]]

    def error(self, fnn, X, Y):
        Yp = fnn.predict(X)
        for i in range(Yp.shape[0]):
            yp=Yp[i]
            y=Y[i]

    def show_errors(self, fnn, X, Y, prefix=""):
        Yp = []
        for i in range(X.shape[0]):
            yp=fnn.activate(X[i,:])
            Yp.append(1 if yp[1] >= yp[0] else 0)
        Yp = np.hstack(Yp).ravel()
        print("Show errors: X=%s, Y=%s, Yp=%s" % (str(X.shape), str(Y.shape), str(Yp.shape)))
        precision1, recall1, f11, support=sklearn.metrics.precision_recall_fscore_support(Y, Yp, pos_label=1, average='binary')
        print(prefix+"Precision: %f, Recall: %f, F1: %f" % (precision1, recall1, f11))
        return (precision1, recall1, f11, support)