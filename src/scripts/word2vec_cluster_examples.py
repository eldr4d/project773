"""
Write example clusters out.
"""

from sklearn import cross_validation
from sklearn import svm
from scipy.stats import ttest_ind
import csv
import public_variables
import util.tweebo_reader
import gensim.models
import numpy as np
import pickle
import re
import itertools

model = gensim.models.Word2Vec.load(public_variables.WORD2VEC)
with open(public_variables.WORD2VEC_KMEANS, "rb") as fin:
    km=pickle.load(fin)

with open(public_variables.WORD2VEC_FEATURES,"rb") as fin:
    feats = pickle.load(fin)
import re
with open("../../../output/w2v_cluster_examples.txt","w", encoding="utf8") as fout:
    for cat in range(public_variables.WORD2VEC_K):
        data = [[],[]]
        for usr, feat in feats.items():
            if usr not in public_variables.SKIP_USERS:
                data[feat["Y"]].append(feat["X"][cat])
        t, p = ttest_ind(data[0], data[1], equal_var=False)
        p = min([p * public_variables.WORD2VEC_K,1])
        if p < 0.05:
            fout.write("\subsubsection{Category %i} \\noindent\n" % cat)
            fout.write("$f_c=%f$, $f_s=%f$\\\\\n$t=%f$, $p=%f$\n" % (np.average(data[0]), np.average(data[1]), t, p))
            fout.write("\\begin{itemize}\n")
            v=km.cluster_centers_[cat,:]
            print("Searching category %i" % cat)
            top = itertools.islice(sorted(([label, np.sum(np.square(model[label]-v))] for label in model.vocab), key=lambda x:x[1]),0,20)
            print("Sorted")
            for label in top:
                fout.write("\item "+re.sub("[\\$_]","", label[0])+"\n")
                print(label[0])
            fout.write("\\end{itemize}\n")

"""
\subsection{Category 70} \noindent
Control: 0.000298, Schizophrenia: 0.001274\\
t: -1.024252, p:1.000000\\
\begin{itemize}
\item bundesregierung\\
\item wulff\\
\item vn-recht\\
\item russland\\
\item energiesparlampen\\
\item hintergeht\\
\item brd-recht\\
\item rechtsanwalt
\end{itemize}
"""