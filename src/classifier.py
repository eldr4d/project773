import csv
import sys
import time

from sklearn import svm

from gensim import corpora, models

import numpy as np
import sklearn.metrics as metrics
import util.run_svm
import topic_features as t
import liwc_features as lf
import pos_features as ps
import public_variables as pv
import perplexity_model as perp
import read_dataset as rd
import doc2vec_features
import word2vec_features
import post_time as pt
import coherence_features as cf

##########################################
#Features to enable
##########################################
ENABLE_LIWC=True
ENABLE_PERPLEXITY=True
ENABLE_TEMPORAL=True
ENABLE_TOPICS=True
ENABLE_POS=True
ENABLE_COHERENCE=True
ENABLE_DOC2VEC=True
ENABLE_WORD2VEC=True

# For full cross validation
Folds_to_use = range(10)
Folds_to_predict = range(10)

# For Testing
# Folds_to_use = [1]
# Folds_to_predict = [0]

def __get_id__(label):
  return 1 if label.startswith('s') else 0

def __labels_2_binary__(vector):
  return [__get_id__(label) for label in vector]

def evaluate_results(inputs_and_labels, predictions):
  print("*******************************************")
  print("")

  avg_precision = 0
  avg_recall = 0
  avg_roc_auc = 0
  avg_f1 = 0

  for fold in Folds_to_predict:
     # print "True: "  + " (len: " + str(len(inputs_and_labels[fold]["labels"])) + ")"
     # print str(inputs_and_labels[fold]["labels"])
     y_true = __labels_2_binary__(inputs_and_labels[fold]["labels"])

     # print ""
     # print "Predicted: " + " (len: " + str(len(predictions[fold])) + ")"
     # print str(predictions[fold])
     y_pred = predictions[fold]

     roc_auc = metrics.roc_auc_score(y_true, y_pred)

     print("")
     print("Fold = " + str(fold))
     print("Scores: ")
     print("ROC AUC: " + str(roc_auc))
     print(metrics.classification_report(y_true, y_pred))
     print("")

     avg_roc_auc += np.float(roc_auc)/np.float(10)
     avg_precision += np.float(metrics.precision_score(y_true, y_pred))/np.float(10)
     avg_recall += np.float(metrics.recall_score(y_true, y_pred))/np.float(10)
     avg_f1 += np.float(metrics.f1_score(y_true, y_pred))/np.float(10)

  print("Overall Scores: ")
  print("ROC AUC: " + str(avg_roc_auc))
  print("Prec: " + str(avg_precision))
  print("Recall: " + str(avg_recall))
  print("F1: " + str(avg_f1))

def train_classifier(inputs_and_labels, kernel='linear'):
  svms = {}
  for i in Folds_to_predict:
    svms[i] = svm.SVC(kernel=kernel, verbose=False)

    inputs = []
    labels = []
    for j in Folds_to_use:
      if(i==j):
        continue
      inputs = inputs + inputs_and_labels[j]["features"]
      labels = labels + inputs_and_labels[j]["labels"]
      # print "I = " + str(i)
      # print inputs
      # print labels
      # print "\n"
    # print str(len(inputs)) + " " + str(len(labels))
    inputs=np.vstack(inputs)
    labels = np.array(__labels_2_binary__(labels)).ravel()
    #print("Svm shapes: %s, %s" % (str(inputs.shape), str(labels.shape)))
    svms[i].fit(inputs, labels)
    #print("C0: %i, C1: %i" % (len([k for k in __labels_2_binary__(labels) if k ==0]),len([k for k in __labels_2_binary__(labels) if k ==1])))
    #print(np.std(inputs,axis=0))
    #print(np.average(inputs,axis=0))
    util.run_svm.RunSvm().show_errors(svms[i],inputs,labels)


  return svms

def do_prediction(inputs_and_labels, svms):
  results = {}
  for i in Folds_to_predict:
    results[i] = svms[i].predict(inputs_and_labels[i]["features"])
  return results

def load_manifest():
  non_english = ['ioY8SXeZ4O', 'kABBqs5cM25', 'fE28aNayZ3KZ2'] # folds: 4, 0, 9

  users = {}
  with open('../input/anonymized_user_manifest.csv', 'rb') as csvfile:
    manifest = csv.reader(csvfile, delimiter=',')

    #skip first line
    manifest.next()
    for row in manifest:
      # if row[0] in non_english: continue
      users[row[0]] = {}
      users[row[0]]["group"] = row[1]
      users[row[0]]["fold"] = int(row[5])
  return users

def create_features(users, users_tweets):

  folds_for_features = Folds_to_use + Folds_to_predict

  if ENABLE_PERPLEXITY:
    print("get_perplexity_of_users")
    perplexity = perp.get_perplexity_of_users(users_tweets)

  if ENABLE_TEMPORAL:
    print("tweet_time")
    temporal = pt.tweet_time(users_tweets)

  if ENABLE_COHERENCE:
    print("get_coherence")
    coherence = cf.get_coherence(users_tweets)

  if ENABLE_TOPICS:
    print("get_features")
    topics = t.get_features(users, users_tweets, pv.__lda_model__, pv.__lda_dict__, folds_for_features)

  if ENABLE_LIWC:
    print("Getting liwc features")
    liwc = lf.get_features(users, users_tweets, pv.__liwc__, folds_for_features)
    print("Got liwc features")
    liwc_keys=list()

  if ENABLE_POS:
    print("get_pos_features")
    pos_feats = ps.get_pos_features(users, users_tweets, folds_for_features)

  print("Collecting features")
  features_and_labels = {}

  for i in Folds_to_use:
    features_and_labels[i] = {}
    features_and_labels[i]["features"] = []
    features_and_labels[i]["labels"] = []

  for i in Folds_to_predict:
    features_and_labels[i] = {}
    features_and_labels[i]["features"] = []
    features_and_labels[i]["labels"] = []

  for user, dic in users.iteritems():
    if users[user]["fold"] not in folds_for_features: continue   # for debugging: ignores users not in either fold...

    user_features = []

    ######### Add topic distribution as features #########

    if ENABLE_TOPICS:
      user_features.append(topics[dic["group"]][user]["num_sig_topics"])
      for topic_id in topics[dic["group"]][user]["topics"].iterkeys():
        user_features.append(topics[dic["group"]][user]["topics"][topic_id])

    ######### Add LIWC category distribution as feature #########
    if ENABLE_LIWC:
      for i in range(63):
       user_features.append(liwc[dic["group"]][user]["liwc"][i][1])
      for c1, c2, minfo in liwc[dic["group"]][user]["liwc_minfo"]:
        user_features.append(minfo)

    ######### Add parts-of-speech as feature #########
    if ENABLE_POS:
      for tag in pos_feats[dic["group"]][user]["avg_pos"].keys():
        #### user_features.append(pos_feats[dic["group"]][user]["avg_pos_per_tweet"][tag])
        user_features.append(pos_feats[dic["group"]][user]["avg_pos"][tag])
        #### user_features.append(pos_feats[dic["group"]][user]["tot_pos"][tag])

    if ENABLE_DOC2VEC:
      doc2vec_features.add_features(user, user_features)
      doc2vec_features.add_shift(user, user_features)
    if ENABLE_WORD2VEC:
      word2vec_features.add_features(user, user_features)    ######### Add perplexity as feature #########
    if ENABLE_PERPLEXITY:
      user_features.append(perplexity[dic["group"]][user]["unigrams"])
      user_features.append(perplexity[dic["group"]][user]["bigrams"])
      user_features.append(perplexity[dic["group"]][user]["trigrams"])

    ######### Add time features #########
    if ENABLE_TEMPORAL:
      user_features.append(temporal[dic["group"]][user]["avg_posting_time"])
      user_features.append(temporal[dic["group"]][user]["frac_AM_posts"])
      user_features.append(temporal[dic["group"]][user]["frac_winter_posts"])
      user_features.append(temporal[dic["group"]][user]["frac_summer_posts"])
      user_features.append(temporal[dic["group"]][user]["daily_tweeting_rate"])
      user_features.append(temporal[dic["group"]][user]["weekly_tweeting_rate"])
      user_features.append(temporal[dic["group"]][user]["monthly_tweeting_rate"])
      user_features.append(temporal[dic["group"]][user]["10_min_span_tweets"])
      user_features.append(temporal[dic["group"]][user]["30_min_span_tweets"])
      user_features.append(temporal[dic["group"]][user]["60_min_span_tweets"])
      user_features.append(temporal[dic["group"]][user]["10_min_span_time"])
      user_features.append(temporal[dic["group"]][user]["30_min_span_time"])
      user_features.append(temporal[dic["group"]][user]["60_min_span_time"])
      user_features.append(temporal[dic["group"]][user]["comp_120ch_twt_60sit"])
    # Add coherence features
    if ENABLE_COHERENCE:
      user_features.append(coherence[dic["group"]][user]["avg_FOC"])
      user_features.append(coherence[dic["group"]][user]["median_FOC"])
      user_features.append(coherence[dic["group"]][user]["std_FOC"])
      user_features.append(coherence[dic["group"]][user]["avg_SOC"])
      user_features.append(coherence[dic["group"]][user]["median_SOC"])
      user_features.append(coherence[dic["group"]][user]["std_SOC"])

    ######### Add Twitter metadata features #########
    #user_features.append(len(users_tweets[dic["group"]][user]["tweets"])) # total number of tweets as feature
    #user_features.append(users_tweets[dic["group"]][user]["friends_count"])
    #user_features.append(users_tweets[dic["group"]][user]["followers_count"])

    features_and_labels[dic["fold"]]["features"].append(user_features)
    features_and_labels[dic["fold"]]["labels"].append(dic["group"])

  return features_and_labels

def main(argv):
  users_tweets = rd.get_tweets()

  users = load_manifest()

  start = time.time()
  print("Using folds: " + str(Folds_to_use))
  print("Predicting folds: " + str(Folds_to_predict))
  print("Creating features")
  features_and_labels = create_features(users, users_tweets)
  print("Features created")
  # print features_and_labels.keys()
  # print ""
  # print features_and_labels[fold].keys()
  # print ""
  # print str(len(features_and_labels[fold]["features"])) + " " + str(len(features_and_labels[fold]["labels"]))
  # print ""
  # print features_and_labels[fold]["labels"]
  svms = train_classifier(features_and_labels, kernel='rbf')
  results = do_prediction(features_and_labels, svms)
  # print "\n"
  # print "\n"
  # print results
  evaluate_results(features_and_labels, results)
  end = time.time()
  print("")
  print("Time: " + str(end - start))

if __name__ == "__main__":
  main(sys.argv[1:])
