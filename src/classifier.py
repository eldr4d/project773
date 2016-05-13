import csv
import sys
import time

from sklearn import svm

from gensim import corpora, models

import sklearn.metrics as metrics

import topic_features as t
import liwc_features as pb
import pos_features as ps
import public_variables as pv
import perplexity_model as perp
import read_dataset as rd

# For full cross validation
Folds_to_use = range(10)
Folds_to_predict = range(10)

# For Testing
# Folds_to_use = [1]
# Folds_to_predict = [0]

def feature_selection(inputs_and_labels):
  for idx, fold in enumerate(Folds_to_predict):
    y_true = __labels_2_binary__(inputs_and_labels[fold]["labels"])
    y_pred = __labels_2_binary__(predictions[fold])
    feature_selection.chi2(inputs_and_labels[i]["features"], inputs_and_labels[i]["labels"])
  raise NotImplementedError

def __get_id__(label): 
  return 1 if label.startswith('s') else 0

def __labels_2_binary__(vector): 
  return [__get_id__(label) for label in vector]

def evaluate_results(inputs_and_labels, predictions):
  print "*******************************************"
  print ""

  for fold in Folds_to_predict:
     print "True: " 
     print str(inputs_and_labels[fold]["labels"])
     y_true = __labels_2_binary__(inputs_and_labels[fold]["labels"])

     print ""
     print "Predicted: " 
     print str(predictions[fold])
     y_pred = __labels_2_binary__(predictions[fold])
     
     print ""
     print "Scores: "
     print "ROC AUC: " + str(metrics.roc_auc_score(y_true, y_pred))
     print metrics.classification_report(y_true, y_pred)
     print ""

def train_classifier(inputs_and_labels, kernel='linear'):
  svms = {}
  for i in Folds_to_predict:
    svms[i] = svm.SVC(kernel=kernel, verbose=True)

    inputs = []
    labels = []
    for j in Folds_to_use:
      if(i==j):
        continue
      inputs = inputs + inputs_and_labels[j]["features"]
      labels = labels + inputs_and_labels[j]["labels"]
      print "I = " + str(i)
      print inputs
      print labels
      print "\n"
    print str(len(inputs)) + " " + str(len(labels))
    svms[i].fit(inputs, labels)

  return svms

def do_prediction(inputs_and_labels, svms):
  results = {}
  for i in Folds_to_predict:
    results[i] = svms[i].predict(inputs_and_labels[i]["features"])
  return results

def load_manifest():
  users = {}
  with open('../input/anonymized_user_manifest.csv', 'rb') as csvfile:
    manifest = csv.reader(csvfile, delimiter=',')

    #skip first line
    manifest.next()
    for row in manifest:
      users[row[0]] = {}
      users[row[0]]["group"] = row[1]
      users[row[0]]["fold"] = int(row[5])
  return users

def create_features(users, users_tweets):
  
  folds_for_features = Folds_to_use + Folds_to_predict

  perplexity = perp.get_perplexity_of_users(users_tweets)

  lda_model = models.ldamodel.LdaModel.load(pv.__lda_model__)
  dictionary = corpora.Dictionary.load(pv.__lda_dict__)
  topics = t.get_features(users, users_tweets, lda_model, dictionary, folds_for_features)

  liwc_dic = pb.read_liwc(pv.__liwc__)
  liwc = pb.get_features(users, users_tweets, liwc_dic, folds_for_features)
  
  pos_feats = ps.get_pos_features(users, users_tweets, folds_for_features)

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

    # Add topic distribution as features
    user_features.append(topics[dic["group"]][user]["num_sig_topics"])
    for topic_id in topics[dic["group"]][user]["topics"].iterkeys(): 
      user_features.append(topics[dic["group"]][user]["topics"][topic_id])

    # Add LIWC category distribution as feature 
    for i in range(63): 
      user_features.append(liwc[dic["group"]][user]["liwc"][i][1])
    for i in range(15):
      user_features.append(liwc[dic["group"]][user]["liwc_var"][i])
    for minfo in liwc[dic["group"]][user]["liwc_minfo"]: 
      user_features.append(minfo)

    # Add parts-of-speech as feature
    for tag in pos_feats[dic["group"]][user]["avg_pos"].keys(): 
      user_features.append(pos_feats[dic["group"]][user]["avg_pos_per_tweet"][tag])
      user_features.append(pos_feats[dic["group"]][user]["avg_pos"][tag])
      user_features.append(pos_feats[dic["group"]][user]["tot_pos"][tag])

    # Add perplexity as feature
    user_features.append(perplexity[dic["group"]][user]["unigrams"])
    user_features.append(perplexity[dic["group"]][user]["bigrams"])
    user_features.append(perplexity[dic["group"]][user]["trigrams"])

    # Add Twitter metadata features
    user_features.append(len(users_tweets[dic["group"]][user]["tweets"])) # total number of tweets as feature
    #user_features.append(users_tweets[dic["group"]][user]["friends_count"])
    #user_features.append(users_tweets[dic["group"]][user]["followers_count"])

    features_and_labels[dic["fold"]]["features"].append(user_features)
    features_and_labels[dic["fold"]]["labels"].append(dic["group"])

  return features_and_labels

def main(argv):
  users_tweets = rd.get_tweets()

  users = load_manifest()

  start = time.time()
  print "Using folds: " + str(Folds_to_use)
  print "Predicting folds: " + str(Folds_to_predict)
  fold = Folds_to_use[0]
  features_and_labels = create_features(users, users_tweets)
  print features_and_labels.keys()
  print ""
  print features_and_labels[fold].keys()
  print ""
  print str(len(features_and_labels[fold]["features"])) + " " + str(len(features_and_labels[fold]["labels"]))
  print ""
  print features_and_labels[fold]["labels"]
  svms = train_classifier(features_and_labels, kernel='rbf')
  results = do_prediction(features_and_labels, svms)
  print "\n"
  print "\n"
  print results
  evaluate_results(features_and_labels, results)
  end = time.time()
  print "Time: " + str(end - start)

if __name__ == "__main__":
  main(sys.argv[1:])