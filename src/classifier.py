import csv
import sys

from sklearn import svm

import matplotlib.pyplot as plt
import sklearn.metrics as metrics

import perplexity_model as perp
import read_dataset as rd

def feature_selection(inputs_and_labels): 
  # Feel free to choose different test. I just went with chi2:  http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection
  for i in range(10): 
    feature_selection.chi2(inputs_and_labels[i]["features"], inputs_and_labels[i]["labels"])
  # TODO:  plot pvalues for each feature tested (pyplot imported as plt above)... 
  raise NotImplementedError

def evaluate_results(inputs_and_labels, predictions):
  for i in range(10): 
    # TODO:  choose metric ... http://scikit-learn.org/stable/modules/classes.html#classification-metrics
     print metrics.roc_auc_score(inputs_and_labels[i]["labels"], predictions[i])
     print metrics.classification_report(inputs_and_labels[i]["labels"], predictions[i])
     # TODO:  plot scores ... example: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#example-model-selection-plot-roc-crossval-py
  return 1.0

def train_classifier(inputs_and_labels, kernel='linear'):
  svms = {}
  for i in range(10):
    svms[i] = svm.SVC(kernel=kernel, verbose=True)

    inputs = []
    labels = []
    for j in range(10):
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
  for i in range(10):
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
  perplexity = perp.get_perplexity_of_users(users_tweets)

  features_and_labels = {}

  for i in range(10):
    features_and_labels[i] = {}
    features_and_labels[i]["features"] = []
    features_and_labels[i]["labels"] = []

  for user, dic in users.iteritems():
    user_features = []

    # Add perplexity as feature
    user_features.append(perplexity[dic["group"]][user]["unigrams"])
    user_features.append(perplexity[dic["group"]][user]["bigrams"])
    user_features.append(perplexity[dic["group"]][user]["trigrams"])

    # Add total Number of tweets as feature
    user_features.append(len(users_tweets[dic["group"]][user]["tweets"]))

    features_and_labels[dic["fold"]]["features"].append(user_features)
    features_and_labels[dic["fold"]]["labels"].append(dic["group"])

  return features_and_labels

def main(argv):
  users_tweets = rd.get_tweets()

  users = load_manifest()

  features_and_labels = create_features(users, users_tweets)
  print features_and_labels.keys()
  print ""
  print features_and_labels[2].keys()
  print ""
  print str(len(features_and_labels[2]["features"])) + " " + str(len(features_and_labels[2]["labels"]))
  print ""
  print features_and_labels[2]["labels"]
  svms = train_classifier(features_and_labels, kernel='rbf')
  results = do_prediction(features_and_labels, svms)
  print "\n"
  print "\n"
  print results

if __name__ == "__main__":
  main(sys.argv[1:])