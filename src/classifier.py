import csv
import sys

from sklearn import svm

import perplexity_model as perp
import read_dataset as rd

def evaluate_results(inputs_and_labels, predictions):
  #TODO someone should do that

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