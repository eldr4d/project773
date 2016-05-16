# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
from nltk.stem.porter import *

from datetime import date
from collections import Counter, defaultdict


import pickle

import scipy as scipy
from scipy.stats import wilcoxon
from scipy.stats.stats import spearmanr
from sklearn import metrics, feature_selection

import re, os, time, string
import numpy as np
import pickle as pickle

import public_variables as pv
import topic_features as tp


'''

These features look at the Pennebaker mental health categories. It takes a user's tweet history and
gets the proportion of words used in the user's tweets that appear in each Pennebaker category.

It also breaks a user's tweets into documents by week and gets the proportion of Pennebaker categories for each week's tweets.
It then finds the variance in each category through the user's weeks (probably want something different than variance though)
as well as the mutual info btwn certain category pairs for the tweeter.

'''

stemmer = PorterStemmer()

def __preprocess__(text):
  text = text.replace("'", "")
  text = text.replace("-", "")
  words = text.split()
  words = [stemmer.stem(word.lower()) for word in words]
  return words

def __num_days__(d1, d2):
  delta = d1 - d2
  return abs(delta.days)

def __iter_tweets_by_week__(user_tweets):
  tweets_by_week = []
  d1 = None
  instance = ""
  try:
    timestamps = user_tweets['timestamps']
    for tweet in user_tweets["tweets"]:
      tmp = time.strftime('%m/%d/%Y', time.strptime(timestamps.pop(),'%a %b %d %H:%M:%S +0000 %Y'))
      tmp = tmp.split('/')
      d2 = date(int(tmp[2]), int(tmp[0]), int(tmp[1]))
      if d1 is None:
        d1 = d2
        instance = " "
      elif __num_days__(d1, d2) > 7:
        d1 = d2
        if (len(instance.split()) > 99):
          instance += u"\n"
          yield instance
        instance = " "
      else:
        instance += u" ".join([l.strip() for l in __preprocess__(tweet)])
  except Exception:
    print "Timestamp Error: Discarding Users Tweets"
  finally:
    yield instance

def read_liwc(path):
  txt = open(path, "r+").read()
  _, categories, words = txt.split("%")
  categories = filter(str.strip, categories.splitlines())
  words = filter(str.strip, words.splitlines())
  id_2_cat =  {}
  cat_2_words = {}
  for line in categories:
    _id, cat = line.split()
    id_2_cat[_id] = cat
    cat_2_words[cat] = []
  for line in words:
    if "<of>" in line or "(02 134)" in line: continue
    tmp = line.split()
    word = tmp[0]
    ids = tmp[1:]
    for _id in ids:
      cat_2_words[id_2_cat[_id]].append(stemmer.stem(word.replace("*", "").replace("'", "").replace("-", "").lower()))
  return cat_2_words

def __get_counts__(doc):
  tokens = __preprocess__(doc)
  return Counter(tokens), len(tokens)

def __get_liwc_distr__(doc, liwc_dic):
  doc_bow, num_tokens = __get_counts__(doc)
  feats = defaultdict(lambda:0,{})

  for cat, words in liwc_dic.items():
    for w in words:
      if not feats[cat]: feats[cat] = 0
      if doc_bow[w] == 0: continue
      feats[cat] += np.float64(doc_bow[w])/np.float64(num_tokens)

  feats_lst = []
  for cat in feats.keys():
    if np.isnan(feats[cat]): feats[cat] = 0
    feats_lst.append((cat, feats[cat]))

  return sorted(feats_lst, key=lambda tup: (tup[0], tup[1]))

def __get_var__(vec):
  var = np.var(vec)
  if np.isnan(var): var = 0
  return var

# check which categories...
def __get_variance__(user_feats, liwc_dic):
  sig = ["we", "bio", "achieve", "family", "home", "past", "sexual", "space", "negemo", "incl", "affect", "social", "sad", "posemo", "humans", "number", "assent", 
  "ingest", "work", "time", "present", "tentat", "incl", "motion", "shehe", "social", "friend", "body", "percept", "cause", "certain", "verb", "adv", "affect", 
  "pronoun", "funct", "preps", "auxverb"]
  return [(c, __get_var__(user_feats[c])) for c in sorted(liwc_dic.keys()) ]

# mutual info/similarity of certain cat's time progressions....
def __get_mutual_info__(user_feats, bins=3):
  pairs = [("social", "anx"), ("body", "health"), ("posemo", "negemo"), ("death", "posemo"), ("affect", "social"), ("see", "anx"), ("body", "anger"), ("body", "death"), ("body", "negemo"), ("body", "posemo"), ("health", "negemo"), ("health", "posemo"), ("social", "excl"), ("social", "inhib"), ("social", "humans"), ("swear", "ingest"), ("swear", "anger"), ("ingest", "sad"), ("insight", "percept"), ("insight", "inhib"), ("insight", "incl"), ("body", "anger"), ("body", "death"), ("money", "bio"), ("work", "humans"), ("work", "i"), ("work", "insight"), ("tentat", "sad"), ("anx", "health"), ("anx", "insight"), ("anx", "you"), ("affect", "bio"), ("affect", "body"), ("affect", "sad")]
  m_info = []

  for c1, c2 in pairs:
    plot = np.histogram2d(user_feats[c1], user_feats[c2], bins=bins)[0]
    try:
      mi = metrics.mutual_info_score(None, None, contingency=plot)
    except ValueError:
      mi = 0
    finally:
      if np.isnan(mi):
        mi= 0
      m_info.append((c1, c2, mi))
  return m_info

def __get_correlation__(user_feats):
  pairs = [("social", "anx"), ("body", "health"), ("posemo", "negemo"), ("death", "posemo"), ("affect", "social"), ("see", "anx"), ("anger", "affect"), 
  ("anger", "family"), ("anger", "posemo"), ("negemo", "affect"), ('negemo', "cogmech"), ("body", "cause"), ("body", "posemo"), ("body", "family"), 
  ("work", "time"), ("time", "you"), ("social", "you"), ("social", "posemo"), ("social", "excl"), ("leisure", "affect"), ("anx", "death"), ("anx", "discrep"), 
  ("anx", "sad"), ("anx", "time"), ("anx", "health"), ("anx", "percept"), ("anx", "work"), ("see", "assent"), ("sexual", "sad"), ("sexual", "inhib")]
  s_corr = []
  s_pval = []


  for c1, c2 in pairs:
    s = spearmanr(user_feats[c1], user_feats[c2])
    corr = s[0]
    pval = s[1]
    if np.isnan(corr):
      corr = 0
      pval = 0
    if np.isnan(pval):
      pval = 0
    s_corr.append((c1, c2, corr))
    s_pval.append((c1, c2, pval))

  return s_corr, s_pval

# returns max and avg. cosine distance btween weeks liwc distributions.
def __get_cosine_distance__(weeks):
  avg_cd = 0
  max_cd = 0
  num_wks = len(weeks)

  prev = weeks.pop()
  for curr in weeks:
    cd = scipy.spatial.distance.cosine(prev, curr)
    avg_cd +=  cd
    if cd > max_cd: max_cd = cd
    prev = curr

  avg_cd = np.float(avg_cd)/np.float(num_wks)
  if np.isnan(avg_cd):
    avg_cd = 0
  return (avg_cd, max_cd)


import csv
def get_features(users, users_tweets, liwcdic_file, folds):
  feats = {}
  if(os.path.isfile(pv.__input_path__ + 'liwc_features.dic')):
    print("Loading LIWC from cache")
    with open(pv.__input_path__ + r'liwc_features.dic', 'rb') as liwc_feats_file:
      feats = pickle.load(liwc_feats_file)
  else:
    print("Calculating LIWC")
    feats["control"] = {}
    feats["schizophrenia"] = {}
    liwc_dic = read_liwc(liwcdic_file)
    for label, usertweets in users_tweets.items():
      for user, tweets in usertweets.items():
        if users[user]["fold"] not in folds: continue

        feats[label][user] = {}

        print("Calculating liwc_by_week")
        feats[label][user]["liwc_by_week"] = {}
        for cat in liwc_dic: feats[label][user]["liwc_by_week"][cat] = []

        by_week = []
        all_user_tweets = ""
        for week_of_tweets in __iter_tweets_by_week__(users_tweets[label][user]):
          week = []

          for cat, weight in __get_liwc_distr__(week_of_tweets, liwc_dic):
            week.append(weight)
            feats[label][user]["liwc_by_week"][cat].append(weight)

          by_week.append(week)
          all_user_tweets += week_of_tweets

        print("Calculating liwc")
        feats[label][user]["liwc"] = __get_liwc_distr__(all_user_tweets, liwc_dic)

        # print("Calculating liwc_var")
        # feats[label][user]["liwc_var"] = __get_variance__(feats[label][user]["liwc_by_week"], liwc_dic)

        print("Calculating liwc_minfo")
        feats[label][user]["liwc_minfo"] = __get_mutual_info__(feats[label][user]["liwc_by_week"])

        # print("Calculating liwc distance")
        # avg_cd, max_cd = __get_cosine_distance__(by_week)
        # feats[label][user]["liwc_avg_cos_dis"] = avg_cd
        # feats[label][user]["liwc_max_cos_dis"] = max_cd

        # print("Calculating liwc correlation")
        # s_corr, s_pval = __get_correlation__(feats[label][user]["liwc_by_week"])
        # feats[label][user]["liwc_spearman_corr"] = s_corr
        # feats[label][user]["liwc_spearman_pval"] = s_pval

    with open(pv.__input_path__ + r'liwc_features.dic', 'wb') as liwc_feats_file:
      pickle.dump(feats, liwc_feats_file)

  return feats
