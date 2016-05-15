# -*- coding: utf-8 -*-
from collections import defaultdict
import CMUTweetTagger
import numpy as np
import pickle
import os

import public_variables as pv

def get_pos_list(users_tweets):
  pos = set()
  for label in users_tweets.iterkeys():
    for user in users_tweets[label].iterkeys():
       for pos in users_tweets[label][user]["tot_pos"].iterkeys():
        pos.add(pos)
  print pos
  return list(pos)

def dd():
    return 0

def get_pos_features(users, users_tweets, folds):
  feats = {}
  if(os.path.isfile(pv.__input_path__ + 'pos_features.dic')):
    pos_feats_file = open(pv.__input_path__ + r'pos_features.dic', 'rb')
    feats = pickle.load(pos_feats_file)
    pos_feats_file.close()
  else:
    feats['control'] = {}
    feats['schizophrenia'] = {}

    tag_set = set()

    for label in users_tweets.iterkeys():
      for user in users_tweets[label].iterkeys():
        if users[user]["fold"] not in folds: continue

        feats[label][user] = {}
        feats[label][user]["tot_pos"] = defaultdict(dd,{})
        tagged_tweets = CMUTweetTagger.runtagger_parse(users_tweets[label][user]["tweets"])
        num_tokens = 0
        num_tweets = len(tagged_tweets)
        for tweet in tagged_tweets:
          for token, tag, _ in tweet:
            num_tokens += 1
            feats[label][user]["tot_pos"][tag] += 1
            tag_set.add(tag)
    tag_set = list(tag_set)

    for label in users_tweets.iterkeys():
      for user in users_tweets[label].iterkeys():
        if users[user]["fold"] not in folds: continue

        feats[label][user]["avg_pos"] = defaultdict(dd,{})
        feats[label][user]["avg_pos_per_tweet"] = defaultdict(dd,{})
        for tag in tag_set:
          if num_tokens == 0:
            feats[label][user]["avg_pos"][tag] = 0
            feats[label][user]["avg_pos_per_tweet"][tag] = 0
            continue

          feats[label][user]["avg_pos"][tag] = np.float(feats[label][user]["tot_pos"][tag])/np.float(num_tokens)
          feats[label][user]["avg_pos_per_tweet"][tag] = np.float(feats[label][user]["tot_pos"][tag])/np.float(num_tweets)
          if np.isnan(feats[label][user]["avg_pos"][tag]):
            feats[label][user]["avg_pos"][tag] = 0

          if np.isnan(feats[label][user]["avg_pos_per_tweet"][tag]):
            feats[label][user]["avg_pos_per_tweet"][tag] = 0
    pos_feats_file = open(pv.__input_path__ + r'pos_features.dic', 'wb')
    pickle.dump(feats, pos_feats_file)
    pos_feats_file.close()

  return feats