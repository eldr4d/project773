# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
from nltk.stem.porter import *

from datetime import date
from collections import Counter, defaultdict
from scipy.stats.stats import spearmanr 
from sklearn import metrics 

import re, os, time, string
import numpy as np

import public_variables as pv


'''

These features look at the Pennebaker mental health categories. It takes a user's tweet history and 
gets the proportion of words used in the user's tweets that appear in each Pennebaker category. 

It also breaks a user's tweets into documents by week and gets the proportion of Pennebaker categories for each week's tweets.
It then finds the variance in each category through the user's weeks as well as the mutual info and corr between certain topic pairs. 

'''

stop = stopwords.words('english') + ['__url__', '__um__', '__sm__','__ht__', 'da', 'would', 'that', 'rt', 'gt', 'lt', 'ht', 'via', 'amp', 'hi', 'hello', 'that', 'ive', 'dont', 'isnt', 'ill', 'hell', 'no', 'yeah', 'youve', 'teh', 'lot', 'didnt', 'dont']
stemmer = PorterStemmer()
regex = re.compile('[%s]' % re.escape(string.punctuation))

def __tokenize_and_normalize__(text):
	text = text.replace("'", "")
	text = re.sub("([^0-9A-Za-z__ \t])|(\w+:\/\/\S+)"," ",text)
	text = re.sub("[0-9]+", "", text)
	words = text.split()
	words = [stemmer.stem(word.lower()) for word in words if (word.strip().lower() not in stop) and (len(word) > 1)]
	words = [re.sub(regex, '', word) for word in words]
	return words

def __num_days__(d1, d2): 
	delta = d1 - d2
	return abs(delta.days)

def __iter_tweets_by_week__(user_tweets):
	tweets_by_week = []
	d1 = None
	instance = "" 
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
			instance += u" ".join([l.strip() for l in __tokenize_and_normalize__(tweet)])
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
			cat_2_words[id_2_cat[_id]].append(stemmer.stem(word.replace("*", "").replace("'", "").lower()))
	return cat_2_words

def __get_counts__(doc): 
	tokens = __tokenize_and_normalize__(doc)
	return Counter(tokens), len(tokens)

def __get_liwc_distr__(doc, liwc_dic):
	doc_bow, num_tokens = __get_counts__(doc)
	feats = defaultdict(lambda:0,{})
	for cat in liwc_dic:
		for w in liwc_dic[cat]: 
			if num_tokens > 0: 
				feats[cat] += np.float64(doc_bow[w])/np.float64(num_tokens)
	feats_lst = []
	for cat in feats: 
		feats_lst.append((cat, feats[cat]))
	return sorted(feats_lst, key=lambda tup: (tup[0], tup[1]))

def __get_variance__(user_feats):
	cats = ["affect", "posemo", "negemo", "cogmech", "health", "body", "social", "incl", "excl", "see", "hear", "feel", "sad", "anger", "anx"]
	var = []
	for c in cats: 
		var.append(np.var(user_feats[c])) 
	return var

def __get_mutual_info_and_correlation__(user_feats, bins=10): 
	pairs = [("social", "anx"), ("body", "health"), ("posemo", "negemo"), ("death", "posemo"), ("affect", "social"), ("see", "anx")]
	minfo = []
	for c1, c2 in pairs:
		corr = spearmanr(user_feats[c1], user_feats[c2])[0] 
		minfo.append(corr)
		plot = np.histogram2d(user_feats[c1], user_feats[c2], bins=bins)[0]
		try: 
			mi = metrics.mutual_info_score(None, None, contingency=plot)
		except ValueError: 
			mi = 0
		finally: 
			minfo.append(mi)
	return minfo

def get_features(users_tweets, liwc_dic): 
	feats = {}
	feats["control"] = {}
	feats["schizophrenia"] = {}
	for label in users_tweets.iterkeys():
		for user in users_tweets[label].iterkeys():
			feats[label][user] = {}
			all_user_tweets = ""
			for cat in liwc_dic: feats[label][user][cat] = []
			for tweet in __iter_tweets_by_week__(users_tweets[label][user]): 
				for cat, weight in __get_liwc_distr__(tweet, liwc_dic): 
					feats[label][user][cat].append(weight)
				all_user_tweets += tweet	
			feats[label][user]["liwc"] = __get_liwc_distr__(all_user_tweets, liwc_dic) 
			feats[label][user]["liwc_var"] = __get_variance__(feats[label][user])
			feats[label][user]["liwc_minfo_corr"] = __get_mutual_info_and_correlation__(feats[label][user])
	return feats
