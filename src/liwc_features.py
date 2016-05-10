# -*- coding: utf-8 -*-
from nltk.corpus import TwitterCorpusReader
from nltk.corpus import stopwords
from nltk.stem.porter import *

from datetime import date
import numpy as np

import re, os, time, string

import public_variables as pv

from collections import Counter, defaultdict


'''

These features look at the Pennebaker mental health categories. It takes a user's tweet history and 
gets the proportion of words used in the user's tweets that appear in each Pennebaker category. 

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
	for tweet in user_tweets:
		tmp = time.strftime('%m/%d/%Y', time.strptime(tweet['joined'],'%a %b %d %H:%M:%S +0000 %Y'))
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
	feats = []
	for cat in liwc_dic:
		for w in liwc_dic[cat]: 
			feats.append((cat, np.float64(doc_bow[w])/np.float64(num_tokens)))
	return sorted(feats, key=lambda tup: (tup[0], tup[1]))
	
def get_features(users_tweets, liwc_dic): 
	feats = defaultdict(lambda:0,{})
	feats["control"] = {}
	feats["schizophrenia"] = {}
	for label in users_tweets.iterkeys():
		for user in users_tweets[label].iterkeys():
			feats[label][user] = {}
			all_user_tweets = ""
			for tweet in users_tweets[label][user]["tweets"]: 
				all_user_tweets += tweet	
			feats[label][user]["liwc"] = __get_liwc_distr__(all_user_tweets, liwc_dic) 
	return feats
