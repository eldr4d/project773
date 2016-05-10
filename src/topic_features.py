# -*- coding: utf-8 -*-
from nltk.corpus import TwitterCorpusReader
from nltk.corpus import stopwords
from nltk.stem.porter import *

from gensim import corpora, models, similarities, matutils

import numpy as np

from datetime import date
import pickle

import re, string, subprocess, os, time, random

import public_variables as pv
import read_dataset as rd
import classifier as cl


'''

Gives topics for a users full tweet history based on a topic model built from qntfy data and pooled by author. 

'''

 
stop = stopwords.words('english') + ['__url__', '__um__', '__sm__','__ht__', 'da', 'would', 'that', 'rt', 'gt', 'lt', 'ht', 'via', 'amp', 'hi', 'hello', 'that', 'ive', 'dont', 'isnt', 'ill', 'hell', 'no', 'yeah', 'youve', 'teh', 'lot', 'didnt', 'dont']
stemmer = PorterStemmer()
regex = re.compile('[%s]' % re.escape(string.punctuation))

def __run_cmd__(cmd): 
	p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
	output, err = p.communicate()

def __tokenize__(text):
	text = text.replace("'", "")
	text = re.sub("([^0-9A-Za-z__ \t])|(\w+:\/\/\S+)"," ",text)
	text = re.sub("[0-9]+", "", text)
	words = text.split()
	words = [stemmer.stem(word.lower()) for word in words if (word.strip().lower() not in stop) and (len(word) > 1)]
	words = [re.sub(regex, '', word) for word in words]
	return words

def __get_doc_topic_distribution__(dictionary, lda_model, doc, minimum_probability=None):
	doc_bow = dictionary.doc2bow(__tokenize__(doc))
	topic_distr = lda_model.get_document_topics(doc_bow, minimum_probability=minimum_probability)
	print topic_distr
	return topic_distr

def get_features(users_tweets, model, dictionary): 
	feats = {}
	feats["control"] = {}
	feats["schizophrenia"] = {}
	for label in users_tweets.iterkeys():
		for user in users_tweets[label].iterkeys():
			feats[label][user] = {}
			doc = ""
			for tweet in users_tweets[label][user]["tweets"]:
				doc += tweet
			topics = __get_doc_topic_distribution__(dictionary, model, doc)
			sorted_topics = sorted(topics, key=lambda x: x[0])
			feats[label][user]["topics"] = sorted_topics
			num_sig = 0
			for topic_id, weight in sorted_topics: 
				if weight > .05: num_sig += 1
			feats[label][user]["num_sig_topics"] = num_sig 
	return feats

