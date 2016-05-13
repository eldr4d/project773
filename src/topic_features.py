# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
from nltk.stem.porter import *

from gensim import corpora, models, similarities, matutils

import numpy as np

from datetime import date

import re, string, subprocess, os, time

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

def __tokenize_and_clean__(text):
	text = text.replace("'", "")
	text = re.sub("([^0-9A-Za-z__ \t])|(\w+:\/\/\S+)"," ",text)
	text = re.sub("[0-9]+", "", text)
	words = text.split()
	words = [stemmer.stem(word.lower()) for word in words if (word.strip().lower() not in stop) and (len(word) > 1)]
	words = [re.sub(regex, '', word) for word in words]
	return words

def __get_doc_topic_distribution__(dictionary, lda_model, doc, minimum_probability=None):
	doc_bow = dictionary.doc2bow(__tokenize_and_clean__(doc))
	topic_distr = lda_model.get_document_topics(doc_bow, minimum_probability=minimum_probability)
	return topic_distr

def get_features(users, users_tweets, model, dictionary, folds): 
	feats = {}
	feats["control"] = {}
	feats["schizophrenia"] = {}
	
	for label in users_tweets.iterkeys():
		for user in users_tweets[label].iterkeys():
			if users[user]["fold"] not in folds: continue
			
			feats[label][user] = {}
			doc = ""
			
			for tweet in users_tweets[label][user]["tweets"]:
				doc += tweet
			
			topics = __get_doc_topic_distribution__(dictionary, model, doc)

			feats[label][user]["topics"] = {}
			num_sig = 0
			
			for i in range(70): feats[label][user]["topics"][i] = 0
			for topic_id, prop in topics: 
				feats[label][user]["topics"][topic_id] = prop
				if prop > .05: num_sig += 1
			
			feats[label][user]["num_sig_topics"] = num_sig 
	
	return feats

