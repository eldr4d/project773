# -*- coding: utf-8 -*-
"""
Created on Sun May 15 21:24:24 2016

@author: Vasudha
"""



import gensim
from collections import OrderedDict
import numpy as np
from scipy import spatial

from topic_features import __tokenize__



def tok(tweets):
    tokens = []
    for item in tweets:
        loop = __tokenize__(item)
        tokens.append(loop)
    return tokens

def Bedi_LSA(tweets):
    tokens = []
    for item in tweets:
        loop = __tokenize__(item)
        tokens.append(loop)
    
    words = [item for sublist in tokens for item in sublist]
    
    
    id2word = gensim.corpora.Dictionary(tokens)
    corpus = [id2word.doc2bow(b) for b in tokens]
    tfidf = gensim.models.TfidfModel(corpus)
    tf = []
    for sentence in corpus:
        toop = tfidf[sentence]
        tf.append(toop)

    lsi = gensim.models.lsimodel.LsiModel(corpus=tf, id2word=id2word)
    word_vecs = lsi.projection.u
    vecs = word_vecs.tolist()
    word_vec_dic = OrderedDict(zip(words, vecs))
    return word_vec_dic 

def first_order_coherence(dic,tweets):
    
    word_to_vec = [[np.array(dic[x])for x in item if x in dic] for item in tweets]
    ve = [sum(y)/len(y) for y in word_to_vec if len(y) > 0]
    summary_vec = [(1-spatial.distance.cosine(t,s)) for s, t in zip(ve, ve[1:])]
    FOH = np.average(summary_vec) 
    #foh_min = [np.minimum(x) for x in summary_vec]
    foh_median = np.median(summary_vec) 
    foh_std = np.std(summary_vec) 
    return FOH,foh_median,foh_std

def sec_order_coherence(dic,tweets):
    word_to_vec = [[np.array(dic[x]) for x in item if x in dic] for item in tweets]
    tweet_vec = [sum(x)/len(x) for x in word_to_vec if len(x) > 0]
    summary_vec = [(1-(spatial.distance.cosine(tweet_vec[i],tweet_vec[i+2]))) for i in range(len(tweet_vec)-2)] 
    SOH = np.average(summary_vec) 
    #soh_min = [np.minimum(x) for x in summary_vec]
    soh_median = np.median(summary_vec) 
    soh_std = np.std(summary_vec) 
    return SOH,soh_median,soh_std

def get_coherence(users_tweets):
    coherence = {}
    coherence["control"] = {}
    coherence["schizophrenia"] = {}
    
    for key in users_tweets.keys():
        for k in users_tweets[key].keys():
            coherence[key][k] = {}
            
            LSA_model = Bedi_LSA(users_tweets[key][k]["tweets"])
            FOC = first_order_coherence(LSA_model,tok(users_tweets[key][k]["tweets"]))
            SOC = sec_order_coherence(LSA_model,tok(users_tweets[key][k]["tweets"]))
           
            coherence[key][k]["avg_FOC"] = FOC[0]
            coherence[key][k]["median_FOC"] = FOC[1]
            coherence[key][k]["std_FOC"] = FOC[2]

            coherence[key][k]["avg_SOC"] = SOC[0]
            coherence[key][k]["median_SOC"] = SOC[1]
            coherence[key][k]["std_SOC"] = SOC[2]
