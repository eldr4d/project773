import gzip
import json
import os
import re
import pickle

import public_variables as pv
import helper_functions as hf


def load_tweets():
  tweets = {}
  for category in ['control', 'schizophrenia']:
    directory = ''
    if(category == 'control'):
      directory = 'anonymized_control_tweets'
    elif(category == 'schizophrenia'):
      directory = 'anonymized_schizophrenia_tweets'


    print 'Loading category: ' + category + ' from dir: ' + pv.__input_path__ + directory

    users_dic = {}
    for filename in os.listdir(pv.__input_path__ + directory):
      if(filename.endswith('tweets.gz') == False):
        continue
      user = re.sub('\.tweets\.gz$', '', filename)
      print user
      users_dic[user] = {}
      users_dic[user]["prof"] = {}
      users_dic[user]["tweets"] = []
      users_dic[user]["timestamps"] = []

      with gzip.open(pv.__input_path__ + directory + '/'+filename,'rb') as in_file:
        for line in in_file:
          json_line = json.loads(line);
          if(users_dic[user]['prof'] == False):
            user_prof = {}
            ver = True
            if(json_line['entities']['user']['verified'] == "false"):
              ver = False
            user_prof['ver'] = ver
            user_prof['followers'] = json_line['entities']['user']['followers_count']
            user_prof['lang'] = json_line['entities']['user']['lang']
            user_prof['friends'] = json_line['entities']['user']['friends_count']
            user_prof['joined'] = json_line['entities']['user']['created_at']
            users_dic[user]['prof'] = user_prof
          
          #ignore retweets
          if(json_line["text"].startswith(pv.__RT_start__) == False):
            tweet = hf.replace_entities(json_line["text"], json_line["entities"])
            users_dic[user]["tweets"].append(tweet)
            users_dic[user]["timestamps"].append(json_line['created_at'])
    tweets[category] = users_dic

  tweets_file = open(pv.__input_path__ + r'all_tweets.dic', 'wb')
  pickle.dump(tweets, tweets_file)
  tweets_file.close()
  return tweets

def get_tweets():
  tweets = {}
  if(os.path.isfile(pv.__input_path__ + 'all_tweets.dic')):
    tweets_file = open(pv.__input_path__ + r'all_tweets.dic', 'rb')
    tweets = pickle.load(tweets_file)
    tweets_file.close()
  else:
    tweets = load_tweets()

  print len(tweets)
  print len(tweets['control'])
  print len(tweets['schizophrenia'])
  return tweets

# load_tweets()
# get_tweets()