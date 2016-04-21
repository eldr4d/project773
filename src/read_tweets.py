import gzip
import json
import os
import re
import pickle

__input_path__ = '../input/'


def load_tweets():
  tweets = {}
  for category in ['control', 'schizophrenia']:
    directory = ''
    if(category == 'control'):
      directory = 'anonymized_control_tweets'
    elif(category == 'schizophrenia'):
      directory = 'anonymized_schizophrenia_tweets'


    print 'Loading category: ' + category + ' from dir: ' + __input_path__ + directory

    users_dic = {}
    for filename in os.listdir(__input_path__ + directory):
      if(filename.endswith('tweets.gz') == False):
        continue
      user = re.sub('\.tweets\.gz$', '', filename)
      print user
      users_dic[user] = {}
      users_dic[user]["prof"] = {}
      users_dic[user]["twits"] = []

      with gzip.open(__input_path__ + directory + '/'+filename,'rb') as control:
        for line in control:
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
          users_dic[user]["twits"].append(json_line["text"])
    tweets[category] = users_dic

  tweets_file = open(__input_path__ + r'all_tweets.dic', 'wb')
  pickle.dump(tweets, tweets_file)
  tweets_file.close()
  return tweets

def get_tweets():
  tweets = {}
  if(os.path.isfile(__input_path__ + 'all_tweets.dic')):
    tweets_file = open(__input_path__ + r'all_tweets.dic', 'rb')
    tweets = pickle.load(tweets_file)
    tweets_file.close()
  else:
    tweets = load_tweets()

  print len(tweets)
  print len(tweets['control'])
  print len(tweets['schizophrenia'])

get_tweets()