import gzip
import pickle
import os
import public_variables as pv

def substitute_string(text, start, stop, new_text):
  return text[:start] + new_text + text[stop:]

def replace_entities(tweet, entities):
  indices = []
  tags = []
  # print entities
  for entity in entities:
    # print entity
    for obj in entities[entity]:
      # print obj
      indices.append(obj["indices"])
      if entity == "user_mentions":
        tags.append(pv.__users_mention__)
      elif entity == "urls":
        tags.append(pv.__urls__)
      elif entity == "hashtags":
        tags.append(pv.__has_tag__)
      else:
        tags.append(pv.__symbols__)
  ind = sorted(range(len(indices)), key=lambda k: indices[k], reverse=True)
  # print twit
  for i in ind:
    tweet = substitute_string(tweet, indices[i][0], indices[i][1], tags[i])
  # print twit
  # print indices
  # print ind
  # print tags
  return tweet

def load_tweets_from_twitter():
  tweets = []
  if(os.path.isfile(pv.__input_path__ + pv.__file_to_save_tweets__)):
    tweets_file = open(pv.__input_path__ + pv.__file_to_save_tweets__, 'rb')
    tweets = pickle.load(tweets_file)
    tweets_file.close()
  else:
    print("You need first to collect tweets from tweeter")
  print(len(tweets))
  return tweets