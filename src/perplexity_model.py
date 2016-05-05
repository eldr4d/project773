# Threading example
import operator
import re
import sys
import os
import math
import pickle

import read_dataset as rd
import helper_functions as hf
import public_variables as pv

test_tweets = {}
twitter_tweets = {}
train_dic = {}
all_unigrams = {}

def calculate_perplexity(truth, model):
  C_H = 0
  for word in model.iterkeys():
    C_H = C_H - truth[word]*math.log(model[word],2)

  return str(math.pow(2,C_H))
  
def calculate_unigrams(tweets):
  unigrams = {}
  # Join all tweets 
  line = " ".join(line.strip() for line in tweets)

  # Filter out non-letter symbols (comma, etc.)
  line = re.sub(r'[^\w\-_ ]', '', line)

  # Convert uper-case to lower-case
  line = line.lower()

  # Remove extra spaces
  line = re.sub(' +',' ',line)

  # Go to words now:
  for word in line.split(" "):
    if word not in unigrams:
      unigrams[word] = 1
    else:
      unigrams[word] = unigrams[word] + 1

  return unigrams

def get_unigrams_of_stream(twitter_tweets):
  if(os.path.isfile(pv.__input_path__ + pv.__unigrams_file__)):
    unigrams_file = open(pv.__input_path__ + pv.__unigrams_file__, 'rb')
    unigrams = pickle.load(unigrams_file)
    unigrams_file.close()
    return unigrams
  else:
    unigrams = calculate_unigrams(twitter_tweets)
    unigrams_file = open(pv.__input_path__ + pv.__unigrams_file__, 'wb')
    pickle.dump(unigrams, unigrams_file)
    unigrams_file.close()
    return unigrams

def calculate_bigrams(tweets):
  bigrams = {}
  for line in tweets:
    # Filter out non-letter symbols (comma, etc.)
    line = re.sub(r'[^\w\-_ ]', '', line)

    # Convert uper-case to lower-case
    line = line.lower()

    # Remove extra spaces
    line = re.sub(' +',' ',line)

    line = "__start__ " + line + " __end__"

    previous_word = ""
    current_word = ""
    for word in line.split(' '):
      # Calculate bigrams
      current_word = word
      if(previous_word != ""): 
        if(current_word == ""):
          continue
        key = previous_word + " " + current_word
        if(key not in bigrams):
          bigrams[key] = 1
        else:
          bigrams[key] = bigrams[key]+1
      previous_word = current_word

  return bigrams

def smooth_and_normalize(model, new_tokens):
  total_tokens = len(model)*1.0
  for token in new_tokens:
    if token not in model:
      model[token] = 1/total_tokens

  t_values = sum(model.values())*1.0
  for token in model.iterkeys():
    model[token] = model[token]/t_values

  return model

def get_bigrams_of_stream(twitter_tweets):
  if(os.path.isfile(pv.__input_path__ + pv.__bigrams_file__)):
    bigrams_file = open(pv.__input_path__ + pv.__bigrams_file__, 'rb')
    bigrams = pickle.load(bigrams_file)
    bigrams_file.close()
    return bigrams
  else:
    bigrams = calculate_bigrams(twitter_tweets)
    bigrams_file = open(pv.__input_path__ + pv.__bigrams_file__, 'wb')
    pickle.dump(bigrams, bigrams_file)
    bigrams_file.close()
    return bigrams

def extract_all_words(users_tweets):
  all_words = set()
  for key_1 in users_tweets.iterkeys():
    for key_2 in users_tweets[key_1].iterkeys():
      for tweet in users_tweets[key_1][key_2]["tweets"]:
        for word in tweet:
          all_words.add(word)
  return all_words

def perplexity_of_users(unigrams, bigrams, users_tweets):
  perplexity = {}
  all_words = extract_all_words(users_tweets)

  unigrams = smooth_and_normalize(unigrams, all_words)
  all_unig = list(unigrams.keys())

  bigrams = smooth_and_normalize(bigrams, all_words)
  all_bigr = list(bigrams.keys())

  for key_1 in users_tweets.iterkeys():
    perplexity[key_1] = {}
    for key_2 in users_tweets[key_1].iterkeys():
      unig_user = calculate_unigrams(users_tweets[key_1][key_2]["tweets"])
      unig_user = smooth_and_normalize(unig_user, all_unig)

      bigr_user = calculate_bigrams(users_tweets[key_1][key_2]["tweets"])
      bigr_user = smooth_and_normalize(bigr_user, all_bigr)

      perplexity_unig = calculate_perplexity(unig_user, unigrams)
      perplexity_bigr = calculate_perplexity(bigr_user, bigrams)

      perplexity[key_1][key_2] = {}
      perplexity[key_1][key_2]["unigrams"] = perplexity_unig
      perplexity[key_1][key_2]["bigrams"] = perplexity_bigr

  return perplexity


def get_perplexity_of_users(unigrams, bigrams, users_tweets):
  if(os.path.isfile(pv.__input_path__ + pv.__perplexity_file__)):
    perplexity_file = open(pv.__input_path__ + pv.__perplexity_file__, 'rb')
    perplexity = pickle.load(perplexity_file)
    perplexity_file.close()
    return perplexity
  else:
    perplexity = perplexity_of_users(unigrams, bigrams, users_tweets)
    perplexity_file = open(pv.__input_path__ + pv.__perplexity_file__, 'wb')
    pickle.dump(perplexity, perplexity_file)
    perplexity_file.close()
    return perplexity

def main(argv):
  twitter_tweets = hf.load_tweets_from_twitter()
  unigrams = calculate_unigrams(twitter_tweets)
  bigrams = calculate_bigrams(twitter_tweets)
  users_tweets = rd.get_tweets()

  perplexity = get_perplexity_of_users(unigrams, bigrams, users_tweets)
  print perplexity


if __name__ == "__main__":
  main(sys.argv[1:])

