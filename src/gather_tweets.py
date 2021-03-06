#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

import json
import pickle
import sys
import os

import public_variables as pv
import helper_functions as hp

#Variables that contains the user credentials to access Twitter API 
access_token = "225541300-tFKgt5Zn7p4hMemrSwSPObFiQsQv2oPJ3cFvvJDb"
access_token_secret = "S5UarqRnaA4xTS4iY08sz8YESjJRivuyVJkjMlWOmb600"
consumer_key = "ygRXzJAJByP3kl4L1EXltGSiG"
consumer_secret = "b7k3OR9isIbjiYLF6b99cJKHgyzKjHpG4nLDEAZ0eK6DrhJsa6"

tweets = []
total_tweets = 0

#This listener filters out Retweets and saves only the tweet
class TweetListener(StreamListener):
	def on_data(self, data):
		global total_tweets
		global tweets
		json_line = json.loads(data)

		# We hit the limit
		if("limit" in json_line):
			return True

		if(json_line["text"].startswith(pv.__RT_start__) == False):
			tweet = hp.replace_entities(json_line["text"], json_line["entities"])
			tweets.append(tweet)
			total_tweets = total_tweets + 1

		if(total_tweets%10000 == 0):
			print total_tweets
			tweets_file = open(pv.__input_path__ + pv.__file_to_save_tweets__, 'wb')
			pickle.dump(tweets, tweets_file)
			tweets_file.close()

		if(total_tweets == 3000000):
			tweets_file = open(pv.__input_path__ + pv.__file_to_save_tweets__, 'wb')
			pickle.dump(tweets, tweets_file)
			tweets_file.close()
			sys.exit(0)

		return True

	def on_error(self, status):
		print status


if __name__ == '__main__':

	#Check and load prev tweets
	if(os.path.isfile(pv.__input_path__ + pv.__file_to_save_tweets__)):
		tweets_file = open(pv.__input_path__ + pv.__file_to_save_tweets__, 'rb')
		tweets = pickle.load(tweets_file)
		tweets_file.close()
		total_tweets = len(tweets)
		print "Total tweets loaded = " + str(total_tweets)

	#This handles Twitter authetification and the connection to Twitter Streaming API
	l = TweetListener()
	auth = OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	stream = Stream(auth, l)

	try:
		stream.filter(languages=['en'], locations=[-180,-90,180,90])
	except KeyboardInterrupt:
		print "Total tweets = " + str(total_tweets)
		tweets_file = open(pv.__input_path__ + pv.__file_to_save_tweets__, 'wb')
		pickle.dump(tweets, tweets_file)
		tweets_file.close()
		sys.exit(0)
	# stream.sample(languages=['en'])



