# Project 773

In this project we will try to classify Twitter users as schizophrenic or not.

# Data initialization
Extract the tar file and move all the files from the data/clpsych2015/schizophrenia inside the input folder.
In order to get all the tweets in python call the get_tweets() function from the read_tweets.py file. Keep in mind that we discard a lot of information from the actual data files.

# For LDA and Pennebaker features:
Download the file Prof. Resnik links below (LIWC2007_updated.dic) and place it in the ./input dir.

https://piazza.com/class/ijw37b2d7ku236?cid=51

Before running classifier.py please re-load the all_tweet.dic dictionary by either deleting the all_tweets.dic file from ./inputs or uncommenting the call to 'load_tweets()' at the bottom of read_dataset.py and running read_dataset directly. The new dict will include a list of timestamps for each user that will be used to construct the Pennebaker feature. 


Running Tweebo and word2vec is a lengthy process that was broken into separate steps.
* tweebo_write_tweets.py: write tweets to text files and write a shell script to run tweebo.
(Running the shell script will run Tweebo for close to 24 hours)
* tweet_db_create.py: write parsed tweets to a database for easy access

Word2Vec:
* word2vec_train.py: train word2vec on parsed tweets from database
* word2vec_cluster.py: run kmeans on word2vec vectors
* word2vec_features.py: calculate frequencies by cluster for each user
* word2vec_stats.py: run t-tests on features
* word2vec_svm.py: run svm on features

Doc2Vec:
* doc2vec_train.py: train doc2vec on parsed tweets from database
* doc2vec_shift_calculate.py: calculate shift for each user