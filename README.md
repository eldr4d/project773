# Project 773

In this project we will try to classify Twitter users as schizophrenic or not.

# Data initialization
Extract the tar file and move all the files from the data/clpsych2015/schizophrenia inside the input folder.
In order to get all the tweets in python call the get_tweets() function from the read_tweets.py file. Keep in mind that we discard a lot of information from the actual data files.

# For LDA and Pennebaker features:
Download the file Prof. Resnik links below (LIWC2007_updated.dic) and place it in the ./input dir.

https://piazza.com/class/ijw37b2d7ku236?cid=51

Before running classifier.py please re-load the all_tweet.dic dictionary by either deleting the all_tweets.dic file from ./inputs or uncommenting the call to 'load_tweets()' at the bottom of read_dataset.py and running read_dataset directly. The new dict will include a list of timestamps for each user that will be used to construct the Pennebaker feature.

# Running the classifier
To run the classifier just type:
python classifier.py

By default the feature will contain all the features that we described in our report. If you want to create feature vectors that contain only a subset of the total features you will need to set the appropriate boolean values (lines 26-33 inside the classifier.py file).

Keep in mind that the first run will be slow since it needs to create all the features. Consequent runs will be a lot faster since we cache our generated features.
