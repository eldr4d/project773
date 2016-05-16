"""
Removes special words from tweets and converts to lower case
"""

import re

class TweetCleaner:
    EXCLUDE = "schiz" #regex to exclude tweets
    URL = "__url__"
    USER = "__user__"
    HASHTAG = "__tag__"
    NUMBER = "__num__"
    SPECIAL = [URL, USER, HASHTAG, NUMBER]
    @staticmethod
    def clean_whitespace(str):
        return re.sub("\\s+"," ",str)
    @staticmethod
    def clean(sentence):
        for word in sentence:
            yield TweetCleaner.clean_word(word)
    @staticmethod
    def clean_word(word):
        if re.match("^@.*", word):
            return TweetCleaner.USER
        elif re.match("https?:.*|ftp:.*",word):
            return TweetCleaner.URL
        elif re.match("^#.*", word):
            return TweetCleaner.HASHTAG
        elif re.match("^\\d+$", word):
            return TweetCleaner.NUMBER
        else:
            return word.lower()
    @staticmethod
    def is_special(word):
        return word in TweetCleaner.SPECIAL
    @staticmethod
    def not_special(word):
        return not (word in TweetCleaner.SPECIAL)
    @staticmethod
    def count_words(tweet):
        count=0
        for word in tweet:
            if TweetCleaner.not_special(word["clean"]):
                count+=1
        return count
    @staticmethod
    def word_string(tweet):
        return " ".join(word["clean"] for word in tweet)
    @staticmethod
    def regex_exclude(tweet):
        return re.search(TweetCleaner.EXCLUDE, TweetCleaner.word_string(tweet))
    @staticmethod
    def should_include(tweet):
        """
        Determine if tweet should be included in training
        :param tweet:
        :return:
        """
        return TweetCleaner.is_english(TweetCleaner.word_string(tweet)) and (not TweetCleaner.regex_exclude(tweet)) and (TweetCleaner.count_words(tweet) > 4)
    @staticmethod
    def is_character(c):
        return re.match("[a-zA-Z\\s]",c)
    @staticmethod
    def is_english(text):
        """
        filter for >= 2/3 english letters
        :return:
        """
        a = len(text)
        b = len(list(c for c in text if TweetCleaner.is_character(c)))
        return b/a >= 2/3