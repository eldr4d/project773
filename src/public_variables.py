__input_path__ = '../input/'

__file_to_save_tweets__ = 'all_tweets_from_stream.dic'

__has_tag__ = '__HT__'
__urls__ = '__URL__'
__users_mention__ = '__UM__'
__symbols__ = '__sm__'

__RT_start__ = 'RT @'

__unigrams_file__ = 'unigram_twitter_tweets.dic'
__bigrams_file__ = 'bigram_twitter_tweets.dic'
__trigrams_file__ = 'trigram_twitter_tweets.dic'
__perplexity_file__ = 'perplexity_twitter_tweets.dic'

__lda_dict__ = __input_path__ + "lda.qntfy.all.1.dic"
__lda_model__= __input_path__ + "lda.qntfy.all.1.70.gensim"

__liwc__= __input_path__ + "LIWC2007_updated.dic"

__path_to_tweeboparser__ = '../tools/ark-tweet-nlp-0.3.2.jar'

__unknown_word__ = '__UNKNOWN__'

#General
CSV_PATH = "../../../input/anonymized_user_manifest.csv"
DB_PATH = "../../../output/tweets.db"

#Tweebo
TWEEBO_INPUT = "../../../output/tweebo/tweets_for_tweebo_%s.txt"
TWEEBO_OUTPUT = "../../../output/tweebo/tweets_for_tweebo_%s.txt.predict"

#Doc2Vec
DOC2VEC_WIDTH = 100
DOC2VEC = "../../../doc2vec.d2v"
DOC2VEC_K = 50 #number of clusters
DOC2VEC_KMEANS = "../../../output/doc2vec_kmeans.pickle"
DOC2VEC_FEATURES = "../../../output/doc2vec_features.pickle"

#Word2Vec
WORD2VEC = "../../../output/word2vec.w2v"
WORD2VEC_WIDTH = 100
WORD2VEC_K = 75 #number of clusters
WORD2VEC_KMEANS = "../../../output/word2vec_kmeans.pickle"
WORD2VEC_FEATURES = "../../../output/word2vec_features.pickle"
