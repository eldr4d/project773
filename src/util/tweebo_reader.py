import util.reader
import nltk.corpus.reader
import codecs
import util.tweet_cleaner
class TweeboReader(object):

    COLUMNS = ["index","word","lemma","pos1","pos2","rel","dep","feats"]

    def __init__(self, fmt):
        self.fmt=fmt

    def parse_line(self, line):
        s = line.rstrip().split("\t")
        if(len(s) != len(TweeboReader.COLUMNS)):
            print("Invalid length")
            raise Exception("Invalid length")
        h = {}
        for i, c in enumerate(TweeboReader.COLUMNS):
            h[c] = s[i]
        h["clean"] = util.tweet_cleaner.TweetCleaner.clean_word(h["word"])
        return h

    def parsed_tweets(self, usr):
        file = self.fmt % usr
        stack = []
        with codecs.open(file,'r',encoding='utf8') as fin:
            for line in fin.readlines():
                if line == "\n":
                    yield stack
                    stack = []
                else:
                    stack.append(self.parse_line(line))
        if(any(stack)):
            yield stack





if __name__ == "__main__":
    file = "../../input/anonymized_user_manifest.csv"
    rdr = util.reader.Reader(file)
    fmt = "../../../output/tweebo/tweets_for_tweebo_%s.txt.predict"
    trdr = TweeboReader(fmt)
    count = 0
    for usr in rdr.read_csv():
        print("Reading %s" % usr['anonymized_name'])
        for t in trdr.parsed_tweets(usr['anonymized_name']):
            count+=1

    print("Read %i tweets" % count)
