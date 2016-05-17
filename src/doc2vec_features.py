import public_variables
import pickle

def add_features(user, user_features):
    with open(public_variables.DOC2VEC_FEATURES,"rb") as fin:
        feats = pickle.load(fin)[user]["X"]
        for f in feats:
            user_features.append(f)

def add_shift(user, user_features):
    with open(public_variables.DOC2VEC_SHIFTS,"rb") as fin:
        dict = pickle.load(fin)
        if user in dict:
            feats = dict[user]["X"]
            user_features.append(feats[0])
        else:
            user_features.append(0)
