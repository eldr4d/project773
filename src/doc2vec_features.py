import public_variables
import pickle

def add_features(user, user_features):
    with open(public_variables.DOC2VEC_FEATURES,"rb") as fin:
        feats = pickle.load(fin)[user]["X"]
        for f in feats:
            user_features.append(f)