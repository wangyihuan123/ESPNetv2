# to solve the pickle issue between python2 and python3
# https://stackoverflow.com/a/43290778
# dumb work around: save in python3 and load from python2
import pickle

with open("a.pkl", "rb") as f:
    w = pickle.load(f)

pickle.dump(w, open("a_py2.pkl","wb"), protocol=2)