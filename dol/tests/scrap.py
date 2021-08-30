from dol import Store
import pickle

s = Store({"a": 1, "b": 2})
t = pickle.dumps(s)
ss = pickle.loads(t)

dict(s) == dict(ss)
