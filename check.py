import pickle
import numpy as np

with open("/home/hjliu/project/rna-backbone-design/test1/oi/2OIU.pkl", "rb") as f:
    data = pickle.load(f)

for k, v in data.items():
    if hasattr(v, "shape"):
        print(k, type(v), v.shape)
    elif isinstance(v, list):
        print(k, type(v), len(v), v[:10])
    else:
        print(k, type(v), v)

print(data["ss"])
print(data["residue_index"]) 
