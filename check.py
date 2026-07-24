import pickle
import numpy as np


with open("/home/hjliu/project/rna-backbone-design/test1/oi/2OIU.pkl", "rb") as f:
    data = pickle.load(f)
   

for  k , v in data.items():
    print (k , v.shape)


print (data["chain_index"])
print (data["aatype"])

    
