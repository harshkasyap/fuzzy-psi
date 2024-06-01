import csv
import re
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from random import randint
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
#from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#import jellyfish
import itertools
from sklearn.metrics import jaccard_score
#from nltk import ngrams
from sklearn.preprocessing import OneHotEncoder
import string
import hashlib
import math
from numpy.linalg import norm
import time
import pickle
import matplotlib.pyplot as plt
#import faiss
import tenseal as ts
import base64
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import gc
import copy

print("Started")
dataset1 = pd.read_pickle('dataset/df_names_ncvoter2014_10k_lsh200-50-100.pkl')
print("dataset 1 loaded")

dataset1 = dataset1[['ID', 'Signature_Norm-50']]

#print(dataset1.head())

dataset2 = pd.read_pickle('dataset/df_fuzzy_names_ncvoter2017_10k_lsh200-50-100.pkl')
print("dataset 2 loaded")

dataset2 = dataset2[['ID', 'Fuzzy Signature_Norm-50']]

#print(dataset2.head())


poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
global_scale= 2**40

context = ts.context(
            ts.SCHEME_TYPE.CKKS, 
            poly_modulus_degree = poly_modulus_degree,
            coeff_mod_bit_sizes = coeff_mod_bit_sizes
            )
context.generate_galois_keys()
context.global_scale = global_scale

def writeCkks(ckks_vec, filename):
    ser_ckks_vec = base64.b64encode(ckks_vec)

    with open(filename, 'wb') as f:
        f.write(ser_ckks_vec)

def readCkks(filename):
    with open(filename, 'rb') as f:
        ser_ckks_vec = f.read()
    
    return base64.b64decode(ser_ckks_vec)

query_IDs = dataset1['ID']
query_IDs = np.array([x for x in query_IDs.to_numpy()])

normalised_signatures_50 = dataset1['Signature_Norm-50']
normalised_signatures_50 = np.array([x for x in normalised_signatures_50.to_numpy()])

fuzzy_normalised_signatures_50 = dataset2['Fuzzy Signature_Norm-50']
fuzzy_normalised_signatures_50 = np.array([x for x in fuzzy_normalised_signatures_50.to_numpy()])

fuzzy_query_IDs = dataset2['ID']
fuzzy_query_IDs = np.array([x for x in fuzzy_query_IDs.to_numpy()])

no_of_queries = 1000
querier_IDs = query_IDs[0:no_of_queries]

enc_querier = ts.ckks_tensor(context, [normalised_signatures_50[i] for i in range(no_of_queries)], None, True)

writeCkks(enc_querier.serialize(), "out/enc_querier")

start_time = time.time()

res = []
for i in range(len(fuzzy_normalised_signatures_50)):
    inner_time = time.time()

    enc_querier_tmp = enc_querier + 0

    enc_querier_tmp.mul_(fuzzy_normalised_signatures_50[i]).sum_(axis=1)
    
    writeCkks(enc_querier_tmp.serialize(), "out/cs")
    print(time.time() - inner_time)

    cs = enc_querier_tmp.decrypt().tolist()
    res.append(cs)
    print(i, " CS: ", cs)

print(time.time() - start_time)    

res = np.array(res).transpose()
res[res > 1.01] = 0
print(res.shape)

TP = 0
FP = 0
TN = 0
FN = 0
for i in range(no_of_queries):
    most_matching_cs_index = np.argmax(res[i])
    if res[i][most_matching_cs_index] > 0.8:
        if querier_IDs[i] == fuzzy_query_IDs[most_matching_cs_index]:
            TP += 1
        else:
            FP += 1
    else:
        if querier_IDs[i] in fuzzy_query_IDs:
            FN += 1
        else:
            TN += 1


print("TP =", TP, ", FP =", FP)
print("TN =", TN, ", FN =", FN)
print(time.time() - start_time)