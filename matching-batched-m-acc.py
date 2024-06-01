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

dataset1 = dataset1[['ID', 'Signature_Norm-200', 'Signature_Norm-50']]

#print(dataset1.head())

cluster_dataset2 = pd.read_pickle('out/df_fuzzy_names_ncvoter2017_10k_lsh200-50-100_c50.pkl')
print("dataset 2 loaded")

#print(cluster_dataset2.head())

cluster_dataset2_IDs = pd.read_pickle('out/df_fuzzy_names_ncvoter2017_10k_lsh200-50-100_c50_IDs.pkl')
print("dataset 2 IDs loaded")

cluster_dataset2_IDs=cluster_dataset2_IDs.drop('Cluster_Id', axis=1).to_numpy()

#print(cluster_dataset2_IDs.head())

max_cluster_size = len(cluster_dataset2.columns) - 1
print("max_cluster_size", max_cluster_size)

cluster_dataset3 = cluster_dataset2.drop('Cluster_Id', axis=1)

cluster_centroids = np.array(np.load("out/df_fuzzy_names_ncvoter2017_10k_lsh200-50-100_c50_centroids.npy"))
print("dataset 2 centroids loaded")

no_of_clusters = len(cluster_centroids)

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

scaler = StandardScaler()
normalised_signatures = dataset1['Signature_Norm-200']
normalised_signatures = np.array([x for x in normalised_signatures.to_numpy()])
normalised_signatures_scaled = scaler.fit_transform(normalised_signatures)

query_IDs = dataset1['ID']
query_IDs = np.array([x for x in query_IDs.to_numpy()])

normalised_signatures_50 = dataset1['Signature_Norm-50']
normalised_signatures_50 = np.array([x for x in normalised_signatures_50.to_numpy()])

no_of_queries = 1000
querier_IDs = query_IDs[0:no_of_queries]
enc_querier = ts.ckks_tensor(context, normalised_signatures_50[0:no_of_queries], None, True)
enc_querier_scaled = ts.ckks_tensor(context, normalised_signatures_scaled[0:no_of_queries], None, True)

writeCkks(enc_querier.serialize(), "out/enc_querier")
writeCkks(enc_querier_scaled.serialize(), "out/enc_querier_scaled")

start_time = time.time()

enc_querier_scaled.mul_(cluster_centroids)
enc_querier_scaled.sum_(axis=2)

print("Returned scores", np.array(enc_querier_scaled.decrypt().tolist()).shape)

print("first round", time.time() - start_time)

writeCkks(enc_querier_scaled.serialize(), "out/cos_sim_with_centroids")

dec_cos_sim_with_centroids=enc_querier_scaled.decrypt().tolist() 

most_matching_cluster = []
for i in range(no_of_queries):
    most_matching_cluster.append(np.argmax(dec_cos_sim_with_centroids[i]))

print("Most matching cluster ", most_matching_cluster)

sign_of_centroids = np.zeros((no_of_queries, no_of_clusters))
for i in range(no_of_queries):
    sign_of_centroids[i][most_matching_cluster[i]] = 1

enc_sign_of_centroids = ts.ckks_tensor(context, sign_of_centroids, None, True)
writeCkks(enc_sign_of_centroids.serialize(), "out/enc_sign_of_centroids")

start_time = time.time()

res = []
for i in range(max_cluster_size):
    col = np.array([x for x in cluster_dataset3["Item_"+str(i)].to_numpy()]).transpose()

    inner_time = time.time()
    
    enc_sign_of_centroids_tmp = enc_sign_of_centroids + 0
    
    enc_sign_of_centroids_tmp.mul_(col).sum_(axis=2)    
    enc_sign_of_centroids_tmp.mul_(enc_querier).sum_(axis=1)
    
    writeCkks(enc_sign_of_centroids_tmp.serialize(), "out/cs")
    print(time.time() - inner_time)

    cs = enc_sign_of_centroids_tmp.decrypt().tolist()
    res.append(cs)
    print(i, " CS: ", cs)

print(time.time() - start_time)    
    
res = np.array(res).transpose()
res[res > 1.01] = 0

TP = 0
FP = 0
TN = 0
FN = 0
for i in range(no_of_queries):
    most_matching_cs_index = np.argmax(res[i])
    if res[i][most_matching_cs_index] > 0.9:
        if querier_IDs[i] == cluster_dataset2_IDs[most_matching_cluster[i]][most_matching_cs_index]:
            TP += 1
        else:
            FP += 1
    else:
        if querier_IDs[i] in cluster_dataset2_IDs:
            FN += 1
        else:
            TN += 1

print("TP =", TP, ", FP =", FP)
print("TN =", TN, ", FN =", FN)
print(time.time() - start_time)