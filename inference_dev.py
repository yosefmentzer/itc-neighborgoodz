"""
Inference module
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask
from flask import request

DATA_FILENAME = 'Groceries_dataset.csv'
SVD_COMPONENTS = 30
SVD_NITER = 7
ITEMS_FILENAME = 'items.csv'
N_NEIGHBORS = 5

# REQUEST INPUT
items_list = [5, 6, 7, 8, 9, 10, 50]
user_id = 1128
###################

df = pd.read_csv('Groceries_dataset.csv')
df_s = pd.concat([df, pd.get_dummies(df['itemDescription'])], axis=1).drop(columns=['itemDescription'])
df_m = df_s.groupby(by=['Member_number']).sum()

neighbor_ids = df_m.index

svd = TruncatedSVD(n_components=SVD_COMPONENTS, n_iter=SVD_NITER)
svd.fit(df_m)
df_svd = pd.DataFrame(svd.transform(df_m))

# get sample
items = pd.read_csv(ITEMS_FILENAME, index_col='id')
basket_df = items.iloc[items_list]
sample = pd.Series(df_m.columns).isin(list(basket_df['name'].values)).astype(int).values

# transform sample
sample_t = svd.transform(sample.reshape(1, -1))
# similarity
similarity = cosine_similarity(sample_t, df_svd)
similarity_df = pd.DataFrame(cosine_similarity(sample_t, df_svd)[0], index=df_m.index, columns=['similarity'])
# exclude user from neighbors df
if user_id in similarity_df.index:
    similarity_df = similarity_df.drop([user_id])
top5 = similarity_df.sort_values(by='similarity', ascending=False).head(N_NEIGHBORS)
top5_ids = list(top5.index)

print(top5_ids)
print(top5)
