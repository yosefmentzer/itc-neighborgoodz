"""
Inference module
"""


import json
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask
from flask import request

# constants and inputs
DATA_FILENAME = 'Groceries_dataset.csv'
SVD_COMPONENTS = 30
SVD_NITER = 7
ITEMS_FILENAME = 'items.csv'
ZIPCODES_FILENAME = 'zipcodes.csv'
AWS_HOST = '0.0.0.0'
AWS_PORT = 8080

# loading, preprocessing and dimensionality reduction
df = pd.read_csv('Groceries_dataset.csv')
items = pd.read_csv(ITEMS_FILENAME, index_col='id')
zipcodes = pd.read_csv(ZIPCODES_FILENAME)

# get sparse matrix
df_s = pd.concat([df, pd.get_dummies(df['itemDescription'])], axis=1).drop(columns=['itemDescription'])
# get matrix grouped by person
df_m = df_s.groupby(by=['Member_number']).sum(numeric_only=True)
neighbor_ids = df_m.index

# perform dimensionality reduction with TruncatedSVD
svd = TruncatedSVD(n_components=SVD_COMPONENTS, n_iter=SVD_NITER)
svd.fit(df_m)
df_svd = pd.DataFrame(svd.transform(df_m))

# initialize Flask object app
app = Flask(__name__)


@app.route('/get_neighbors', methods=['POST'])
def get_neighbors():
    """
    get arg 'basket', a list of item ids
    :return: list of top5 neighbors as a JSON object
    """
    # get args
    args = request.get_json()

    items_list = json.loads(args['basket'])  # list of ids of items of input basket
    user_id = args['user_id']
    user_zip = args['user_zip']
    n = args['n']  # number of neighbors to return
    # get one-hot encoded of input basket
    basket_df = items.iloc[items_list]
    ohe_basket = pd.Series(df_m.columns).isin(list(basket_df['name'].values)).astype(int).values
    # apply dimensionality reduction to ohe_basket
    ohe_basket_t = svd.transform(ohe_basket.reshape(1, -1))
    # filter out neighbors from other zipcodes
    valid_neighbor_ids = zipcodes[zipcodes['zipcode'] == user_zip]['Member_number'].values
    mask = pd.Series(neighbor_ids).isin(valid_neighbor_ids).values
    # similarity between input basket and other baskets
    similarity = cosine_similarity(ohe_basket_t, df_svd[mask])  # filter zipcode before calculation
    similarity_df = pd.DataFrame(similarity[0], index=df_m.index[mask], columns=['similarity'])
    print(similarity_df.shape)
    # exclude user from neighbors df, so we don't recommend the user for herself
    if user_id in similarity_df.index:
        similarity_df = similarity_df.drop([user_id])
    top5 = similarity_df.sort_values(by='similarity', ascending=False).head(n)
    top5_ids = list(top5.index)
    json_obj = json.dumps(top5_ids)

    return json_obj


if __name__ == '__main__':
    app.run(host=AWS_HOST, port=AWS_PORT)
