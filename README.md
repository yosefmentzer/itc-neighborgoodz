# Neighborgoodz
### connecting neighbors with similar buying habits

This repo contains a recommender system that takes a list of ids of groceries, the id and location of the user, and returns a list of other users with similar grocery buying habits. An inference program was served in an AWS Virtual Machine.  
This work was developed at the Israel Tech Challenge's hackathon in October, 2022, in partnership with [Jia Ying](https://www.linkedin.com/in/jia-ying-25a61418/).

**Business Problem**
Retailers, especially supermarkets, tend to offer quantity discounts, but consumers may not need or want to buy large quantities of the same product at once.

**App Idea in more detail**
The app takes a list of groceries that the user intends to buy and their location. A recommender engine will suggest other users that live nearby and have similar buying habits, so they can get in touch, benefiting from quantity discounts if they buy together.
A cool positive externality is the opportunity for new social relationships, so the app contributes both for users' financial and general wellbeing. 

**What the model does**
The model uses an input groceries dataset, with data on items purchased by customers. After one-hot-encoding and grouping by customer, we perform dimensionality reduction with TruncatedSVD.  
The client sends a request with a basket of items for a user that wants a list of neighbors with similar buying habits. The model calculates the cosine similarity between the user's basket and all other neighbors' baskets in the same location, filtering by ZIP code.  
The `ids` of neighbors whose baskets are most similar to the user's basket are returned to the client.

Dataset link
https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset

Files:
-- `inference.py`: inference program.
-- `client.py`: test client code.
-- `Groceries_dataset.csv`: input dataset.
-- `items.csv`: ids of grocery items.
-- `zipcodes.csv`: location (ZIP codes) of users.