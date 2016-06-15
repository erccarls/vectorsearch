from __future__ import print_function
import pandas as pd
import pickle
import numpy as np
from itertools import chain
from collections import OrderedDict
import random
import sys
sys.path.append('../vectorsearch/')
import LDA
from random import shuffle


n_topics=30
n_features=10000
max_df=.75
min_df=3
max_iter=10
alpha=6./n_topics


# Load the bar review dataset 
review = pd.read_pickle('../output/bar_reviews_cleaned_and_tokenized_SF.pickle')
review.head(5)
df_businesses = pd.read_pickle('../input/yelp_academic_dataset_business_SF.pickle')
city_state_list = list(set([df_businesses.city.iloc[i_city]+', '+df_businesses.state.iloc[i_city] for i_city, city in enumerate(df_businesses.city)]))[1:]
import pickle
pickle.dump(city_state_list, open('../output/city_state_list.pickle','wb'))





# Drop 20% of the users from the dataset for testing
user_set = list(set(review.user_id.values[:]))

random.seed(0)
shuffle(user_set) # Randomize 
n_users = float(len(user_set))

user_set_training = user_set[:int(n_users*float(0.8))]
with open('../output/training_users.pickle', 'wb') as f: 
    pickle.dump(user_set_training, f)
    
# Save a test set
test_users = user_set[int(n_users*float(0.8)):]
with open('../output/test_users.pickle', 'wb') as f: 
    pickle.dump(test_users, f)
    
# Make the active review set training only 
review = review[review.user_id.isin(user_set_training)]






# This is for review level not business level 
# docs = [" ".join(list(chain.from_iterable(l))) for l in review.cleaned_tokenized.iloc[:]]

n_reviews = -1 # all of them... 
# Flatten the reviews, so each review is just a single list of words.
reviews_merged_bus = OrderedDict()
business_set = set(review.business_id.values[:n_reviews])
for i_bus, bus_id in enumerate(business_set):
    if ((i_bus%2)==0):
        print ('\r Fraction Processed',float(i_bus+1)/len(business_set), end="") 
    # This horrible line first collapses each review of a corresponding business into a list
    # of lists, and then collapses the list of sentences to a long list of words
    reviews_merged_bus[bus_id] = " ".join(list(chain.from_iterable( 
                                    chain.from_iterable( review.cleaned_tokenized[review.business_id==bus_id] ))))    
docs_bus = reviews_merged_bus.values()

with open('../output/docs_bars_bus.pickle', 'wb') as f: 
    pickle.dump(docs_bus, f)

with open('../output/bus_ids_bars_LDA.pickle', 'wb') as f: 
    pickle.dump(reviews_merged_bus.keys(), f)




lda_bus = LDA.LDA(alpha=alpha, n_topics=n_topics, n_features=n_features, max_df=max_df, min_df=min_df, max_iter=max_iter,)
lda_bus.vectorizecounts(docs_bus)
lda_bus.fitLDA()
LDA.SaveLDAModel('../output/LDA_model_bus.pickle', lda_bus)


# The topic vector for a given business is given by this dataframe. 
bus_lda_ids = pickle.load(open('../output/bus_ids_bars_LDA.pickle', 'rb'))
bus_vectors = pd.DataFrame()
bus_vectors['business_id'] = bus_lda_ids
transformed = lda_bus.lda.transform(lda_bus.tf)


bus_vectors['topic_vector'] = [bus_topic_vec for bus_topic_vec in transformed]
normed_topic_vecs = map(lambda topic_vec: topic_vec/np.sqrt(np.dot(topic_vec, topic_vec)),
                        bus_vectors.topic_vector) 


bus_vectors.topic_vector = normed_topic_vecs
bus_vectors.to_pickle('../output/business_LDA_vectors.pickle')