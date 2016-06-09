import numpy as np 
import pandas as pd 

import sys
sys.path.append('../vectorsearch')
import nltk_helper
import LDA
from itertools import chain

# Given a review
# 1. Clean text
# 2. Derive LDA + doc2vec features
# 4. return list of matching businessses
# 5. Externally, for top ranked users, visualize each bar... 



bus_lda = LDA.LoadLDAModel('../output/LDA_model_bus.pickle')
bus_lda_topics = pd.read_pickle('../output/business_LDA_vectors.pickle')
normed_topic_vecs = np.vstack(map(lambda topic_vec: topic_vec/np.sqrt(np.dot(topic_vec, topic_vec)),
                        bus_lda_topics.topic_vector))


def GetDocTopic(review):
    '''
    Given a raw review, GetDocTopic cleans the data and generates the LDA topics 
    
    Returns
    -----------------
    bus_topics: np.array(n_topics)
        coefficients for the topics of interest
    '''
    # Clean and tokenize the raw text
    cleaned_review = nltk_helper.clean_and_tokenize(review,)
    
    # Chain the sentences together for LDA BOW approach.  word2vec requires sentence sep...
    cleaned_bow = [" ".join(list(chain.from_iterable(cleaned_review)))]
    # Get the topic distribution for the review in the business space. 
    feature_vec = bus_lda.tf_vectorizer.transform(cleaned_bow)
    bus_topics = bus_lda.lda.transform(feature_vec)  
    
    # Return the normalized topic. 
    return bus_topics[0]/np.sqrt(np.dot(bus_topics[0], bus_topics[0]))

def GetTopicWords(topic_idx, n_top_words=10):
    tf_feature_names = bus_lda.tf_vectorizer.get_feature_names()
    topic = bus_lda.lda.components_[topic_idx]
    return [tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]


def FindBusinessSimilarityLDA(rev_topic, business_ids=None, top_n=10):
    '''
    Get the cosine similarity (dot) of the review topic vector onto
    each business contained in business_ids
    
    Parameters
    ------------------
    rev_topic: np.array(n_topics)
        topic vector corresponding to the array. 
    
    business_ids: list
        A list of business_ids to search.  If None, all businesses are included.
    
    Returns
    ----------------
    bus_ids: np.array(top_n)
        business_ids for the top ranked businesses. 
        
    bus_similarities: np.array(top_n)
        LDA cosine similarites for the top_n businesses. 
    '''
    
    # TODO: NEED TO LIMIT SEARCH TO BUSINESS_IDS. 
    # Find the indices of businesses 
#     if business_ids is None:
#         bus_indices = bus_lda_topics.business_id.index
# #     else: 
# #         bus_ids = 


    # Normalize the input review topic 
    rev_topic_normed = rev_topic/np.sqrt(np.dot(rev_topic,rev_topic))
    # Get cosine product
    bus_similarities = np.dot(normed_topic_vecs, rev_topic_normed)
    # Find the top_n entries. 
    top_n_topic_indices = bus_similarities.argsort()[-top_n:][::-1]
    # Return top_n business_ids and simlarities.
    return bus_lda_topics.business_id.iloc[top_n_topic_indices].values,  bus_similarities[top_n_topic_indices] 