import numpy as np 
import pandas as pd 

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import sys
sys.path.append('../vectorsearch')
import nltk_helper
import LDA
from itertools import chain
import os 
# Given a review
# 1. Clean text
# 2. Derive LDA + doc2vec features
# 4. return list of matching businessses
# 5. Externally, for top ranked users, visualize each bar... 

path = "/".join(os.path.realpath(__file__).split("/")[:-2]) 

bus_lda = LDA.LoadLDAModel(path+'/output/LDA_model_bus.pickle')
bus_lda_topics = pd.read_pickle(path+'/output/business_LDA_vectors.pickle')
normed_topic_vecs = np.vstack(map(lambda topic_vec: topic_vec/np.sqrt(np.dot(topic_vec, topic_vec)),
                        bus_lda_topics.topic_vector))
from matplotlib import pyplot as plt


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



def visualize_topic(topic_vector, num_topics=6, save_path=None, top_topics=None):
    '''

    topic_vector : vector of topics to plot. 
    num_topics: number of topics to project into
    save_path: where to save the plotted figure
    top_topics: If not None, specify the topics indices of interest.

    '''

    fig = plt.figure(figsize=(5,5), dpi=80)
    for radius in np.linspace(0,1,5)[1:]:
        circ = plt.Circle((0,0),radius=radius, fill=0)
        plt.gca().add_artist(circ)
    
    # Plot the lines 
    delta_theta = 2*np.pi/num_topics
    for i in range(num_topics):
        plt.plot([0,np.cos(delta_theta*i)],[0,np.sin(delta_theta*i)], color='k')
    
    # Get the most representative topics for the query. 
    if top_topics is None:
        top_n_topics = topic_vector.argsort()[-num_topics:][::-1]
    else:
        top_n_topics = top_topics

    # Copy the topic vector excluding the uninsteresting topics.
    # Also, norm the topics. 
    #topic_vector_copy = topic_vector.copy()
    topic_vector_copy = np.array([topic if i in top_n_topics else 0 for i, topic in enumerate(topic_vector) ])
    print topic_vector_copy
    topic_vector_copy /= np.sqrt(np.dot(topic_vector_copy, topic_vector_copy))
    # Offset a bit to avoid crowding points at the origin. 
    topic_vector_copy[topic_vector_copy<.1] = .1
    # Plot the topics and the topic vector.
    for i, topic in enumerate(top_n_topics):
        x, y = np.cos(delta_theta*i + delta_theta*0.5), np.sin(delta_theta*i + delta_theta*0.5)
        plt.scatter(topic_vector_copy[topic]*x, topic_vector_copy[topic]*y, color='steelblue', s=100)
        
        words = GetTopicWords(topic, )
        if 'food' in words:
            words.remove('food')
        plt.text(1.3*x, 1.3*y, "\n".join(words[:3]), horizontalalignment='center',
                             va='center', fontsize=14)
    
    plt.axis('equal')
    plt.ylim(-1.25,1.25)
    plt.xlim(-1.25,1.25)
    plt.gca().axis('off')
    print 'Saving image to path ', save_path
    fig.savefig(save_path, transparent=True)
    # if save_path is not None:
        
    #     plt.close(fig)    # close the figure
    return 
    