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
from scipy.stats import entropy
from numpy.linalg import norm
from gensim.models import doc2vec


# Given a review
# 1. Clean text
# 2. Derive LDA + doc2vec features
# 4. return list of matching businessses
# 5. Externally, for top ranked users, visualize each bar... 

path = "/".join(os.path.realpath(__file__).split("/")[:-2]) 
doc2vec_model = doc2vec.Doc2Vec.load(path+'/output/doc2vec_bars_100.model')
bus_lda = LDA.LoadLDAModel(path+'/output/LDA_model_bus.pickle')
bus_lda_topics = pd.read_pickle(path+'/output/business_LDA_vectors.pickle')
normed_topic_vecs = np.vstack(map(lambda topic_vec: topic_vec/np.sqrt(np.dot(topic_vec, topic_vec)),
                        bus_lda_topics.topic_vector))
# Normalize PDF!!!
normed_topic_vecs_ = (normed_topic_vecs.T/np.sum(normed_topic_vecs, axis=1)).T

from matplotlib import pyplot as plt


def GetDocTopic(review, n_jobs=1):
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
    bus_lda.lda.n_jobs = n_jobs
    feature_vec = bus_lda.tf_vectorizer.transform(cleaned_bow)
    bus_topics = bus_lda.lda.transform(feature_vec)  
    
    # Return the normalized topic. 
    return bus_topics[0]/np.sqrt(np.dot(bus_topics[0], bus_topics[0]))


def GetDocLength(review):
    '''
    Given a raw review, GetDocTopic cleans the data and generates the LDA topics 
    
    Returns
    -----------------
    bus_topics: np.array(n_topics)
        coefficients for the topics of interest
    '''
    # Clean and tokenize the raw text
    cleaned_review = nltk_helper.clean_and_tokenize(review,)
    return len(list(chain.from_iterable(cleaned_review)))



def GetTopicWords(topic_idx, n_top_words=10):
    tf_feature_names = bus_lda.tf_vectorizer.get_feature_names()
    topic = bus_lda.lda.components_[topic_idx]
    return [tf_feature_names[i] for i in topic.argsort()[-n_top_words:][::-1]#[:-n_top_words - 1:-1]]


def FindBusinessSimilarityLDA(rev_topic, business_ids=None, top_n=10, method='Hel'):
    '''
    Get the cosine similarity (dot) of the review topic vector onto
    each business contained in business_ids
    
    Parameters
    ------------------
    rev_topic: np.array(n_topics)
        topic vector corresponding to the array. 
    
    business_ids: list
        A list of business_ids to search.  If None, all businesses are included.
    
    method: 'cos' or 'JSD' for Jensen_Shannon_divergence

    Returns
    ----------------
    bus_ids: np.array(top_n)
        business_ids for the top ranked businesses. 
        
    bus_similarities: np.array(top_n)
        LDA cosine similarites for the top_n businesses. 
    '''
    

    # By default select all businesses. 
    idx = np.arange(0, len(bus_lda_topics.business_id.values))
    # If passed a list of relevant IDs, use that.
    if business_ids is not None:
        idx = np.where(bus_lda_topics.business_id.isin(business_ids))[0]

    # Normalize the input review topic 
    #rev_topic_normed = rev_topic/np.sqrt(np.dot(rev_topic,rev_topic))
    rev_topic_normed = rev_topic/np.sum(rev_topic) # Normalize PDF 
    
    # if method=='cos':
    #     # Get cosine product
    #     #print "Using Cosine Similarity"
    #     bus_similarities = np.dot(normed_topic_vecs, rev_topic_normed)
    if method=='Hel':
        # Helanger Distance....
        # Negative because unlike the cos similarity we minimize the distance....
        bus_similarities = -np.linalg.norm(np.sqrt(normed_topic_vecs) - np.sqrt(rev_topic_normed), axis=1) / np.sqrt(2)
    elif method=='JSD': 
        #print "Using Jenson-Shannon Divergence"
        bus_similarities = -JSD(normed_topic_vecs, rev_topic_normed)
    else:
        normed_topic_vecs_ = (normed_topic_vecs.T/np.sum(normed_topic_vecs, axis=1)).T
        bus_similarities = -np.array([entropy(normed_topic_vec, rev_topic_normed) for normed_topic_vec in normed_topic_vecs])
    
    # Find the top_n entries. 
    top_n_topic_indices = bus_similarities[idx].argsort()[-top_n:][::-1]
    # Return top_n business_ids and simlarities.
    return bus_lda_topics.business_id.values[idx][top_n_topic_indices],  bus_similarities[idx][top_n_topic_indices] 



def FindBusinessSimilaritydoc2vec(review_text, bus_ids_in_city_state, top_n=10):
    '''
    Given a document review, search businesses in that city, state and return the rank from the top of the list. 
    
    
    '''
    # Clean and tokenize....
    cleaned_review = nltk_helper.clean_and_tokenize(review_text,)
    # Chain the sentences together for LDA BOW approach.  word2vec requires sentence sep...
    cleaned_bow = list(chain.from_iterable(cleaned_review))
    
    doc_length = len(cleaned_bow)
    # Get doc_vector for input review.... 
    doc_vec = doc2vec_model.infer_vector(cleaned_bow,)
    # Get similarity to all businesses 
    top_bus_ids, top_sims = zip(*doc2vec_model.docvecs.most_similar(positive=[doc_vec,], 
                                           topn=len(doc2vec_model.docvecs.doctags)))
    

    #find ranking out of those that are in the city/state
    df = pd.DataFrame.from_dict(dict(bids=top_bus_ids, sims=top_sims))
    # Find the entries in the relevant city
    df = df[df.bids.isin(bus_ids_in_city_state)]
    return df.bids.values[:top_n], df.sims.values[:top_n]
    


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
    #print topic_vector_copy
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
        
    plt.close(fig)    # close the figure
    return 
    


def JSD(P, Q):
    _P = P / norm(P, ord=1, axis=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

