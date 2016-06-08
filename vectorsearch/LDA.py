from __future__ import print_function

import pandas as pd
import pickle
import numpy as np
from itertools import chain
from collections import OrderedDict
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


class LDA():

    def __init__(self, n_topics=10, n_features=5000, max_df=.75, min_df=2, max_iter=5):
        '''
        '''

        self.n_topics = n_topics
        self.n_features = n_features
        self.max_df = max_df
        self.min_df = min_df
        self.max_iter = max_iter
        self.lda = None
        self.tf = None
        self.topics = None

    def vectorizecounts(self, docs):
        '''
        '''

        # Use tf (raw term count) features for LDA.
        print("Extracting tf features for LDA...")
        self.tf_vectorizer = CountVectorizer(max_df=self.max_df, min_df=self.min_df, max_features=self.n_features)
        t0 = time()
        self.tf = self.tf_vectorizer.fit_transform(docs)
        self.n_samples = len(docs)
        print("done in %0.3fs." % (time() - t0))


    def fitLDA(self):
        '''
        '''
        print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
              % (self.n_samples, self.n_features))
        self.lda = LatentDirichletAllocation(n_topics=self.n_topics, max_iter=self.max_iter,
                                        learning_method='online', learning_offset=10.,
                                        random_state=0, n_jobs=6)
        t0 = time()
        self.topics = self.lda.fit(self.tf)
        print("done in %0.3fs." % (time() - t0))

    def print_top_words(self, n_top_words):
        '''
        '''

        tf_feature_names = self.tf_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(self.lda.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

    def get_topic_content(self, topic):
        '''
        Parameters
        --------------
        topic: int 
            Topic index


        Returns
        -----------
        feature_names : list
            Array of words corresponding to the given feature. 

        topic_content : np.array(n_features)
            Topic vector over the feature space
        '''

        return self.tf_vectorizer.get_feature_names(), self.lda.components_

    def get_doc_topics(self, doc):
        
        # Convert the document into feature space. 
        feature_vec = self.tf_vectorizer.fit_transform(doc)
        return self.lda.fit_transform(feature_vec)


def LoadLDAModel(path):
    with open(path, 'rb') as f: 
        return pickle.load(f)


def SaveLDAModel(path, lda_model):
    # Save the model 
    with open(path, 'wb') as f: 
        pickle.dump(lda_model, f)