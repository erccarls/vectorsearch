# Make sure that the query input is run through a stemmer and MWE so the vocab is matching.
from nltk.tokenize import MWETokenizer
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
import string

whitespace_tokenizer = WhitespaceTokenizer()
wnl = WordNetLemmatizer()

# This is for multi-word-phrases. 
MWE = [] 
with open('../input/STREUSLE2.1-mwes.tsv') as f:
    for line in f.readlines():
        multiword_expression = line.split('\t')[0].split()[1:]
        MWE.append(multiword_expression)
MWE_tokenizer = MWETokenizer(MWE, separator='-')
# Add whatever additional custom multi-word-expressions.
MWE_tokenizer.add_mwe((  'dive', 'bar'))
# Stopwords
stops = set(stopwords.words("english") + stopwords.words("spanish"))
keep_list = ['after', 'during', 'not', 'between', 'other', 'over', 'under', 
             'most', ' without', 'nor', 'no', 'very', 'against','don','aren']
stops = set([word for word in stops if word not in keep_list])


table = string.maketrans("","")


def clean_text(text):
    """Clean and lower string.  Cleaning removes all punctuation
    Parameters
    ----------
        text : in string format
    Returns
    -------
        text_clean : clean text input in string format
    """

    return text.lower().translate(table, string.punctuation.replace('-',''))



def clean_nltk(string):
    '''
    Handles the NLP cleaning, stemming, and multiword expression work for a given search phrase.
    This is required to convert input into the model vocab space.

    The process includes:
        -lowercasing 
        -tokenize
        -removal of punctuation
        -multi-word-expression merging
        -stop-word-removal
        -Stemming (actually lematization)
    
    Parameters
    -------------
    string : string to tokenize.
    
    Returns
    --------------
    tokenized: Tokenized cleaned sentence
    '''
    text_cleaned = clean_text(string)
    # Multiword expression tokenizer
    text_tokenize = whitespace_tokenizer.tokenize(text_cleaned)
    text_tokenize = MWE_tokenizer.tokenize(text_tokenize)

    # remove stop words
    text_filtered = [word for word in text_tokenize if word not in stops]
    # Stem words
    
    # text_stemmed = map(lambda x: [stemmer.stem(word) for word in x], text_filtered)
    # unstem with the simplest word.  This helps readability of results...
    text_stemmed = [wnl.lemmatize(word) for word in text_filtered]
    
    #return text_stemmed
    return text_stemmed



#  Adapted, but much improved from  ----   https://github.com/titipata/yelp_dataset_challenge


import time
import collections
#import scipy.sparse as sp
#import nltk.data
from nltk.tokenize import WhitespaceTokenizer
from unidecode import unidecode
from itertools import chain
import numpy as np
#from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import MWETokenizer
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import sys
sys.path.append('../vectorsearch/')
from reverse_stemmer import SnowCastleStemmer
import nltk
import pickle
import string

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
whitespace_tokenizer = WhitespaceTokenizer()
# tb_tokenizer = TreebankWordTokenizer()
stops = set(stopwords.words("english") + stopwords.words("spanish"))
keep_list = ['after', 'during', 'not', 'between', 'other', 'over', 'under', 
             'most', ' without', 'nor', 'no', 'very', 'against','don','aren']
stops = set([word for word in stops if word not in keep_list])


# Multiword tokenizer list taken from: 
# http://www.cs.cmu.edu/~ark/LexSem/
# http://www.cs.cmu.edu/~ark/LexSem/STREUSLE2.1-mwes.tsv

# This parses a list of multiword expressions from # http://www.cs.cmu.edu/~ark/LexSem/STREUSLE2.1-mwes.tsv
# into NLTK format
MWE = [] 
with open('../input/STREUSLE2.1-mwes.tsv') as f:
    for line in f.readlines():
        multiword_expression = line.split('\t')[0].split()[1:]
        MWE.append(multiword_expression)
MWE_tokenizer = MWETokenizer(MWE, separator='-')
# Add whatever additional custom multi-word-expressions.
MWE_tokenizer.add_mwe(('dive', 'bar'))
MWE_tokenizer.add_mwe(('happy','hour'))

# Stemmer
stemmer = SnowCastleStemmer("english")
wnl = WordNetLemmatizer()
table = string.maketrans("","")

def clean_text(text):
    """Clean and lower string
    Parameters
    ----------
        text : in string format
    Returns
    -------
        text_clean : clean text input in string format
    """
    return text.lower().translate(table, string.punctuation.replace('-',''))


def clean_and_tokenize(text):
    """
    1. Divide review into sentences
    2. clean words
    3. tokenize
    4. multiword tokenize
    5. remove stop words
    6. stem words
    Returns
    ------
        text_filtered: list of word in sentence
    """
    # Splits into sentences.
    sentence = sent_detector.tokenize(unidecode(text))
    # Clean text: (remove) Remove extra puncuations marks...
    text_clean = map(clean_text, sentence)

    # Multiword expression tokenizer
    text_tokenize = map(lambda x: whitespace_tokenizer.tokenize(x), text_clean)
    text_tokenize = map(lambda x: MWE_tokenizer.tokenize(x), text_tokenize)
    
    # remove stop words
    text_filtered = map(lambda x: [word for word in x if word not in stops], text_tokenize)
    # lemmetize words (stemming removes too much...)
#     text_stemmed = map(lambda x: [wnl.lemmatize(word) 
#                                   if wnl.lemmatize(word).endswith('e') 
#                                   else stemmer.stem(word) 
#                                   for word in x], text_filtered)
    text_stemmed = map(lambda x: [wnl.lemmatize(word) for word in x], text_filtered)
    #text_stemmed = map(lambda x: [stemmer.stem(word) for word in x], text_filtered)
    return text_stemmed




if __name__ == "__main__":
    clean_nltk(['bars around here',])    
