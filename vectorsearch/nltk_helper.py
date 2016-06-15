# Make sure that the query input is run through a stemmer and MWE so the vocab is matching.
from nltk.tokenize import MWETokenizer
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
import string
import os 
import nltk
whitespace_tokenizer = WhitespaceTokenizer()
wnl = WordNetLemmatizer()

from spacy.en import English
nlp = English()


# This is for multi-word-phrases. 
MWE = [] 
path = "/".join(os.path.realpath(__file__).split("/")[:-2]) + '/input/'
print 'path', path
with open(path+'STREUSLE2.1-mwes.tsv') as f:
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



sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
whitespace_tokenizer = WhitespaceTokenizer()


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

    noun_chunks = map(lambda x: get_noun_chunks(x.decode('unicode-escape')), text_clean)
    noun_chunks = [x for x in noun_chunks if x != []]

    # Multiword expression tokenizer
    text_tokenize = whitespace_tokenizer.tokenize(text_cleaned)
    #text_tokenize = MWE_tokenizer.tokenize(text_tokenize)

    # remove stop words
    text_filtered = [word for word in text_tokenize if word not in stops]
    # Stem words
    
    # text_stemmed = map(lambda x: [stemmer.stem(word) for word in x], text_filtered)
    # unstem with the simplest word.  This helps readability of results...
    text_stemmed = [wnl.lemmatize(word) for word in text_filtered]
    
    noun_chunks = map(lambda x: get_noun_chunks(x), text_stemmed)
    text_stemmed = text_stemmed + noun_chunks
    return text_stemmed



#  Adapted, but much improved from  ----   https://github.com/titipata/yelp_dataset_challenge

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
    
    noun_chunks = map(lambda x: get_noun_chunks(x.decode('unicode-escape')), text_clean)
    noun_chunks = [x for x in noun_chunks if x != []]
    
    # Multiword expression tokenizer
    text_tokenize = map(lambda x: whitespace_tokenizer.tokenize(x), text_clean)
    #text_tokenize = map(lambda x: MWE_tokenizer.tokenize(x), text_tokenize)
    
    # remove stop words
    text_filtered = map(lambda x: [word for word in x if word not in stops], text_tokenize)
    # lemmetize words (stemming removes too much...)
#     text_stemmed = map(lambda x: [wnl.lemmatize(word) 
#                                   if wnl.lemmatize(word).endswith('e') 
#                                   else stemmer.stem(word) 
#                                   for word in x], text_filtered)
    text_stemmed = map(lambda x: [wnl.lemmatize(word) for word in x], text_filtered)
    #text_stemmed = map(lambda x: [stemmer.stem(word) for word in x], text_filtered)

    noun_chunks = map(lambda x: get_noun_chunks(x), text_stemmed)
    text_stemmed = text_stemmed + noun_chunks

    return text_stemmed




if __name__ == "__main__":
    clean_nltk(['bars around here',])    
