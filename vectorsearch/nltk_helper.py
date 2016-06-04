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


if __name__ == "__main__":
    clean_nltk(['bars around here',])    
