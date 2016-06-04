import nltk
from collections import defaultdict
import pickle

class SnowCastleStemmer(nltk.stem.SnowballStemmer):
    """ A wrapper around snowball stemmer with a reverse lookip table """
    
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self._stem_memory = defaultdict(set)
        # switch stem and memstem
        self._stem=self.stem
        self.stem=self.memstem
        
    def memstem(self, word):
        """ Wrapper around stem that remembers """
        stemmed_word = self._stem(word)
        self._stem_memory[stemmed_word].add(word)
        return stemmed_word
        
    def unstem(self, stemmed_word):
        """ Reverse lookup """
        return sorted(self._stem_memory[stemmed_word], key=len)

    def loadstemmer(self, path):
	""" Load a saved memory state """
	self._stem_memory = pickle.loads(path)

    def savestemmer(self, path):
	""" Save a stemmer  memory to file for later reverse lookup"""
	with open(path, 'wb') as f:
            pickle.dump(self._stem_memory, f)


        
if __name__=='__main__':
  stemmer= SnowCastleStemmer('english')
  stemmer.stem("building")
  stemmer.stem("build")
  stemmer.stem("builds")
  assert(['build', 'builds', 'building'] == stemmer.unstem("build"))
