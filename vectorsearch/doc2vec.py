from __future__ import division  # py3 "true division"

import logging
import sys
import os
import heapq
from timeit import default_timer
from copy import deepcopy
from collections import defaultdict
import threading
import itertools
import gensim

from gensim.utils import keep_vocab_item

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import xrange
from types import GeneratorType

logger = logging.getLogger(__name__)

try:
    from gensim.models.word2vec_inner import train_batch_sg, train_batch_cbow
    from gensim.models.word2vec_inner import score_sentence_sg, score_sentence_cbow
    from gensim.models.word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1
    MAX_WORDS_IN_BATCH = 10000

from gensim.models.doc2vec import TaggedDocument



class Doc2Vec(gensim.models.Doc2Vec):

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self._stem_memory = defaultdict(set)

    def most_similar(self, words={}, topn=10, clip_start=0, clip_end=None):
        """
        Find the top-N most similar docvecs known from training. Positive docs contribute
        positively towards the similarity, negative docs negatively.
        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given docs. Docs may be specified as vectors, integer indexes
        of trained docvecs, or if the documents were originally presented with string tags,
        by the corresponding tags.
        The 'clip_start' and 'clip_end' allow limiting results to a particular contiguous
        range of the underlying doctag_syn0norm vectors. (This may be useful if the ordering
        there was chosen to be significant, such as more popular tag IDs in lower indexes.)
        """
        self.init_sims()
        clip_end = clip_end or len(self.doctag_syn0norm)

        # if isinstance(positive, string_types + integer_types) and not negative:
        #     # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
        #     positive = [positive]

        # add weights for each doc, if not already present; default to 1.0 for positive and -1.0 for negative docs
        # positive = [
        #     (doc, 1.0) if isinstance(doc, string_types + (ndarray,) + integer_types)
        #     else doc for doc in positive
        # ]
        # negative = [
        #     (doc, -1.0) if isinstance(doc, string_types + (ndarray,) + integer_types)
        #     else doc for doc in negative
        # ]

        # compute the weighted average of all docs
        all_docs, mean = set(), []
        for doc, weight in words.items():
            if isinstance(doc, ndarray):
                mean.append(weight * doc)
            elif doc in self.doctags or doc < self.count:
                mean.append(weight * self.doctag_syn0norm[self._int_index(doc)])
                all_docs.add(self._int_index(doc))
            else:
                raise KeyError("doc '%s' not in trained set" % doc)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        dists = dot(self.doctag_syn0norm[clip_start:clip_end], mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_docs), reverse=True)
        # ignore (don't return) docs from the input
        result = [(self.index_to_doctag(sim), float(dists[sim])) for sim in best if sim not in all_docs]
        return result[:topn]


class TaggedLineDocument(object):
    """Simple format: one document = one line = one TaggedDocument object.
    Words are expected to be already preprocessed and separated by whitespace,
    tags are constructed automatically from the document line number."""
    def __init__(self, source):
        """
        `source` can be either a string (filename) or a file object.
        Example::
            documents = TaggedLineDocument('myfile.txt')
        Or for compressed files::
            documents = TaggedLineDocument('compressed_text.txt.bz2')
            documents = TaggedLineDocument('compressed_text.txt.gz')
        """
        self.source = source

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for item_no, line in enumerate(self.source):
                yield TaggedDocument(utils.to_unicode(line).split(), ['SENT_%i'%item_no])
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), ['SENT_%i'%item_no])
