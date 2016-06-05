'''A Calculator Implemented With A Top-Down, Recursive-Descent Parser'''
# Author: Erez Shinan, Dec 2012
import re, collections
from operator import add,sub,mul,div
import numpy as np


import nltk_helper

Token = collections.namedtuple('Token', ['name', 'value'])
RuleMatch = collections.namedtuple('RuleMatch', ['name', 'matched'])
token_map = {'+':'ADD', '-':'ADD', '*':'MUL', '/':'MUL', '(':'LPAR', ')':'RPAR'}
rule_map = {
    'add' : ['mul ADD add', 'mul'],
    'mul' : ['atom MUL mul', 'atom'],
    'atom': ['NUM', 'LPAR add RPAR', 'neg'],
    'neg' : ['ADD atom'],
}
fix_assoc_rules = 'add', 'mul'
bin_calc_map = {'*':mul, '/':div, '+':add, '-':sub}

class word2vec_calc():
    '''
    Expression parser for word2vec. 
    '''
    def __init__(self, word2vec_model,):
        '''
        word2vec_model: gensim model 
        '''
        self.model = word2vec_model
        self.calc_map = {
            'NUM' : float,
            'atom': lambda x: x[len(x)!=1],
            'neg' : lambda (op,num): (num,-num)[op=='-'],
            'mul' : self.calc_binary,
            'add' : self.calc_binary,
        }


    def calc_binary(self, x):
        while len(x) > 1:
            
            # replace object with vector of corresponding word
            if type(x[0]) is str:
                x[0] = self.model[nltk_helper.clean_nltk(x[0])]
            if type(x[2]) is str:
                x[2] = self.model[nltk_helper.clean_nltk(x[2])]

            x[:3] = [ bin_calc_map[x[1]](x[0], x[2])]
        return x[0]
    
    def match(self, rule_name, tokens):
        if tokens and rule_name == tokens[0].name:      # Match a token?
            return tokens[0], tokens[1:]
        for expansion in rule_map.get(rule_name, ()):   # Match a rule?
            remaining_tokens = tokens
            matched_subrules = []
            for subrule in expansion.split():
                matched, remaining_tokens = self.match(subrule, remaining_tokens)
                if not matched:
                    break   # no such luck. next expansion!
                matched_subrules.append(matched)
            else:
                return RuleMatch(rule_name, matched_subrules), remaining_tokens
        return None, None   # match not found
    

    def _recurse_tree(self, tree, func):
        return map(func, tree.matched) if tree.name in rule_map else tree[1]
    def flatten_right_associativity(self, tree):
        new = self._recurse_tree(tree, self.flatten_right_associativity)
        if tree.name in fix_assoc_rules and len(new)==3 and new[2].name==tree.name:
            new[-1:] = new[-1].matched
        return RuleMatch(tree.name, new)
    def evaluate(self, tree):
        solutions = self._recurse_tree(tree, self.evaluate)
        return self.calc_map.get(tree.name, lambda x:x)(solutions)
    def calc(self, expr):
        split_expr = re.findall('[\d.]+|[%s]' % ''.join(token_map), expr)
        tokens = [Token(token_map.get(x, 'NUM'), x) for x in split_expr]
        tree = self.match('add', tokens)[0]
        tree = self.flatten_right_associativity( tree )
        return self.evaluate(tree)


#        print( calc(raw_input('> ')) )