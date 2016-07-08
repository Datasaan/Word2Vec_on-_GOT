# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 00:12:44 2016

@author: sanjeet
"""

import pandas as pd
from bs4 import BeautifulSoup as bs
import re 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer as cv
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def preprev(rev):
    b=bs(rev).get_text()
    b=re.sub("[^a-zA-Z]", " ", b)
    b=b.lower().split()
    return b
def prepw2v(rev):
    sens=tokenizer.tokenize(rev.decode('utf8'))
    sentences = []
    for sent in sens:
        if len(sent) > 0:
            sentences.append(preprev(sent))
    return sentences
strg1=open('1.txt','r').read()
strg2=open('2.txt','r').read()
strg3=open('3.txt','r').read()
strg4=open('4.txt','r').read()
strg5=open('5.txt','r').read()
strg=strg1+' '+strg2+' '+strg3+' '+strg4+' '+strg5
sentences=prepw2v(strg)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
num_features = 3000    # Word vector dimensionality                      
min_word_count = 0   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words
from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
model.init_sims(replace=True)
model_name = "GOTcontext"
model.save(model_name)