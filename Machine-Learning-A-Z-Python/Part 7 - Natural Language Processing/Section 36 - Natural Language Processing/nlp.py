# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as plt
import pandas as pd

dataset = pd.read_csv('.\\Part 7 - Natural Language Processing\\Section 36 - Natural Language Processing\\Restaurant_Reviews.tsv',sep='\t')

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
    
#4. Stemming - stem function
from nltk.stem.porter import PorterStemmer
def preprocess(review):
    review = dataset['Review'][0]
    
    #1. Keeping only char a-z
    import re
    review = re.sub('[^a-zA-Z]',' ',review)
    
    #2. Lower the case
    review = review.lower()
    
    review = review.split()
    
    #3. Remove the non significant word - if condition in for

    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english')) ]
    review = ' '.join(review)    
    return review

l = len(dataset)
corpus = []
for i in  range(1,l):
    review = preprocess(dataset['Review'][i])
    corpus.append(review)

#Bag of words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values()




