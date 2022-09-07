#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:34:52 2020

@author: jspector
"""

#data load
from tweepy import OAuthHandler
import tweepy as tw

#data cleaning
import pandas as pd
import re
import numpy as np
import emoji
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import spacy

#Import Libraries
from sklearn import naive_bayes
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer

#analysis
import numpy as np
import pandas as pd
from sklearn import svm
import gensim
from gensim import corpora
#visuals
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.5)
#sentiment
from textblob import TextBlob

#results
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#%matplotlib inline #use this in notebook for visual
import seaborn as sns
import matplotlib as mpl
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

#topic modelling
from sklearn.decomposition import LatentDirichletAllocation

import datetime as dt
import pandas as pd
import tweepy as tw
import re
import json
import csv
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from PIL import Image
import os
from os import path
#!pip install colormap
#!pip install easydev
from colormap import Color, Colormap
from tweepy import OAuthHandler
from sklearn.feature_extraction.text import CountVectorizer
#!pip install scikit-learn
#!pip install scipy
#!pip install sklearn
from sklearn.metrics.pairwise import linear_kernel
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import unicodedata


#data load
from tweepy import OAuthHandler
import tweepy as tw

#data cleaning
import pandas as pd
import re
import numpy as np
import emoji
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import spacy

#Import Libraries
from sklearn import naive_bayes
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer

#analysis
import numpy as np
import pandas as pd
from sklearn import svm
import gensim
from gensim import corpora
#visuals
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.5)
#sentiment
from textblob import TextBlob

#results
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#%matplotlib inline #use this in notebook for visual
import seaborn as sns
import matplotlib as mpl
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

#topic modelling
from sklearn.decomposition import LatentDirichletAllocation