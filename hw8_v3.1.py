#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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


'''Twitter Credentials'''
import twitter_credentials

'''Data Collection'''
auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET) #tells twitter I am valid user
auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_SECRET)

#Connect with API and authenticate:
api = tw.API(auth) #speaks to twitter

#San Diego related tweets
search_words = "San+Diego"
date_since = "2020-1-1"
retweet_filter='-filter:retweets'

tweets = tw.Cursor(api.search,
              q=search_words+retweet_filter,
              lang="en",
              tweet_mode='extended',
              since=date_since).items(200)
sd_tweets = [[tweet.full_text] for tweet in tweets] #full_text for extended mode

sd_tweets_df = pd.DataFrame(data=sd_tweets, columns=['SDTweets'])

#define clean tweet function
def clean_text(text):  
 text = text.lower()
 #remove text in square brackets
 text = re.sub(r'\[.*?\]', '', text)
 #remove punctuation   
 text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) 
 #remove words containing numbers
 text = re.sub(r'\w*\d\w*', '', text)
 return text

#clean tweets
sd_clean = pd.DataFrame(sd_tweets_df.SDTweets.apply(lambda x: clean_text(x)))

nlp = spacy.load("en_core_web_sm")

def lemmatizer(text):        
    twtz = []
    doc = nlp(text)
    for word in doc:
        twtz.append(word.lemma_)
    return " ".join(twtz)

sd_cleanR = pd.DataFrame(sd_clean.SDTweets.apply(lambda x: lemmatizer(x)))
sd_cleanR['SDTweets'] = sd_cleanR['SDTweets'].str.replace('-PRON-', '')

#word cloud w/mask
mask = np.array(Image.open('lion.png'))
custom_mask = np.array(Image.open('lion.png'))
stopwords=set(STOPWORDS)

allWords = ' '.join( [twts for twts in sd_cleanR['SDTweets']])
wc =  WordCloud(background_color='white',
                width=500, 
                height=300, 
                random_state=21, 
                max_font_size=119,
                mask=custom_mask,
                stopwords=stopwords
                
                )

wc.generate(allWords)

image_colors = ImageColorGenerator(custom_mask)
wc.recolor(color_func=image_colors)

plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.show()

#save masked word cloud to file
#wc.to_file("lion_cloud.png")

#bag of words
def top_words(corpus, n=None):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = top_words(sd_cleanR.SDTweets, 10)
unigram = pd.DataFrame(common_words, columns = ['unigram' , 'count'])

#topic modelling
vectorizer = CountVectorizer(
analyzer='word',       
min_df=3,#minimum word occurences parameter
stop_words='english',#remove stop words
lowercase=True,#words to lowercase
token_pattern='[a-zA-Z0-9]{3,}',#num characters > 3
max_features=5000,#maximum unique words
                          )
#document term matriz
dtm = vectorizer.fit_transform(sd_cleanR.SDTweets)
dtm

lda_model = LatentDirichletAllocation(
n_components=5,#change to manipulate topic number
learning_method='online',
random_state=36, #random state       
n_jobs = -1 #for performance use all available CPUs. Not fully sure what this means but it sounds like a good thing lol.
                                     )
lda_output = lda_model.fit_transform(dtm)

for i,topic in enumerate(lda_model.components_):
    print(f'Top 10 words for topic #{i}:')
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]]) #-10 signifies that we want 10 top words per topic
    print('\n')
    
topic_values = lda_model.transform(dtm)
sd_cleanR['Topic'] = topic_values.argmax(axis=1)

























