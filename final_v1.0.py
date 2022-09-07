"""
Text Mining Final
November 2020
"""

'''
Libraries Import
'''
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

#Importing twitter access keys from a separate file in this project folder:
from twitter_credentials import * #Using keys as a variable

#Setting up Twitter API connection:
def twitter_creds():
    """
    Connects to Twitter API.
    For confidentiality purposes the access keys are in a separate file in the project folder.
    Credential file is saved as credentials.py.
    
    Not that I don't trust people but trying to keep my twitter credentials secret.
    More so practicing HOW to keep my twitter credentials secret.
    
    """

    #Authenticate credentials using keys:
    auth = tw.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET) #tells twitter I am valid user
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    #Connect with API and authenticate:
    api = tw.API(auth) #speaks to twitter
    return api

#Collecting Tweets
search_words = "san+diego"
date_since = "2020-11-1"
tweets = tw.Cursor(api.search,
              q=search_words,
              lang="en",
              since=date_since).items(100)

#for tweet in tweets:
#    print(tweet.text)
    
    
sd_tweets = [[tweet.text] for tweet in tweets]
sd_tweets
    
tweet_text = pd.DataFrame(data=sd_tweets, 
                          columns=['Tweets about San Diego'])
tweet_text

#Dataframe is tweet_text
#Cleaning the text in the data frame
#Create function to clean tweets
def cleanTxt(clean_tweets):
    clean_tweets = re.sub(r'@[A-Za-z0-9]+', '', clean_tweets) # r tells python that this is a raw string, + indicates one or more.
    clean_tweets = re.sub(r'#', '', clean_tweets)
    clean_tweets = re.sub(r'RT[\s]+', '', clean_tweets)
    clean_tweets = re.sub(r'https?:\/\/\S+', '', clean_tweets)
    
    return clean_tweets

#Apply function to data frame
tweet_text['Tweets about San Diego'] = tweet_text['Tweets about San Diego'].apply(cleanTxt)

#Changing the name of the df so that it's more easily recognized as a data frame. If that make sense...
df_clean = tweet_text
'''
Subjectivity & Polarity

'''
#Function subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#Function polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

#Create two new columns
df_clean['Subjectivity'] = df_clean['Tweets about San Diego'].apply(getSubjectivity)
df_clean['Polarity'] = df_clean['Tweets about San Diego'].apply(getPolarity)

#Word Cloud
#mask = np.array(Image.open('darth.png'))
#custom_mask = np.array(Image.open("brain1.png"))
stopwords=set(STOPWORDS)

allWords = ' '.join( [twts for twts in df_clean['Tweets about San Diego']])
wc =  WordCloud(background_color='white',
                width=500, 
                height=300, 
                random_state=21, 
                max_font_size=119,
#                mask=custom_mask,
                stopwords=stopwords
                
                )

wc.generate(allWords)

#image_colors = ImageColorGenerator(custom_mask)
#wc.recolor(color_func=image_colors)

plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.show()

'''

Function to separate tweets into negative, neutral, and positive

'''

def getgroup(score):
    
    if score < 0:
        return 'Negative'
    elif score==0:
        return 'Neutral'
    else:
        return'Positive'
        
df_clean['Sentiment'] = df_clean['Polarity'].apply(getgroup)


def basic_clean(text):
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]

words = basic_clean(''.join(str(df_clean['Tweets about San Diego'].tolist())))














'''
Loading City Services requests data from csv file.
'''
#Loading data
sd20 = pd.read_csv("sd20.csv")

#Cleaning data
#Don't delete the key variable 'service_request-id'
remove = ['service_request_parent_id', 'sap_notification_number', 'date_requested',
          'date_closed', 'zip_code', 'comm_plan','park_name', 'iamfloc', 'floc', 'zipcode', 'comm_plan_code' ]

sd20 = sd20[sd20.columns.difference(remove)]














