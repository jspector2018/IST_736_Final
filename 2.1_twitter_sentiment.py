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
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

#Import Libraries
from nltk.corpus import stopwords

#visuals
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.5)
#sentiment
from textblob import TextBlob

#%matplotlib inline #use this in notebook for visual
import seaborn as sns
import nltk
import unicodedata


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
sd_clean['Subjectivity'] = sd_clean['SDTweets'].apply(getSubjectivity)
sd_clean['Polarity'] = sd_clean['SDTweets'].apply(getPolarity)

'''
Function to separate tweets into negative, neutral, and positive

'''

def getSentiment(score):
    
    
    if (score == 0):
         return('Neutral')
    elif (score > 0 and score <= 0.3):
         return('Weakly Positive')
    elif (score > 0.3 and score <= 0.6):
         return('Positive')
    elif (score > 0.6 and score <= 1):
         return('Strongly Positive')
    elif (score > -0.3 and score <= 0):
        return('Weakly Negative')
    elif (score > -0.6 and score <= -0.3):
         return('Negative')
    elif (score > -1 and score <= -0.6):
        return('Strongly Negative')

        
sd_clean['Sentiment'] = sd_clean['Polarity'].apply(getSentiment)

#San Diego Tweet Review Sentiment Scatter Plot
plt.figure(figsize=(8,6))
for i in range(0, sd_clean.shape[0]):
    plt.scatter(sd_clean['Polarity'][i], sd_clean['Subjectivity'][i], color='purple')
    
plt.title('San Diego Sentiment On Twitter')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()


def basic_clean(text):
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]

words = basic_clean(''.join(str(sd_clean['SDTweets'].tolist())))

#Top 50 most frequent words
words[:50]

#Bigrams
(pd.Series(nltk.ngrams(words, 2)).value_counts())[:10] #Specifying 2 for bigrams

bigrams_series = (pd.Series(nltk.ngrams(words, 2)).value_counts())[:50]
bigrams_series.sort_values().plot.barh(color='red', width=.9, figsize=(12, 8)) ##red for republican party
plt.title('Most Frequently Occuring San Diego Bigrams on Twitter')
plt.ylabel('Bigram')
plt.xlabel('Number of Occurances')
#Print all positive tweets

j=1
posDF = sd_clean.sort_values(by=['Polarity'])
for i in range(0, posDF.shape[0]):
  if(posDF['Sentiment'][i] == 'Positive'):
    print(str(j) + ') '+posDF['SDTweets'][i])
    print()
    j = j+1

