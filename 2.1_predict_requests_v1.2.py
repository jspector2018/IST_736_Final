
#Import Libraries
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from urllib.request import urlopen # instead of urllib2 like in Python 2.7
import json
import numpy as np
import nltk
import string
import re
from sklearn import naive_bayes
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import numpy as np

#multi-class classification
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

#Loading Get It Done San Diego Data
gid_csv = pd.read_csv('get_it_done_2020_ALL.csv') #static data of all requests in 2020 to date (11/25/20)
gid_json = "http://san-diego.spotreporters.com/open311/v2/requests.json" #live API connection (most recent 50 requests)

with urlopen("http://san-diego.spotreporters.com/open311/v2/requests.json") as response:
    source = response.read()
#print(source)
gid = json.loads(source)
print(json.dumps(gid, indent = 2)) #Used to better read the json data

#save json url to file
with open('gid.json', 'w') as f:
    json.dump(gid, f)

#Store json data in a data frame
df = pd.read_json('gid.json', orient = 'columns')

#Drop unwanted columns
df_cln = df.drop(columns=['service_code', 'status_notes', 'media_url'])

pd.value_counts(df_cln['service_name'])
'''
Illegal Dumping      31
Encampment           14
Parking Issue         3
Missed Collection     2
'''
#Filtering data with multiple criteria

gid_cln = gid_csv[(gid_csv['service_name'] == 'Missed Collection') | (gid_csv['service_name'] == 'Illegal Dumping') |
(gid_csv['service_name'] == 'Graffiti Removal') | (gid_csv['service_name'] == 'Encampment') | (gid_csv['service_name'] == '72 Hour Violation') |
(gid_csv['service_name'] == 'Pothole') | (gid_csv['service_name'] == 'Other') | (gid_csv['service_name'] == 'Shared Mobility Device') |
(gid_csv['service_name'] == 'Street Light Out') | (gid_csv['service_name'] == 'Parking Zone Violation') | (gid_csv['service_name'] == 'Sidewalk Repair Issue') |
(gid_csv['service_name'] == 'Traffic Sign - Maintain') | (gid_csv['service_name'] == 'Dead Animal') |
(gid_csv['service_name'] == 'Tree/Limb Fallen/Hanging')]


#Specify Relevant Columns to Keep
keep = ['service_name','public_description']
gid_clnR = gid_cln[keep]
len(gid_clnR)
#203702 requests

'''
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = LogisticRegression(multi_class='ovr')
# fit model
model.fit(X, y)
# make predictions
yhat = model.predict(X)
'''


'''
Multinomial Naive Bayes    
tfidVectorizder
'''
#Creating Training and Testing Datasets 70/30
mnb = gid_clnR
print(len(mnb))
#print(len(mnb2))
#58250
mnb_cln = mnb.dropna()
print(mnb_cln)
#print(mnb_cln2)
#[51350 rows x 2 columns]
mnb_cln.head()


stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words = stopset)

y = mnb_cln.service_name
x = vectorizer.fit_transform(mnb_cln.public_description)
  
print(y.shape)
print(x.shape)
#(51350,)
#(51350, 18314)
#There are 51350 observations and 18314 unique words

#set test train split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 36)

#train naive bayes classifier
clf = naive_bayes.MultinomialNB()
clf.fit(x_train, y_train)

#test model accuracy
roc_auc_score(y_test, clf.predict_proba(x_test)[:,1], multi_class = 'ovr', average='weighted')
#0.9822892491459847

#Test with user originated text
case311_array=np.array(["I have some guys living in a tent here on my property. This is nuts..."])
case311_array=np.array(["there is graffiti"])
case311_vector = vectorizer.transform(case311_array)

print(clf.predict(case311_vector))
#['Encampment']
#['Graffiti Removal']
'''
The result of this test is accurate. The intention of user input is that there
is an encampment on their property and they want the city to come take care of the situation.
'''
#Creating a similar program but now it requests input from the user
request_input = input("What your non-emergency issue / request? ")
#Test with user originated text
case311_array=np.array([request_input])
case311_vector = vectorizer.transform(case311_array)

print(clf.predict(case311_vector))

#Results HERE

'''
Test results of user inputs

What is the issue? there is a homeless person
['Encampment']

What is the issue? Someone dumped trash in the street
['Illegal Dumping']

'''

