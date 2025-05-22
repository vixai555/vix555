#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

train = pd.read_csv("train.csv")
print(train.head())
print(train.label.value_counts())
test = pd.read_csv("test_tweets.csv")
print(test.head())
sample = pd.read_csv("sample_submission.csv")
print(sample.head())


# In[2]:


def refine(user_string):      
    text = user_string
    cleaned_user_string = re.sub(r"[^a-zA-Z\s]", "", text)
    return(cleaned_user_string) 


# In[3]:


train_tweet = train[['tweet', 'label']]
desc = train_tweet['tweet']
refined_d = []
for d in desc:
    refined = refine(d)
    refined_d.append(refined)
train_tweet['preprocessed'] = refined_d

test_tweet = test['tweet']
refined_d = []
for d in test_tweet:
    refined = refine(d)
    refined_d.append(refined)
test_tweet['preprocessed'] = refined_d


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(train_tweet["preprocessed"],train_tweet["label"], test_size = 0.2, random_state = 42)


# In[24]:


#from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer(stop_words='english')
## for transforming the 80% of the train data ##
X_train_counts = count_vect.fit_transform(x_train)
## for transforming the 20% of the train data which is being used for testing ##
x_test_counts = count_vect.transform(x_test)


# In[25]:


model = MultinomialNB()
model.fit(X_train_counts,y_train)
#model.fit(X_train_tfidf,y_train)


# In[26]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[27]:


#print(x_test_counts.shape)
#print(X_train_counts.shape)
#print(y_train.shape)

y_pred = model.predict(x_test_counts)
actualValue = y_test
predictedValue = y_pred
print("Confusion Matrix :")
cmt = confusion_matrix(actualValue, predictedValue)
tn, fp, fn, tp = confusion_matrix(actualValue, predictedValue).ravel()
print(cmt)
print("Accuracy :")
acc = accuracy_score(actualValue, predictedValue)
print(acc)
print("Misclassification Rate:")
mis = 1 - accuracy_score(actualValue, predictedValue)
print(mis)
print("Precision :")
ps = average_precision_score(actualValue, predictedValue)
print(ps)
print("Recall :")
re = recall_score(actualValue, predictedValue)
print(re)
print("F1 Score :")
f1 = f1_score(actualValue, predictedValue)
print(f1)
print("False Positives :")
print(fp)


# In[28]:


# Now trying to test the above using test tweets file

## for transforming the whole train data ##
train_counts = count_vect.fit_transform(train_tweet['preprocessed'])
## for transforming the test data ##
test_counts = count_vect.transform(test_tweet['preprocessed'])
## fitting the model on the transformed train data ##
model.fit(train_counts,train_tweet['label'])
## predicting the results ##
predictions = model.predict(test_counts)


# In[30]:


final = pd.DataFrame({'id':test['id'],'label':predictions})
final.to_csv('test_predictions.csv',index=False)


# In[ ]:




