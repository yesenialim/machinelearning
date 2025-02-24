import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#0-non_spam, 1-spam
#can alter to put in your own excel files
spam_df = pd.read_csv("/Users/yesenia/Desktop/emails.csv")
#print(spam_df.head(10))
#print(spam_df.describe())
#print(spam_df.info())

#in the earlier dataset, the spam is signified with either 0 or 1 so 1 means spam and 0 means not spam
#print(spam_df.groupby('spam').describe())

#to investigate further on how people group their emial sby spam , we investigate their length to get a hang on whether it is a spam or not
spam_df['length'] = spam_df['text'].apply(len)
(spam_df.head())
#spam_df['length'].plot(bins=100, kind='hist')
#plt.show()
#u realise that the length is not a very good indicator as well bcus the length varies

spam_df.length.describe()
(spam_df[spam_df['length'] == 43952]['text'].iloc[0])

non_spam = spam_df[spam_df['spam']==0]
spam= spam_df[spam_df['spam']==1]
#print(non_spam)
#print(spam)

#find the percentage of spam & non_spam emails
##print( 'Non_Spam percentage =', (len(non_spam) / len(spam_df) )*100,"%")
sns.countplot(x='spam', data=spam_df, hue='spam', palette=['blue', 'red'])  # Correct
#plt.show()

#step3: CREATE TESTING AND TRAINING DATASET/DATA CLEANING
import string
#print(string.punctuation)

Test = 'Hello :) Mr. Future, I am so happy, to be- learning machine learning now!!'
Test_punc_removed = [char for char in Test if char not in string.punctuation] # this will show letter by letter because of for loop
Test_punc_removed_join = ''.join(Test_punc_removed)
test_fixed=' '.join(Test_punc_removed_join.split())
(test_fixed)


import nltk
nltk.download('stopwords')  # This must come after importing nltk

from nltk.corpus import stopwords
(stopwords.words('english'))
Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
#(Test_punc_removed_join_clean)

from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)

(vectorizer.get_feature_names_out())
(X.toarray())
#-------------------------------------------------------------
def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean
##creating a functionto include all the functions like join, removing punctuation

spam_df_clean = spam_df['text'].apply(message_cleaning)
(spam_df_clean)
(spam_df_clean[0]) #The code is used to print the value of the first entry in the 'text' column of the spam_df DataFrame.

#3training and testing
from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning)
spam_countvectorizer = vectorizer.fit_transform(spam_df['text'])
(vectorizer.get_feature_names_out())
(spam_countvectorizer.toarray())
(spam_countvectorizer.shape)

#4training the model
from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
label = spam_df['spam'].values
NB_classifier.fit(spam_countvectorizer, label)
testing_sample = ['Free money!!!', "Hi Kim, Please let me know if you need any further information. Thanks"]
testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)
test_predict

'''
testing_sample = ['Hello, I am Ryan, I would like to book a hotel in Bali by January 24th', 'money viagara!!!!!']
testing_sample = ['money viagara!!!!!', "Hello, I am Ryan, I would like to book a hotel in SF by January 24th"]
testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)
print(test_predict)
'''

#4: DIVIDE THE DATA INTO TRAINING AND TESTING PRIOR TO TRAINING
X = spam_countvectorizer
y = label
X.shape
y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

#5evaluate model
from sklearn.metrics import classification_report, confusion_matrix
#predicting training result
y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)

# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
plt.show()
print(classification_report(y_test, y_predict_test))




