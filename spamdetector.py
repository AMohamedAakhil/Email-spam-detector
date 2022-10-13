import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

df = pd.read_csv('emails.csv')
df.head(5)
df.drop_duplicates(inplace = True)

nltk.download('stopwords')

def process_text(text):
    
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english ')]
    
    return clean_words

messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['spam'], test_size = 0.20, random_state = 0)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

print(classifier.predict(X_train))
print(y_train.values)

pred = classifier.predict(X_train)
print(classification_report(y_train ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
print()
print('Accuracy: ', accuracy_score(y_train,pred))

print('Predicted value: ',classifier.predict(X_test))

print('Actual value: ',y_test.values)
pred = classifier.predict(X_test)
print(classification_report(y_test ,pred ))
print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))

