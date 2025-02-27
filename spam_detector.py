#import useful modules
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#import spam email csv file
df = pd.read_csv('spam.csv', delimiter='\t')

#Produce list of useful words for AI to read
def process_text(text):
    #Removes all punctuation from the text
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    #Returns a list of all individual words minus stopwords and punctuation
    clean_words = [word.lower() for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    return ' '.join(clean_words)

#Split array of tokens into training data
vectorizer = TfidfVectorizer(preprocessor=process_text, ngram_range=(1,2), max_df=0.9, min_df=5, stop_words='english')
messages_bow = vectorizer.fit_transform(df['Message'])
X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['Type'], test_size=0.2, random_state=0)

#Create a program to predict which emails are spam based on the training data we gave it
classifier = MultinomialNB().fit(X_train, y_train)

#Take User Input
email_subject = input("Enter the email's subject line: ")
email = [email_subject]

#Provide prediction based on input
arr = vectorizer.transform(email)
pred = classifier.predict(arr)
if pred[0] == "spam":
    print("Likely a Scam Email. Block and don't respond.")
else:
    print("Not a scam.")

#Prints Confusion Matrix, Accuracy, and Classification Report for Algorithim
y_pred = classifier.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#WIN BIG $$$$$ BY CALLING 1+013-859-3793