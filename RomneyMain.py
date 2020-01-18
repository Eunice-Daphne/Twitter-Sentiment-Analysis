import pandas as pd
import numpy as np
import re
import sys
from sklearn import *
import nltk
#from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import *
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from nltk.stem import WordNetLemmatizer 

#stop_words = set(stopwords.words('english')) 
ps = PorterStemmer() 
lemmatizer = WordNetLemmatizer()  
lmtzr = WordNetLemmatizer()
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

#predefined function for cleaning the text
def clean_text(row):
    tweet = str(row)
    #Removal of HTTP, RT tags, digits, anchor tags
    cleanr = re.compile('(</?[a-zA-Z]+>|https?:\/\/[^\s]*|(^|\s)RT(\s|$)|@[^\s]+|\d+)')
    cleantext = re.sub(cleanr, ' ', tweet)
    #Removal of usernames
    cleantext = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)',' ',cleantext)
    #Removal of non-letter characters
    cleantext = re.sub('[^\sa-zA-Z]+','',cleantext)
    #Removal of whitespaces
    cleantext = re.sub('\s+',' ',cleantext) 
    #Removal of punctuations
    cleantext = re.sub(r'[^\w\s]','',cleantext)
    #Convert to lowercase
    lower_case = cleantext.lower() 
    #Stemming
    stemmedTweet = [ps.stem(word) for word in lower_case.split(" ")]
    stemmedTweet = " ".join(stemmedTweet)
    tweet = str(stemmedTweet)
    #Lemmatization
    tweet = lmtzr.lemmatize(tweet) 
    return tweet

#Calculating tf-idf for each tweet
def vectorize_data(tweets):
    vector = vectorizer.fit_transform(tweets)
    return vector
def vectorize_data_test(tweets):
    vector = vectorizer.transform(tweets)
    return vector

#Output File Path
file = open("C:/Users/eunic/OneDrive/Desktop/dmtmtest_results.txt", 'w+')

#Loading dataset
data_path = r"C:/Users/eunic/OneDrive/Desktop/trainingObamaRomneytweets.xlsx"
data_romney = pd.read_excel(data_path, sheetname = 'Romney')

data_path_test = r"C:/Users/eunic/Downloads/Obama_Romney_Test_dataset_NO_label/Obama_Romney_Test_dataset_NO_label/Romney_Test_dataset_NO_Label.csv"
data_romney_test = pd.read_csv(data_path_test, encoding="ISO-8859-1")

# Removing rows with classes other than 1,-1,0
data_romney = data_romney[(data_romney['Class'].isin((1,-1,0)))]
test_data_TweetID = data_romney_test['Tweet_ID']

#Extracting only the class and annotation from given file
new_data_romney= data_romney[['Class','Anootated tweet']]
new_data_romney_test = data_romney_test['Tweet_text']

#Creating a column with pre processed tweets
new_data_romney['Cleaned_tweet'] = new_data_romney['Anootated tweet'].apply(clean_text)
new_data_romney_test['Cleaned_tweet'] = new_data_romney_test.apply(clean_text)

# converting the tweets and class to list form
cleaned_romney_tweets = new_data_romney['Cleaned_tweet'].tolist()
romney_class = new_data_romney['Class'].tolist()
cleaned_romney_tweets_test = new_data_romney_test['Cleaned_tweet'].tolist()
tweetID_list = test_data_TweetID.tolist()
#print(cleaned_romney_tweets_test[:10])

#Vectorizing the text data
vector_romney_tweets = vectorize_data(cleaned_romney_tweets)
vector_romney_tweets_test = vectorize_data_test(cleaned_romney_tweets_test)

#Logistic Regression Model
print("Logistic Regression")
model1 = linear_model.LogisticRegression(class_weight='balanced')

predicts = model1.fit(vector_romney_tweets, romney_class).predict(vector_romney_tweets_test)

for ide, p in zip(tweetID_list,predicts):
    file.write(str(ide)+";;"+str(p)+"\n")
file.close()