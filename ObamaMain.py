import pandas as pd
import numpy as np
import re
import sys
from sklearn import *
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from nltk.stem import WordNetLemmatizer 
#nltk.download('wordnet')
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#predefined function for cleaning the text
def clean_text(row):
    text=''
    text = str(row)
    #anchor tags and retweet symbols are removed
    cleanr = re.compile('(<.*?>|(^|\s)RT(\s|$))')
    cleantext = re.sub(cleanr, ' ', text)
    ##entire username with @ symbol is removed
    cleantext0 = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)',' ',cleantext)
    #entire https is removed
    cleantext1 = re.sub(r"http\S+", "", cleantext0)
    #punctuation marks are removed
    cleantext2 = re.sub(r'[^\w\s]','',cleantext1)
    #making its case insensitive
    cleantext3 = ' '.join([word.lower() for word in cleantext2.split()])
    #Removing digits
    cleantext4 = ' '.join([item for item in cleantext3.split() if not item.isdigit()])
    #Stemming
    cleantext5 = ' '.join([ps.stem(w) for w in cleantext4.split()])      
    return cleantext5

#Calculating tf-idf/Count frequency for each tweet
def vectorize_data_TFIDF(tweets):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(tweets)
    vector = vectorizer.transform(tweets)
    return vector
def vectorize_data_Count(tweets):
    vectorizer = CountVectorizer()
    vectorizer.fit(tweets)
    vector = vectorizer.transform(tweets)
    return vector

def vectorize_data_TFIDF_train(tweets):
    vector = vectorizer.fit_transform(tweets)
    return vector

def vectorize_data_TFIDF_test(tweets):
    vector = vectorizer.transform(tweets)
    return vector

train_path = r"C:/Users/eunic/OneDrive/Desktop/trainingObamaRomneytweets.xlsx"
test_path= r"C:/Users/eunic/Downloads/Obama_Romney_Test_dataset_NO_label/Obama_Romney_Test_dataset_NO_label/Obama_Test_dataset_NO_Label.csv"
#Output File Path
file = open("C:/Users/eunic/OneDrive/Desktop/testfile_obama.txt", "w+")


#train_path =sys.argv[1]
#test_path =sys.argv[2]

stop_words = set(stopwords.words('english')) 
ps = PorterStemmer() 
lemmatizer = WordNetLemmatizer() 
data_obama = pd.read_excel(train_path, sheetname = 'Obama')
data_obama_test = pd.read_csv(test_path, encoding="ISO-8859-1")
data_obama = data_obama[(data_obama['Class'].isin((1,-1,0)))]
data_obama['Cleaned_tweet'] = data_obama['Anootated tweet'].apply(clean_text)
data_obama_test['Cleaned_tweet'] = data_obama_test['Tweet_text'].apply(clean_text)
test_data_TweetID = data_obama_test['Tweet_ID']

stop_words = set(stopwords.words('english')) 
ps = PorterStemmer() 
lemmatizer = WordNetLemmatizer() 

"""
#models and K cross validation
model_LR = LogisticRegression(class_weight='balanced', multi_class='auto', solver = 'liblinear')
model_SVM = svm.SVC(kernel='linear', C=0.85, class_weight='balanced', probability=False, decision_function_shape='ovr')
model_RF = RandomForestClassifier(criterion='gini' , n_jobs = 10, n_estimators = 100, class_weight='balanced_subsample')
model_NB = model = naive_bayes.MultinomialNB()
model_MLP = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(5, 2),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
folds = KFold(n_splits = 10, shuffle = False)
model = [model_NB,model_SVM,model_RF,model_MLP,model_LR]
cv = KFold(n_splits=10, shuffle=True)


for model in model:
    if model == model_NB:
        print("Multinomial Naive Bayes")
        vector_obama_tweets = vectorize_data_Count(data_obama['Cleaned_tweet'])
    if model == model_SVM:
        print("Linear SVM Classifier")
        vector_obama_tweets = vectorize_data_TFIDF(data_obama['Cleaned_tweet'])
    if model == model_RF:
        print("RandomForest Classifier")
        vector_obama_tweets = vectorize_data_Count(data_obama['Cleaned_tweet'])
    if model == model_MLP:
        print("MLPClassifier")
        vector_obama_tweets = vectorize_data_Count(data_obama['Cleaned_tweet']) 
    if model == model_LR:
        print("Best Model - Logistic Regression")
        vector_obama_tweets = vectorize_data_TFIDF(data_obama['Cleaned_tweet']) 
    X = vector_obama_tweets
    Y = data_obama['Class'].tolist()
    accuracy = 0
    recall = [0,0,0]
    precision = [0,0,0]
    f_score = [0,0,0]
    for train_index, test_index in cv.split(X):
        x_train = X[train_index]
        x_test = X[test_index]
        y_train = []
        y_test= []
        for i in train_index:
            y_train.append(Y[i])
        for i in test_index:
            y_test.append(Y[i])
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        classes = [1,0,-1]
        accuracy = accuracy + metrics.accuracy_score(y_test, y_pred) # accuracy
        precision_all = metrics.precision_score(y_test, y_pred, average = None, labels=classes) # precision
        recall_all = metrics.recall_score(y_test, y_pred, average = None,labels=classes) # recall
        f_score_all= metrics.f1_score(y_test, y_pred, average = None,labels=classes) #f1score
        for i in [0,1,2]:
            precision[i]= precision[i]+precision_all[i]
            recall[i]= recall[i]+recall_all[i]
            f_score[i]= f_score[i]+f_score_all[i]
    for i in [0,1,2]:
        precision[i]= precision[i]/10
        recall[i]= recall[i]/10
        f_score[i]= f_score[i]/10    
    print("Obama:  Acurracy: ",accuracy/10)
    print("Precision_avg :", precision)
    print("Recall_avg", recall)
    print("F1-Score_avg", f_score, "\n") 
""" 
print("Predicting the class of the test data using Logistic Regression")
model = LogisticRegression(class_weight='balanced', multi_class='auto', solver = 'liblinear')
vectorizer = TfidfVectorizer()
vector_obama_tweets = vectorize_data_TFIDF_train(data_obama['Cleaned_tweet'])
vector_obama_tweetstest = vectorize_data_TFIDF_test(data_obama_test['Cleaned_tweet'])
X = vector_obama_tweets
Y = data_obama['Class'].tolist()
tweetID_list = test_data_TweetID.tolist()
model.fit(X, Y)
y_pred = model.predict(vector_obama_tweetstest)
print(y_pred)
for ide,p in zip(tweetID_list, y_pred):
    file.write(str(ide)+ str(";;")+str(p)+ "\n")
file.close()