import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

import numpy as np

import re

sentiments = ['Positive', 'Negative', 'Neutral']
classes = ["OriginalTweet", "Sentiment"]

def train(file_name):
    df = pd.read_csv(file_name)[classes] 

    df = filterSentiments(df, sentiments)
    
    print('Distribuição das classes')
    print(df.Sentiment.value_counts())

    #Codificar a base
    df.Sentiment = codeSet(df)

    # Normaliza os index do data frame
    #df = transformDataSet(df)
    
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    
    df['OriginalTweet'] = [preProcessing(i, stop_words) for i in df['OriginalTweet']]
  
    #Vetorizar
    tweet_tokenizer = TweetTokenizer()
    vectorizer = CountVectorizer(analyzer="word", tokenizer=tweet_tokenizer.tokenize)

    freq_tweets = vectorizer.fit_transform(df.OriginalTweet)
    type(freq_tweets)
    print(freq_tweets.shape)

    modelo = MultinomialNB()
    modelo.fit(freq_tweets, df['Sentiment'])

    df_test = pd.read_csv('Corona_NLP_test.csv')[classes]
    df_test = filterSentiments(df_test, sentiments)

    test_list = df_test['OriginalTweet'].tolist()
    freq_testes = vectorizer.transform(test_list)
    
    
    expected = codeSet(df_test).tolist()
    predicted = []

    for t, c in zip (test_list,modelo.predict(freq_testes)):
        # t representa o tweet e c a classificação de cada tweet.
        print(t + ", "+ str(c)) 
        predicted.append(c)
    
    printResult(expected, predicted)

def preProcessing(tweet, stop_words):
    tweet = removeStopWords(tweet, stop_words)
    tweet = cleanAttribute(tweet)

    return tweet

def removeStopWords(attribute, stop_words):
    words = [word for word in attribute.split() if not word in stop_words]
    
    return ' '.join(words)      

def cleanAttribute(attribute):
    attribute = re.sub(r"http\S+", "", attribute)
    attribute = re.sub(r"#\S+", "", attribute)
    attribute = re.sub(r"@\S+", "", attribute).lower().replace('.','').replace(';','').replace('-','').replace(':','').replace(')','')

    return attribute
    

def transformDataSet(df):
    data = []

    for row in df.itertuples():                   
        data.append([row[1], row[2]])

    return pd.DataFrame(np.array(data), columns=['OriginalTweet', 'Sentiment'])

def filterSentiments(dataFrame, sentiments):
    return dataFrame[dataFrame['Sentiment'].isin(sentiments)]

def codeSet(dataFrame):
    return dataFrame['Sentiment'].map({'Positive': 0, 'Negative': 1, 'Neutral': 2})

def printResult(expected_values, predict_values):
    matrix = confusion_matrix(expected_values, predict_values)
    print("Matrix de confusão das classes", " ".join(sentiments), '\n', matrix, "\n")
    
    metrics = classification_report(expected_values, predict_values, target_names=sentiments)
    print(metrics)


if __name__ == "__main__":
    train('Corona_NLP_train.csv')
    
    #refactor
    #train_data = makeTrainData('Corona_NLP_train.csv')
    #train(data)
    
    #test_data = makeTestData('')
    #test()
