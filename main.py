import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_predict

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn import svm
from sklearn import neural_network

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

from textblob import TextBlob

import numpy as np

import matplotlib.pyplot as plt

import re

import time

sentiments = ['Positive', 'Negative']
classes = ["OriginalTweet", "Sentiment"]


def getData(file_name):
    df = pd.read_csv(file_name)[classes]
    # plotDf(df)
    df = filterSentiments(df, sentiments)

    print('Distribuição das classes')
    print(df.Sentiment.value_counts())

    # Codificar a base
    df.Sentiment = codeSet(df)

    # Normaliza os index do data frame
    #df = transformDataSet(df)

    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    df['OriginalTweet'] = [preProcessing(
        i, stop_words) for i in df['OriginalTweet']]

    return df


def naiveBayesClassifier():
    return Pipeline([
        ('counts', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])


def SVMClassifier():
    return Pipeline([
        ('counts', CountVectorizer()),
        ('classifier', svm.LinearSVC(random_state=9))
    ])


def neuralNetwork():
    return Pipeline([
        ('counts', CountVectorizer()),
        ('classifier', neural_network.MLPRegressor(random_state=1, 
            learning_rate_init=0.049, activation='logistic',hidden_layer_sizes=3))
    ])


def metricas(modelo, tweets, classes):
    start_time = time.time()

    modelo.fit(tweets, classes)
    result = cross_val_predict(modelo, tweets, classes, cv=10)

    end_time = time.time()
    print('Time (miliseconds): ', (end_time - start_time)*1000)

    printResult(classes, result)


def plotDf(df):
    values = df.Sentiment.value_counts()
    data = {'Positive': values['Positive'], 'Negative': values['Negative'], 'Neutral': values['Neutral'],
            'Extremely Positive': values['Extremely Positive'], 'Extremely Negative': values['Extremely Negative']}

    courses = list(data.keys())
    values = list(data.values())

    fig = plt.figure(figsize=(10, 5))

    plt.bar(courses, values, color='maroon',
            width=0.4)

    plt.xlabel("Classe (Sentimentos)")
    plt.title("Distribuição das classes")
    plt.show()


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
    attribute = re.sub(r"@\S+", "", attribute).lower().replace('.',
                                                               '').replace(';', '').replace('-', '').replace(':', '').replace(')', '')

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
    print("Matrix de confusão das classes",
          " ".join(sentiments), '\n', matrix, "\n")

    print(classification_report(expected_values,
                                predict_values, target_names=sentiments))

    print('Acurácia do modelo: {}'.format(
        accuracy_score(expected_values, predict_values)))

def tweet_analysis(tweets):
    predict_list = []
    subjectivities = []
    polarities = []

    for tweet in tweets.OriginalTweet:
        phrase = TextBlob(tweet)

        if phrase.sentiment.polarity != 0.0 and phrase.sentiment.subjectivity != 0.0:
            polarities.append(phrase.sentiment.polarity)
            subjectivities.append(phrase.sentiment.subjectivity)

        #print('Tweet: ' + tweet)
        #print('Polarity: ' + str(phrase.sentiment.polarity) + ' \ Subjectivity: ' + str(phrase.sentiment.subjectivity))
        #print('.....................')
        
        predict_list.append(get_result(phrase.sentiment.polarity))

    printResult(tweets.Sentiment.tolist(), predict_list)

    return {'polarity':polarities, 'subjectivity':subjectivities}

def get_result(polarity):
    if polarity >= 0.0:
        return 1
    elif polarity < 0.0:
        return 0
    elif polarity == 0.0:
        return 2
    
if __name__ == "__main__":
    data = getData('Corona_NLP_full.csv')

    #tweet_analysis(data)
    naiveBayesModel = naiveBayesClassifier()
    svmModel = SVMClassifier()
    neural = neuralNetwork()

    metricas(naiveBayesModel, data.OriginalTweet, data.Sentiment)
    metricas(svmModel, data.OriginalTweet, data.Sentiment)
    metricas(neural, data.OriginalTweet, data.Sentiment)
