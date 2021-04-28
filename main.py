import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_predict

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn import svm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

import numpy as np

import matplotlib.pyplot as plt

import re

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


def metricas(modelo, tweets, classes):
    modelo.fit(tweets, classes)
    result = cross_val_predict(modelo, tweets, classes, cv=10)

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
    return dataFrame['Sentiment'].map({'Positive': 0, 'Negative': 1})


def printResult(expected_values, predict_values):
    matrix = confusion_matrix(expected_values, predict_values)
    print("Matrix de confusão das classes",
          " ".join(sentiments), '\n', matrix, "\n")

    print(classification_report(expected_values,
                                predict_values, target_names=sentiments))

    print('Acurácia do modelo: {}'.format(
        accuracy_score(expected_values, predict_values)))


if __name__ == "__main__":
    data = getData('Corona_NLP_train.csv')

    naiveBayesModel = naiveBayesClassifier()
    svmModel = SVMClassifier()

    metricas(naiveBayesModel, data.OriginalTweet, data.Sentiment)
    metricas(svmModel, data.OriginalTweet, data.Sentiment)
