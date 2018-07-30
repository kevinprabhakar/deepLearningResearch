from pymongo import MongoClient
import json
import string
import csv, json
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from matplotlib import pyplot

kMeans = KMeans(n_clusters=20, random_state=0)
pca = PCA(n_components=5)


#Loads all words from database, creates Word2Vec Model, returns model
def getWordEmbeddingsModel():
    client = MongoClient('localhost', 27017)
    db = client.textcorpus
    tweets = db.tweets

    sentences = []

    for tweet in tweets.find():
        sentences.append(tweet["cleanedTextBelow3000"].split())

    model = Word2Vec(sentences, size=300, window=3, min_count=5, workers=8)

    model.save('model.bin')

    return model

#Does PCA Analysis on variable X, prints result, and returns Principal Components
def doPcaAnalysis(X):
    pcaResult = pca.fit_transform(X)

    print "Variance of Components:", pca.explained_variance_

    return pcaResult

#Loads pre-saved model, returns model and Embedding Matrix
def getWordEmbedding():
    newModel = Word2Vec.load('model.bin')

    X = newModel[newModel.wv.vocab]

    return newModel, X

#Creates Clusters from PCAResult. Returns Clusters and PCA Result
def createClustersAndPCAResult():
    newModel, X = getWordEmbedding()

    pcaResult = doPcaAnalysis(X)

    kMeansIndexes = kMeans.fit_predict(pcaResult)

    wordToCluster = {}

    for index in range(0,len(kMeansIndexes)):
        if kMeansIndexes[index] not in wordToCluster:
            wordToCluster[kMeansIndexes[index]] = [vocabList[index]]
        else:
            wordToCluster[kMeansIndexes[index]].append(vocabList[index])

    return wordToCluster, pcaResult

#Ranks all words in cluster by their respective frequency. Saves to file
def generateClusterByFreqRanking(wordToCluster):
    with open("cleanedWordToFreqMap.json", 'r') as f:
        wordFreqMap = json.load(f)

        correctOrderMap = {}

        for cluster in wordToCluster:
            sortedList = list(reversed(sorted(wordToCluster[cluster], key=lambda x: wordFreqMap[x])))
            correctOrderMap[str(cluster)] = sortedList

        with open("clusterWordsByFreqRanking.json", 'w') as writeFile:
            json.dump(correctOrderMap, writeFile)

model, _ = getWordEmbedding()
vocabList = model.wv.index2word
wordClusters, pcaResult = createClustersAndPCAResult()
generateClusterByFreqRanking(wordClusters)

print "K-Means Score:",kMeans.score(pcaResult)

def printDendogram(pcaResult, vocabList):
    Z = linkage(pcaResult, 'ward')

    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('words')
    plt.ylabel('distance')
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=12,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        labels=vocabList
    )
    plt.show()

