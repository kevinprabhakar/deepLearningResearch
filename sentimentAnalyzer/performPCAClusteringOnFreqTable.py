import pandas as pd
import csv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
import codecs

def savePCAResultToFile(pcaResult):
    pcaListFormat = pcaResult.tolist()
    json_file = "frequencyTables/pcaResultOnFreqTableIndividualWords.json"
    json.dump(pcaListFormat, codecs.open(json_file, 'w', encoding='utf-8'), sort_keys=True, indent=4)

def createClustersAndPCAResult():
    kMeansIndexes = kMeans.fit_predict(pcaResult)

    wordToCluster = {}

    for index in range(0,len(kMeansIndexes)):
        if kMeansIndexes[index] not in wordToCluster:
            wordToCluster[kMeansIndexes[index]] = [headers[index]]
        else:
            wordToCluster[kMeansIndexes[index]].append(headers[index])

    return wordToCluster

def generateClusterByFreqRanking(wordToCluster):
    with open("frequencyTables/cleanedWordToFreqMap.json", 'r') as f:
        wordFreqMap = json.load(f)

        correctOrderMap = {}

        for cluster in wordToCluster:
            sortedList = list(reversed(sorted(wordToCluster[cluster], key=lambda x: wordFreqMap[x])))
            correctOrderMap[str(cluster)] = sortedList

        with open("frequencyTables/freqTablePCAClusteringInOrder.json", 'w') as writeFile:
            json.dump(correctOrderMap, writeFile)

frequencyTable = pd.read_csv("frequencyTables/frequencyTableByTweetIndividualWords.csv")

headers = list(frequencyTable.columns.values)

freqTableTranspose = frequencyTable.as_matrix().T

pca = PCA(n_components=79)
kMeans = KMeans(n_clusters=20, random_state=0)

pcaResult = pca.fit_transform(freqTableTranspose)

savePCAResultToFile(pcaResult)

print pca.explained_variance_
print pcaResult.shape

clusters = createClustersAndPCAResult()
generateClusterByFreqRanking(clusters)

print clusters