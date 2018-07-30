from pymongo import MongoClient
import json
import string
import csv


client = MongoClient('localhost', 27017)
db = client.textcorpus
tweets = db.tweets

wordToFreqMap = json.load(open("wordFreqMap.json"))
cleanedWordToFreqMap = {}

def updateWithNewText(objectID, cleanedText):
    tweets.update({"_id":objectID},{"$set":{"cleanedTextBelow3000":cleanedText}},upsert=False)

def removeExtraWhiteSpace(tweetText):
    words = tweetText.decode("utf8").strip().split()
    reformed = ""
    for word in words:
        reformed += word + " "
    # return reformed.encode("utf8")
    printable = set(string.printable)
    return filter(lambda x: x in printable, reformed)

def addWordToMap(wordFreqMap, word):
    if word in wordFreqMap:
        wordFreqMap[word] += 1
    else:
        wordFreqMap[word] = 1

#Adds all words in tweet to Frequency Map
def insertTweetToMap(wordFreqMap, tweetText):
    words = tweetText.decode("utf8").strip().split()
    for word in words:
        addWordToMap(wordFreqMap, word)

for tweet in tweets.find():
    words = tweet["cleanedText"].split()
    tweetId = tweet["_id"]
    newTweet = [word for word in words if wordToFreqMap[word] > 57]
    reformed = ""
    for word in newTweet:
        reformed += word + " "

    updateWithNewText(tweetId, reformed)
    insertTweetToMap(cleanedWordToFreqMap, reformed)

print "Writing To JSON File cleanedWordToFreqMap.json"

with open('cleanedWordToFreqMap.json', 'w') as fp:
    json.dump(cleanedWordToFreqMap, fp)

print "Creating Frequency Table"

with open("frequencyTable.csv", "wb") as f:
    writer = csv.writer(f)
    headers = ["_id"]
    for key in cleanedWordToFreqMap.iterkeys():
        headers.append(key)
    writer.writerow(headers)

    for tweet in tweets.find():
        line = [tweet["_id"]]
        words = tweet["cleanedTextBelow3000"].split()
        for key in cleanedWordToFreqMap.iterkeys():
            if key in words:
                line.append(cleanedWordToFreqMap[key])
            else:
                line.append(0)
        writer.writerow(line)






