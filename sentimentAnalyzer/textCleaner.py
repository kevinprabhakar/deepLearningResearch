from langdetect import *
import HTMLParser
from pymongo import MongoClient
import json
import string

client = MongoClient('localhost', 27017)
db = client.textcorpus
tweets = db.tweets
keepEnglishProbability = 0.9
html_parser = HTMLParser.HTMLParser()
apostrophesMap = json.load(open("apostrophes.json"))
stopwordMap = json.load(open("stopword.json"))
punctuation = "!#'()*+,-./:;<=>?@[\]^_`{|}~\"&$"
wordFreqMap = {}

#Removes Stopwords and returns text
def removeStopWords(tweetText):
    words = tweetText.decode("utf8").split()

    for index in range(0,len(words)):
        if words[index] in stopwordMap:
            words[index] = ""

    reformed = ""
    for word in words:
        reformed += word + " "

    return reformed.encode("utf8")

#Remove fin tags and hash tags from text and add to database
def finHashUserTagRemoval(objectId, tweetText):
    words = tweetText.decode("utf8").split()

    finTags = []
    hashTags = []
    users = []

    for index in range(0,len(words)):
        if words[index] == "$":
            if index+1 != len(words):
                finTags.append(words[index+1])
                words[index] = "<FINTAG>"
                words[index+1] = ""
            else:
                words[index] = ""
        elif words[index][0:1] == "$":
            try:
                float(words[index][1:])
            except ValueError:
                finTags.append(words[index][1:])
                words[index] = "<FINTAG>"
        elif words[index][0:1] == "#":
            hashTags.append(words[index][1:])
            words[index] = "<HSHTAG>"
        elif words[index][0:1] == "@":
            users.append(words[index][1:])
            words[index] = "<USRTAG>"


    reformed = " ".join(words)

    #todo: Add all hashtags and fintags to document
    tweets.update({"_id": objectId}, {"$set": {"finTags": finTags}}, upsert=False)
    tweets.update({"_id": objectId}, {"$set": {"hashTags": hashTags}}, upsert=False)
    tweets.update({"_id": objectId}, {"$set": {"userTags": users}}, upsert=False)

    return reformed.encode("utf8")

#Removes hyperlink and returns string
def removeHyperLinks(tweetText):
    text = tweetText.decode("utf8")

    try:
        httpIndex = text.index("http")
    except ValueError:
        return text.encode("utf8")

    return text[:httpIndex].encode("utf8")

def removePictures(tweetText):
    text = tweetText.decode("utf8")

    try:
        picIndex = text.index("pic.twitter")
    except ValueError:
        return text.encode("utf8")

    return text[:picIndex].encode("utf8")

#Returns Escaped Tweet without contractions
def contractionsRemoval(tweetText):
    words = tweetText.decode("utf8").split()

    reformed = [apostrophesMap[word] if word in apostrophesMap else word for word in words]

    reformed = " ".join(reformed)

    return reformed.encode("utf8")

#HTML Unescaping any characters and UTF8 Encode. Returns Escaped Tweet
def HTMLEscape(html_parser,tweetText):
    tweet = html_parser.unescape(tweetText)
    tweet = tweet.encode("utf8")
    return tweet

#For each document, add a blank field called cleanedtext
#Finished
def addBlankCleanedField(objectID):
    tweets.update({"_id":objectID},{"$set":{"cleanedText":""}},upsert=False)

#If Probability(tweet is English) < keepEnglishProbability, remove it from Database
#Finished
def isEnglishLanguage(objectID, tweetText):
    langProbabilities = detect_langs(tweetText)

    if "en" == langProbabilities[0].lang:
        if langProbabilities[0].prob > keepEnglishProbability:
            print "Document", objectID, "is English"
        else:
            print "Document", objectID, "is Not English"
            tweets.remove({"_id": objectID})
    else:
        print "Document", objectID, "is Not English"
        tweets.remove({"_id":objectID})

#Removes punctuation and returns string
#Only do this one after you remove hash/user/fin tags
def removePunctuation(tweetText):
    words = tweetText.decode("utf8")

    for p in punctuation:
        words = words.replace(p," ")

    return words.encode("utf8")

#Removes white spaces larger than 1
#Do this at the end
def removeExtraWhiteSpace(tweetText):
    words = tweetText.decode("utf8").strip().split()
    reformed = ""
    for word in words:
        reformed += word + " "
    # return reformed.encode("utf8")
    printable = set(string.printable)
    return filter(lambda x: x in printable, reformed)


#Update Document with new text
def updateWithNewText(objectID, cleanedText):
    tweets.update({"_id":objectID},{"$set":{"cleanedText":cleanedText}},upsert=False)

#Removes unecessary numbers
def removeNumbers(tweetText):
    words = tweetText.decode("utf8").strip().split()

    def isInteger(word):
        try:
            int(word)
        except ValueError:
            return False

        return True

    def isFloat(word):
        try:
            float(word)
        except ValueError:
            return False

        return True

    for index in range(0,len(words)):
        if (words[index][0] == "$"):
            words[index] = "<DLRAMT>"
            continue
        if (words[index][len(words[index])-1] == "%"):
            words[index] = "<PCTAMT>"
            continue
        if any(char.isdigit() for char in words[index]):
            if isInteger(words[index]):
                words[index] = "<INTVAL>"
            elif isFloat(words[index]):
                words[index] = "<FLTVAL>"
            else:
                words[index] = "<LTRNUM>"

    reformed = ""
    for word in words:
        reformed += word + " "

    return reformed.encode("utf8")

#Adds word to Frequency Map
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

if __name__ == "__main__":
    tweetCount = tweets.count()
    currTweet = 0

    #Processing and Removal of Non-English Tweets Already Done

    print "Processing", tweetCount, "tweets"
    for document in tweets.find():
        tweetText = document["text"]
        tweetId = document["_id"]
        htmlEscaped = HTMLEscape(html_parser, tweetText)
        lowerCase = htmlEscaped.lower()
        removedContractions = contractionsRemoval(lowerCase)
        removedHyperlinks = removeHyperLinks(removedContractions)
        removedPictures = removePictures(removedHyperlinks)
        removedTags = finHashUserTagRemoval(tweetId, removedPictures)
        removedNumbers = removeNumbers(removedTags)
        removedPunctuation = removePunctuation(removedNumbers)
        removedStopWords = removeStopWords(removedPunctuation)
        result = removeExtraWhiteSpace(removedStopWords)

        updateWithNewText(tweetId, result)

        insertTweetToMap(wordFreqMap, result)
        currTweet += 1

        if (currTweet % 1000 == 0):
            print tweetCount-currTweet, "tweets Remaining"

    print "Finished Processing Tweets"
    print "Writing To JSON File wordFreqMap.json"

    with open('wordFreqMap.json', 'w') as fp:
        json.dump(wordFreqMap, fp)