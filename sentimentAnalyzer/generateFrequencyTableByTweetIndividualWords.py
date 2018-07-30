from pymongo import MongoClient
import csv


client = MongoClient('localhost', 27017)
db = client.textcorpus
tweets = db.tweets

with open("frequencyTables/frequencyTableWordTotalsByTweet.csv", 'r') as f:
    reader = csv.reader(f)

    for row in reader:
        headers = row[1:]
        break

print headers

myFile = open('frequencyTables/frequencyTableByTweetIndividualWords.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerow(headers)

    for tweet in tweets.find():
        tweetFrequencyPerWord = [0 for x in headers]
        tweetWords = tweet["cleanedTextBelow3000"].split()
        for word in tweetWords:
            tweetFrequencyPerWord[headers.index(word)] = tweetFrequencyPerWord[headers.index(word)] + 1
        writer.writerow(tweetFrequencyPerWord)

