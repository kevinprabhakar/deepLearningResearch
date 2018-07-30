import json

newMap = json.load(open("wordFreqMap.json"))

wordToFreqMap = {}

for item in newMap.iteritems():
    if int(item[1]) not in wordToFreqMap:
        wordToFreqMap[int(item[1])] = 1
    else:
        wordToFreqMap[int(item[1])] += 1

total = 0
for i in range(1,58):
    total += wordToFreqMap[i]

print len(newMap)
print len(newMap) - total
