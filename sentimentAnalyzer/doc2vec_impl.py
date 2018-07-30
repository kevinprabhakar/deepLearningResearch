from gensim.models.doc2vec import Doc2Vec
from pymongo import MongoClient
import gensim
import os
import collections
import smart_open
import random

#Yield function to generate corpus from files
def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

print "Reading Corpus"
train_corpus = list(read_corpus('/Users/kprabhakar/Python Code/deepPortfolioTheory/sentimentAnalyzer/all-the-news/articles3.csv'))
# test_corpus = list(read_corpus('/Users/kprabhakar/Python Code/deepPortfolioTheory/sentimentAnalyzer/all-the-news/articles1.csv', tokens_only=True))
#

doc2VecModel = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=1, workers=4)
print "Building Vocab"

doc2VecModel.build_vocab(train_corpus[1:])

print "Beginning Training"
doc2VecModel.train(train_corpus, total_examples=doc2VecModel.corpus_count, epochs=doc2VecModel.epochs)

print "Inferring Vector"
print doc2VecModel.infer_vector(['trump','russia','pence'])

#Assessing the model
ranks = []
second_ranks = []
for cds in range(len(train_corpus)):
    doc_id = cds+1
    inferred_vector = doc2VecModel.infer_vector(train_corpus[doc_id].words)
    sims = doc2VecModel.docvecs.most_similar([inferred_vector], topn=len(doc2VecModel.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])

    collections.Counter(ranks)  # Results vary due to random seeding and very small corpus

    print 'Document ({}): {}\n'.format(doc_id, ' '.join(train_corpus[doc_id].words))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % doc2VecModel)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: %s\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
