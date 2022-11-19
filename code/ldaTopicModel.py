import re
import os
import numpy as np
import pandas as pd
import nlp

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from nltk.stem.porter import *
# spacy for lemmatization
import spacy
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
import pprint

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're','in','for', 'and', 'of','the', 'is', 'edu', 'to', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])


# data cleaning
def sentences_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        #convert sentence to word
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  
# remove stopwords
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# get all the document
def get_doc_list(folder_name):
    doc_list = []
    doc_name = []
    file_list = [folder_name+'/'+name for name in os.listdir(folder_name) if name.endswith('txt')]
    for file in file_list:
        doc_name.append(file)
        st = open(file,'r').read()
        doc_list.append(st)
    print ('Found %s documents under the dir %s .....'%(len(file_list),folder_name))
    return doc_name, doc_list

# make bigrams
def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

# make trigrams
def make_trigrams(texts, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# lemmatized words into their root form
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def getLDAModel(textfile):
    data = []
    data.append(textfile)
    #print(data)
    sentences = data
    data_words = list(sentences_to_words(sentences))

    #bigram = gensim.models.Phrases(data_words, min_count=1, threshold=100) # higher threshold fewer phrases.
    #trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    #bigram_mod = gensim.models.phrases.Phraser(bigram)
    #trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # Form Bigrams
    #data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
    
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

   # texts = #data_lemmatized

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]

    # Build LDA model

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=1, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=10,
                                            passes=10,
                                            alpha='symmetric',
                                            iterations=100,
                                            per_word_topics=True)
    #vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
    #pyLDAvis.save_html(vis, f'lda_html/1.html')
    # Compute Perplexity
    #print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
    # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    #coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    #coherence_lda = coherence_model_lda.get_coherence()
    #print('\nCoherence Score: ', coherence_lda)

    return lda_model



def compareLDAModel(ldaModel1, ldaModel2):
    topic_list1 = ldaModel1.print_topics()
    topic_list2 = ldaModel2.print_topics()
    topic_word_list1 = re.findall('"\w+"', topic_list1[0][1])
    topic_word_list2 = re.findall('"\w+"', topic_list2[0][1])
    print(f'topic1: {topic_word_list1}')
    print(f'topic2: {topic_word_list2}')

    count = 0
    for x in topic_word_list1:
        for y in topic_word_list2:
            if (x == y):
                count +=1

    return count / 10

def getTopicKeywordsWithItsWeight(ldaModel):
    topic_list1 = ldaModel.print_topics()
    list = topic_list1[0][1]
    list = list.replace(" ", "")
    list = list.split('+')
    keywordWithItsWeightList = []
    for x in list:
        print(x)
        temp = x.split('*')
        temp[1] = re.sub("\"", "", temp[1])
        temp[1] = re.sub("\"", "", temp[1])
        temp[1].replace(" ", "")
        print(temp)
        keywordWithItsWeightList.append(temp)
    res = dict()
    for sub in keywordWithItsWeightList:
        res[sub[1]] = sub[0]

    return res
def getTopicKeywordsWithItsWeight1(ldaModel):
    topic_list1 = ldaModel.show_topic(0, topn=20)
    print(topic_list1)
    res = dict()
    for x in topic_list1:
        res[x[0]] = x[1]

    return res


doc_name, documents = get_doc_list('txt2')
lda_model_list = []

document_list = documents

n = 0
for x in documents:
    print(doc_name[n])
    if (len(x) > 100):
        lda_model = getLDAModel(x)
        print(lda_model.show_topic(0, topn=20))
        print()
        print()
        lda_model_list.append(lda_model)
    n+=1

keyWordWeightList = dict()
key = []
key.append("climate")
key.append("change")

for y in range(0, len(lda_model_list)):
    #topic_list.append(y.print_topics())
    keywordWithItsWeightList = getTopicKeywordsWithItsWeight1(lda_model_list[y])
    print(keywordWithItsWeightList)
    weight = 0
    for x in key:
        result = keywordWithItsWeightList.get(x)
        if result == None:
            result = 0
        weight += result

    keyWordWeightList[doc_name[y]] = weight

for x in keyWordWeightList:
    print(f"{x},{keyWordWeightList.get(x)}")


dict(sorted(keyWordWeightList.items(), key=lambda item: item[1]))

for x in keyWordWeightList:
    print(f"{x},{keyWordWeightList.get(x)}")

sorted_people = sorted(keyWordWeightList.items(), key=lambda item: item[1])

#a = sorted(keyWordWeightList, key=lambda x: x[1], reverse=True)
print("----------------------------------") 
#for p in a:
    #print(f"{p}, {keyWordWeightList.get(p)}")

#print(len(document_list))
#print(len(topic_list))
#import pandas as pd
#data = pd.DataFrame(document_list, columns=['text'])
#data['topic'] = topic_list
#data.to_csv('data.csv')

#for x in lda_model_list:
#    for y in lda_model_list:
##        if(x != y):
#            print(compareLDAModel(x,y))

#for x in lda_model_list:
#    for y in lda_model_list:
#        if(x != y):
#            mdiff, annotation = x.diff(y, distance='jaccard', num_words=50)
#            print(mdiff)
#            print(annotation)