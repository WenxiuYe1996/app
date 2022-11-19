from bs4 import BeautifulSoup
import requests
import re

from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import numpy as np
import PyPDF2
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt
# NLTK Stop words
from nltk.corpus import stopwords
import string
import warnings
warnings.simplefilter('ignore')
from nltk.stem.wordnet import WordNetLemmatizer
from pprint import pprint
import gensim
import gensim.corpora as corpora

from lib2to3.pgen2.tokenize import tokenize
import re
import os
import numpy as np
import pandas as pd

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from nltk.stem.porter import *
# spacy for lemmatization
import spacy
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import re
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from pprint import pprint
warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
def sentences_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  



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


def extractTextFromScannedPDFs(fname):

    #code to extract text from scanned pdfs
    pages = convert_from_path(fname)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)

    new_text = text.strip("\n")
    return new_text


def extractTextFromPDFs(fname):
    #code to extract text from pdfs
    pdfFileObj = open(fname, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    text = ""

    for i in range(pdfReader.numPages):
        pageObj = pdfReader.getPage(i)
        text += pageObj.extractText()
    pdfFileObj.close()

    return text


def getText(path):

    text = extractTextFromPDFs(path)
    print("text length:" )
    print(len(text))
    if (len(text) < 5):
        print("<5")
        text = extractTextFromScannedPDFs(path)
    return text
def getTextFromNasaWeb(input_topic):

    #input_topic = input("What topic do you want to search for? ")
    input_topic_split = input_topic.split()
    key_words=""
    for word in input_topic_split:
        key_words+=word
        key_words+="%20"

    #remove last three charater
    key_words = key_words[:-3]
    print(key_words)

    #url for nasa website
    url = f"https://ntrs.nasa.gov/search?q={key_words}"
    #numb of pange 
    url += "&page=%7B%22size%22:100,%22from%22:0%7D"
    print(url)

    #use BeautifulSoup to get the html information
    page = requests.get(url)
    doc = BeautifulSoup(page.text, "html.parser")


    c = doc.find('script', id="serverApp-state").text
    #print(c)
    result_list = re.findall('&q;/(.+?)&q;', c)
    res = []
    #[res.append(x) for x in result_list if x not in res and 'txt' not in x and 'xlsx' not in x and 'pptx' not in x and 'docx' not in x]
    [res.append(x) for x in result_list if x not in res and 'txt' not in x and 'xlsx' not in x and 'pptx' not in x and 'docx' not in x and 'mp4' not in x]
  
    pdf_link_list = []
    link_list = []
    for x in res:
        pdf_link_list.append(f"https://ntrs.nasa.gov/{x}")
        link = x.split("/", 1)[1]
        link = link.split("/downloads")[0]
        link_list.append(f"https://ntrs.nasa.gov/{link}")

    print(f"length pdf link:{len(pdf_link_list)}")
    for v in pdf_link_list:
        print(v)
    
    text_list = []
    pdf_path = []
    #download the pdfs
    print(f"length link :{len(pdf_link_list)}")
    i = 0
    for p in pdf_link_list:
        print(f"i:{i}")
        response = requests.get(p)
        loc = res[i].split("downloads/")[1]
        print(loc)
        path = f"pdf/{loc}"
        open(path, "wb").write(response.content)
        print(path)
        text = getText(path)
        text_list.append(text)
        pdf_path.append(path)

        new_text = text.strip("\n")
        #print(new_text)


        with open(f'txt/{key_words}{i}.txt', 'w') as f:
            f.write(new_text)
            f.close() 
        i+=1
        # creating df object with columns specified    
        

    return pdf_path


def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out


def generateLDAmodel(doc_name, documents):
    lda_model_link = []
    for x in range(0, len(documents)):
        # Import Dataset
        #data = pd.read_csv('/Users/wenxiuye/Desktop/clean_code/txt/myfile0.txt')
        data = []
        
        data.append(documents[x])
        print(doc_name[x])
        #print(data)
        # Convert sentences into words
    #sentences = data.text.values.tolist()
        sentences = data
        #print(sentences)
        data_words = list(sentences_to_words(sentences))
        #print(data_words)
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=1, threshold=100) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        data_ready = data_words#process_words(data_words)
        # Create Dictionary
        id2word = corpora.Dictionary(data_ready)

        # Create Corpus: Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in data_ready]

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

        pprint(lda_model.print_topics())


        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
        pyLDAvis.save_html(vis, f'lda_html/{doc_name[x]}.html')
        lda_model_link.append(f'lda_html/{doc_name[x]}.html')

    return lda_model_link

def search(term, num_results=10, lang="en"):

    #link_list =  getTextFromNasaWeb(term)
    doc_name, documents = get_doc_list('txt')
    lda_model_link = generateLDAmodel(doc_name, documents)


    return lda_model_link
