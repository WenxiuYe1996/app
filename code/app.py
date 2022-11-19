from bs4 import BeautifulSoup
import requests
import re
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import PyPDF2


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
def getUserInputTopic():
    input_topic = input("What topic do you want to search for? ")
    input_topic_split = input_topic.split()
    key_words=""
    for word in input_topic_split:
        key_words+=word
        key_words+="%20"
    #remove last three charater
    key_words = key_words[:-3]
    return key_words

def getArticleCount(url):

    page = requests.get(url)
    doc = BeautifulSoup(page.text, "html.parser")
    c = doc.find('script', id="serverApp-state").text
    #print(c)
    result_list = re.findall('&q;/(.+?)&q;', c)
    number_of_articles_found = int(re.findall('PUBLIC&q;,&q;doc_count&q;:(.+?)}]', c)[0])
    return number_of_articles_found
def getPDFNameFromGivenURL(url):
    page = requests.get(url)
    doc = BeautifulSoup(page.text, "html.parser")
    c = doc.find('script', id="serverApp-state").text
    result_list = re.findall('&q;/(.+?)&q;', c)
    res = []
    [res.append(x) for x in result_list if x not in res and 'txt' not in x and 'xlsx' not in x and 'pptx' not in x and 'docx' not in x and 'mp4' not in x]
    return res

def getDPFLink(result_list):
    pdf_link_list = []
    link_list = []
    for x in result_list:
        pdf_link_list.append(f"https://ntrs.nasa.gov/{x}")
        link = x.split("/", 1)[1]
        link = link.split("/downloads")[0]
        link_list.append(f"https://ntrs.nasa.gov/{link}")
    return pdf_link_list, link_list

def DownloadPDFandConvertItIntoText(result_list, pdf_link_list, key_words):
    text_list = []
    pdf_path = []
    i = 0
    for p in pdf_link_list:
        response = requests.get(p)
        loc = result_list[i].split("downloads/")[1]
        path = f"pdf/{key_words}{i}.pdf"
        print(f"path :{path}")
        open(path, "wb").write(response.content)
        text = getText(path)
        text_list.append(text)
        pdf_path.append(path)
        new_text = text.strip("\n")

        with open(f'txt/{key_words}{i}.txt', 'w') as f:
            f.write(new_text)
            f.close() 
        i+=1
    #return pdf_link_list, text_list, pdf_path

def getText(path):
    text = extractTextFromPDFs(path)
    print("text length:" )
    print(len(text))
    if (len(text) < 5):
        #print("<5")
        print("Scanned PDF")
        text = extractTextFromScannedPDFs(path)
    return text

def getTextFromNasaWeb():

    key_words = getUserInputTopic()
    url = f"https://ntrs.nasa.gov/search?q={key_words}"
    number_of_articles_found = getArticleCount(url)
    print(f"number_of_articles_found: {number_of_articles_found}")

    list = []
    counter = 0
    for i in range (0, number_of_articles_found, 100):
        link = url + f'&page=%7B"size":100,"from":{i}%7D'
        print(getArticleCount(link))
        result_list = getPDFNameFromGivenURL(link)
        list = list + result_list
        pdf_link_list, link_list =getDPFLink(result_list)
        DownloadPDFandConvertItIntoText(result_list, pdf_link_list, key_words)

getTextFromNasaWeb()

