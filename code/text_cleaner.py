import re
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


def getText(path):

    text = extractTextFromPDFs(path)
    print("text length:" )
    print(len(text))
    if (len(text) < 5):
        print("<5")
        text = extractTextFromScannedPDFs(path)
    return text

#data cleaning
def sentences_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        #convert sentence to word
    return sentences
    
text = getText(f'b.pdf')
print(text)

with open(f'b.txt', 'w') as f:
        f.write(text)
        f.close() 
