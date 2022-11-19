import urllib.request
import PyPDF2
import io
import pytesseract
from pdf2image import convert_from_path

URL = 'https://ntrs.nasa.gov/api/citations/20030067676/downloads/20030067676.pdf'
req = urllib.request.Request(URL, headers={'User-Agent' : "Magic Browser"})
remote_file = urllib.request.urlopen(req).read()
remote_file_bytes = io.BytesIO(remote_file)
pdfdoc = PyPDF2.PdfFileReader(remote_file_bytes)

text = ""

for i in range(pdfdoc.numPages):
    current_page = pdfdoc.getPage(i)
    print("===================")
    print("Content on page:" + str(i + 1))
    print("===================")
    print(current_page.extractText())
    print(len(current_page.extractText()))
    text += current_page.extractText()

print("------------------------------------------")
#code to extract text from scanned pdfs
URL = 'https://ntrs.nasa.gov/api/citations/20030067676/downloads/20030067676.pdf'
from PIL import Image
import requests



response = requests.get(URL)
img = Image.open(BytesIO(response.content))
pages = convert_from_path(img)
text = ""
for page in pages:
    text += pytesseract.image_to_string(page)

new_text = text.strip("\n")

print(new_text )

