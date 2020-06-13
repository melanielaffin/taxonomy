import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re


with open('sample.pdf','rb') as pdf_file, open('sample.txt', 'w') as text_file:
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    number_of_pages = read_pdf.getNumPages()
    for page_number in range(number_of_pages):   # use xrange in Py2
        page = read_pdf.getPage(page_number)
        page_content = page.extractText()
        text_file.write(page_content)


with open('sample.txt', 'r', encoding="latin-1") as f:
    read_output = f.readlines()
read_output = '\n'.join(read_output).strip().replace('\n','')


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;.]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
SHORTWORD = re.compile(r'\W*\b\w{1,3}\b')

stop_words_lst = ['cell','spectral','data','double','bond','barrier','soluble', 'cells', 'performed using']

def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = str(text).lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    text = re.sub(r'\b[a-zA-Z]\b','',str(text)) # remove single letter words
    text = re.sub('\d+','',text) # remove digits
    text = SHORTWORD.sub('',text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    for w in stop_words_lst:
        pattern = r'\b'+w+r'\b'
        text = re.sub(pattern, '', str(text))

    return text

pdf_o = clean_text(read_output)
text_tokens = word_tokenize(pdf_o)

pdf_i=[pdf_o]

# Getting bigrams
vectorizer = CountVectorizer(ngram_range =(2, 3))
X1 = vectorizer.fit_transform(pdf_i)
features = (vectorizer.get_feature_names())

X2 = vectorizer.fit_transform(pdf_i)
scores = (X2.toarray())

sums = X2.sum(axis = 0)
data1 = []
for col, term in enumerate(features):
    data1.append( (term, sums[0, col] ))
ranking = pd.DataFrame(data1, columns = ['term', 'rank'])
words = (ranking.sort_values('rank', ascending = False))
print ("\n\nWords : \n", words.head(50))
