import os
from fastapi import FastAPI, Query
import uvicorn
import openai
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dotenv import load_dotenv

load_dotenv()


# Configure logging
logging.basicConfig(level=logging.DEBUG)
app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost",
    "http://localhost:8005/"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

@app.get("/")
def read_root():
    message = {"msg": "hello world"}
    logging.info("Hello")
    return JSONResponse(content={"message": "Hello"})
# Response(content=message, media_type='application/json')


@app.get("/prompt/query")
def read_query(prompt: str = Query(None)):
    print("userQuery" + str(prompt))
    responseFromLLM = callChatGPT(prompt)
    return JSONResponse(content={"message": responseFromLLM})


from urllib.request import urlopen
from bs4 import BeautifulSoup

url = "https://www.mayoclinic.org/diseases-conditions/flu/in-depth/flu-shots/art-20048000"
html = urlopen(url).read()
soup = BeautifulSoup(html, features="html.parser")

# kill all script and style elements
for script in soup(["script", "style"]):
    script.extract()    # rip it out

# get text
text = soup.get_text()

# break into lines and remove leading and trailing space on each
lines = (line.strip() for line in text.splitlines())
# break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drop blank lines
text = '\n'.join(chunk for chunk in chunks if chunk)

print(text)

import numpy as np
import nltk
import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq
nltk.download('punkt_tab')
s = text
print(s)
#class for preprocessing and creating word embedding
class Preprocessing:
    #constructor
    def __init__(self,txt):
        # Tokenization
        nltk.download('punkt')  #punkt is nltk tokenizer
        # breaking text to sentences
        tokens = nltk.sent_tokenize(txt)
        self.tokens = tokens
        self.tfidfvectoriser=TfidfVectorizer()

    # Data Cleaning
    # remove extra spaces
    # convert sentences to lower case
    # remove stopword
    def clean_sentence(self, sentence, stopwords=False):
        sentence = sentence.lower().strip()
        sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
        if stopwords:
            sentence = remove_stopwords(sentence)
        return sentence

    # store cleaned sentences to cleaned_sentences
    def get_cleaned_sentences(self,tokens, stopwords=False):
        cleaned_sentences = []
        for line in tokens:
            cleaned = self.clean_sentence(line, stopwords)
            cleaned_sentences.append(cleaned)
        return cleaned_sentences

    #do all the cleaning
    def cleanall(self):
        cleaned_sentences = self.get_cleaned_sentences(self.tokens, stopwords=True)
        cleaned_sentences_with_stopwords = self.get_cleaned_sentences(self.tokens, stopwords=False)
        # print(cleaned_sentences)
        # print(cleaned_sentences_with_stopwords)
        return [cleaned_sentences,cleaned_sentences_with_stopwords]

    # TF-IDF Vectorizer
    def TFIDF(self,cleaned_sentences):
        self.tfidfvectoriser.fit(cleaned_sentences)
        tfidf_vectors=self.tfidfvectoriser.transform(cleaned_sentences)
        return tfidf_vectors

    #tfidf for question
    def TFIDF_Q(self,question_to_be_cleaned):
        tfidf_vectors=self.tfidfvectoriser.transform([question_to_be_cleaned])
        return tfidf_vectors

    # main call function
    def doall(self):
        cleaned_sentences, cleaned_sentences_with_stopwords = self.cleanall()
        tfidf = self.TFIDF(cleaned_sentences)
        return [cleaned_sentences,cleaned_sentences_with_stopwords,tfidf]
    
class AnswerMe:
    #cosine similarity
    def Cosine(self, question_vector, sentence_vector):
        dot_product = np.dot(question_vector, sentence_vector.T)
        denominator = (np.linalg.norm(question_vector) * np.linalg.norm(sentence_vector))
        return dot_product/denominator

    #Euclidean distance
    def Euclidean(self, question_vector, sentence_vector):
        vec1 = question_vector.copy()
        vec2 = sentence_vector.copy()
        if len(vec1)<len(vec2): vec1,vec2 = vec2,vec1
        vec2 = np.resize(vec2,(vec1.shape[0],vec1.shape[1]))
        return np.linalg.norm(vec1-vec2)

    # main call function
    def answer(self, question_vector, sentence_vector, method):
        if method==1: return self.Euclidean(question_vector,sentence_vector)
        else: return self.Cosine(question_vector,sentence_vector)

def RetrieveAnswer(question_embedding, tfidf_vectors,method=1):
    similarity_heap = []
    if method==1: max_similarity = float('inf')
    else: max_similarity = -1
    index_similarity = -1

    for index, embedding in enumerate(tfidf_vectors):
        find_similarity = AnswerMe()
        similarity = find_similarity.answer((question_embedding).toarray(),(embedding).toarray() , method).mean()
        if method==1:
            heapq.heappush(similarity_heap,(similarity,index))
        else:
            heapq.heappush(similarity_heap,(-similarity,index))

    return similarity_heap

# @app.get("/getGPTResponse/{userQuery}")
def callChatGPT(userQuery : str):
    # print("hello gpt")    
    # openai.api_key = os.getenv('OPENAPI_KEY')
    # # Encode the input prompt
    # input_ids = tokenizer.encode(userQuery, return_tensors='pt')

    # # Generate a response
    # output = model.generate(input_ids, max_length=50, num_return_sequences=1)

    # # Decode the generated text
    # reply = tokenizer.decode(output[0], skip_special_tokens=True)
    
    user_question = "High risk groups"
    #define method
    method = 1
    preprocess = Preprocessing(s)
    cleaned_sentences,cleaned_sentences_with_stopwords,tfidf_vectors = preprocess.doall()
    question = preprocess.clean_sentence(user_question, stopwords=True)
    question_embedding = preprocess.TFIDF_Q(question)
    similarity_heap = RetrieveAnswer(question_embedding , tfidf_vectors ,method)
    print("Question: ", user_question)
    # number of relevant solutions you want here it will print 2
    number_of_sentences_to_print = 2
    while number_of_sentences_to_print>0 and len(similarity_heap)>0:
        x = similarity_heap.pop(0)
        replyFromModel = cleaned_sentences_with_stopwords[x[1]]
        print(replyFromModel)
        number_of_sentences_to_print-=1
        # print(f"GPT-2: {reply}")
        # print(f"ChatGPT: {reply}")
    return replyFromModel


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8005)