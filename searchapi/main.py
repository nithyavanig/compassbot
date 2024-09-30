import os
from fastapi import FastAPI, Query
import uvicorn
import openai
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import constants
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
    return JSONResponse(content={"message": responseFromLLM.message,"url": responseFromLLM.urls})


import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import io
import base64
from PIL import Image
from unidecode import unidecode
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Vigne\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def fetch_url_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text_content = soup.get_text()
    # Extract images and apply OCR
    images = soup.find_all('img')
    image_texts = []
    for img in images:
        img_url = img.get('src')
        if img_url:
            if img_url.startswith('data:image'): # Handle base64 encoded images
                img_data = img_url.split(',')[1]
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
                img_text = pytesseract.image_to_string(img)
                # img_text = unidecode(img_text)
            else:
                if not img_url.startswith('http'):
                    img_url = urljoin(url, img_url)  # Use urljoin to create absolute URL
                img_text = ocr_image(img_url)
            image_texts.append(img_text)

    return text_content + " ".join(image_texts)

def ocr_image(img_url):
    # Download image
    img_response = requests.get(img_url, stream=True)
    img = Image.open(img_response.raw)

    # Use Tesseract to extract text
    text = pytesseract.image_to_string(img)
    return text
def jinai(url):
    response=requests.get("https://r.jina.ai/"+url)
    return response.text

def fetch_all_urls(urls):
        contents = {}
        for url in urls:
            contents[url] = fetch_url_content(url)
        return contents

from sentence_transformers import SentenceTransformer, util
import torch

def generate_embeddings(contents):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = {}
    chunk_size = 300  # Larger chunk size for more context
    overlap_size = 100  # Overlap to preserve context across chunks

    for url, content in contents.items():
        paragraphs = []
        for i in range(0, len(content), chunk_size - overlap_size):
            chunk = content[i:i+chunk_size]
            paragraphs.append(chunk)

        embeddings[url] = (paragraphs, torch.tensor(model.encode(paragraphs)))
    return embeddings

def search_query(embeddings, query, model):
    query_embedding = model.encode([query])

    best_url = None
    best_snippets = []
    best_similarity = -1

    for url, (paragraphs, paragraph_embeddings) in embeddings.items():
        for paragraph, paragraph_embedding in zip(paragraphs, paragraph_embeddings):
            similarity = util.pytorch_cos_sim(query_embedding, paragraph_embedding.unsqueeze(0))[0][0].item()
            best_snippets.append((url, paragraph, similarity))

    best_snippets = sorted(best_snippets, key=lambda x: x[2], reverse=True)
    return best_snippets[:3]


from transformers import RobertaTokenizer, RobertaForQuestionAnswering 
def extract_answer(snippets, query):
    tokenizer = RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2')
    model = RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')

    best_answer = None
    highest_confidence = -float('inf')

    for url, snippet, _ in snippets:
        inputs = tokenizer.encode_plus(query, snippet, return_tensors='pt')
        input_ids = inputs['input_ids'].tolist()[0]

        outputs = model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1

        confidence = outputs.start_logits[0][answer_start].item() + outputs.end_logits[0][answer_end - 1].item()

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        if confidence > highest_confidence and answer.strip() != "" and "[CLS]" not in answer and "[SEP]" not in answer:
            best_answer = answer.strip()
            highest_confidence = confidence

    return best_answer

def answer_question(urls, query):
        contents = fetch_all_urls(urls)
        embeddings = generate_embeddings(contents)
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        best_snippets = search_query(embeddings, query, model)
        best_answer = extract_answer(best_snippets, query)

        # If no valid answer was found, return a fallback snippet
        if not best_answer:
            best_answer = best_snippets[0][1]
        return best_snippets[0][0], best_answer


    
# @app.get("/getGPTResponse/{userQuery}")
def callChatGPT(userQuery : str):
    print("hello gpt")    
    openai.api_key = os.getenv('OPENAPI_KEY')
    # Encode the input prompt
    input_ids = tokenizer.encode(userQuery, return_tensors='pt')

    # Generate a response
    # output = model.generate(input_ids, max_length=50, num_return_sequences=1)

    # # Decode the generated text
    # reply = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(f"GPT-2: {reply}")
    # print(f"ChatGPT: {reply}")
    
    samplequery = "What is the price of  cake ?"
    best_url, content = answer_question(constants.urls, userQuery)
    
    # return reply
    return {"message": content ,"urls": best_url}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8005)
