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


# @app.get("/getGPTResponse/{userQuery}")
def callChatGPT(userQuery : str):
    print("hello gpt")    
    openai.api_key = os.getenv('OPENAPI_KEY')
    # Encode the input prompt
    input_ids = tokenizer.encode(userQuery, return_tensors='pt')

    # Generate a response
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)

    # Decode the generated text
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"GPT-2: {reply}")
    print(f"ChatGPT: {reply}")
    return reply


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8005)
