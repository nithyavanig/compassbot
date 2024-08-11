import os
from fastapi import FastAPI
import uvicorn
import openai
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys
from dotenv import load_dotenv 
print(sys.path)


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

@app.get("/")
def read_root():
    message = {"msg": "hello world"}
    logging.info("Hello")
    return JSONResponse(content={"message": "Hello"})
# Response(content=message, media_type='application/json')


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.get("/getGPTResponse/{userQuery}")
def callChatGPT(userQuery : str):
    print("hello gpt")    
    openai.api_key = os.getenv('OPENAPI_KEY')
    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[
        {
            "role": "user",
            "content": userQuery,
        }
    ]
    )
    reply = chat.choices[0].message.content
    print(f"ChatGPT: {reply}")
    return reply


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8005)
