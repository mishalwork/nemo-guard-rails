from fastapi import FastAPI
import asyncio
from nemoguardrails import LLMRails, RailsConfig
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

class ChatRequest(BaseModel):
    content: str

@app.post("/chat")
async def chat(request: ChatRequest):
    config = RailsConfig.from_path("./config")
    rails = LLMRails(config)

    response = await rails.generate_async(messages=[{
        "role": "user",
        "content": request.content
    }])
    
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
