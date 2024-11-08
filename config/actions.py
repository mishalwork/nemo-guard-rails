from fastapi import FastAPI, HTTPException
import asyncio
from nemoguardrails import LLMRails, RailsConfig
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import time
from datetime import datetime
from functools import lru_cache
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Chi AI Service")

class ChatRequest(BaseModel):
    content: str

# Cache for configuration and rails instance
@lru_cache(maxsize=1)
def get_rails_instance():
    """Cache the rails configuration and instance"""
    try:
        logger.info("Initializing Rails instance...")
        config = RailsConfig.from_path("./config")
        rails = LLMRails(config)
        logger.info("Rails instance initialized successfully")
        return rails
    except Exception as e:
        logger.error(f"Failed to initialize Rails: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize AI system")

# Initialize rails instance at startup
rails_instance = get_rails_instance()

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with error handling and performance monitoring"""
    start_time = time.time()
    
    try:
        logger.info(f"Received chat request: {request.content}")
        
        # Generate response
        generation_start = time.time()
        response = await rails_instance.generate_async(messages=[{
            "role": "user",
            "content": request.content
        }])
        generation_time = time.time() - generation_start
        
        # Calculate total time
        total_time = time.time() - start_time
        
        logger.info(f"Generated response in {generation_time:.3f} seconds")
        
        return {
            "response": response,
            "timing_metrics": {
                "total_time_seconds": round(total_time, 3),
                "generation_time": round(generation_time, 3),
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    """Root endpoint for basic API information"""
    return {
        "status": "online",
        "service": "Chi AI Chat Service",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

def start_server():
    """Start the server with the correct configuration"""
    try:
        uvicorn.run(
            "config.actions:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            workers=1  # Start with 1 worker for debugging
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

if __name__ == "__main__":
    start_server()
