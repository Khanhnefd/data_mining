from fastapi import FastAPI
import service
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()

FASTAPI_PORT = os.getenv("FASTAPI_PORT")
FASTAPI_HOST = os.getenv("FASTAPI_HOST")


app = FastAPI(title="Logging data capstone project")

app.include_router(service.router)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}


if __name__ == "__main__":
    uvicorn.run(app, host=FASTAPI_HOST, port=int(FASTAPI_PORT))
