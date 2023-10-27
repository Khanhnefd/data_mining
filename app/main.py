from fastapi import FastAPI
import service
import uvicorn


app = FastAPI(
    title="Logging data capstone project"
)

app.include_router(service.router)
@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)