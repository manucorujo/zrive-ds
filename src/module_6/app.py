import uvicorn
from fastapi import FastAPI
from fastapi import status

# Create an instance of FastAPI
app = FastAPI()


# Define a route for the root URL ("/")
@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


@app.get("/status")
def read_status():
    return {"status": status.HTTP_200_OK}


# This block allows you to run the application using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
