from fastapi import FastAPI
from pydantic import BaseModel

from model import Handler

# See https://fastapi.tiangolo.com/tutorial/first-steps/ for more general information on FastAPI

app = FastAPI()
handler = Handler()

# Create schemas for the input and output for the POST requests
# See: https://fastapi.tiangolo.com/tutorial/body/
class Text(BaseModel):
    text: str

class Texts(BaseModel):
    text_1: str
    text_2: str
    
class TextResponse(BaseModel):
    textresponse: str
    
class TextsResponse(BaseModel):
    textsresponse: str
    
    
# Create FastAPI endpoints here
# See an example of a POST request here: https://fastapi.tiangolo.com/tutorial/body/#declare-it-as-a-parameter

@app.post("/embed")
def get_embs(text: str):
    embedding = handler.embed(text)
    return {"Text Embeding for {} is {}".format(text, embedding)}

@app.post("/compare")
def get_sim_score(text_1: str, text_2: str):
    sim_score = handler.similarity(text_1, text_2)
    return {'Similarity Score for {} and {} is {}'.format(text_1, text_2, sim_score)}
    


# This part runs the FastAPI app if you execute this Python file using `python app.py`.
# You do not have to change this part.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
