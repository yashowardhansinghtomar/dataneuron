from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

class TextPair(BaseModel):
    text1: str
    text2: str

@app.get("/")
def read_root():
    return {"message": "Semantic Similarity API"}

@app.post("/similarity/")
def get_similarity_score(pair: TextPair):
    embeddings = model.encode([pair.text1, pair.text2])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    similarity =min(max(similarity, 0), 1)
    return {"similarity score": round(similarity, 3)}
