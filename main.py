from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import io
import os
import gcsfs
from cv_utils import process_image_cv
from crud import save_search_history, get_user_history
from db import engine
from models import Base

Base.metadata.create_all(bind=engine)  # Auto-create table on launch



# ---------- CONFIG ----------
BUCKET_NAME = "fashionclip-api"
MODEL_PATH = f"{BUCKET_NAME}/model"
EMBEDDING_PATH = f"{BUCKET_NAME}/embeddings"

fs = gcsfs.GCSFileSystem(project="semesterproject")

# ---------- APP SETUP ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- LOAD MODEL ----------
with fs.open(f"{MODEL_PATH}/config.json") as f:
    config_data = f.read()
with fs.open(f"{MODEL_PATH}/preprocessor_config.json") as f:
    preproc_data = f.read()

model = CLIPModel.from_pretrained(f"gs://{MODEL_PATH}")
processor = CLIPProcessor.from_pretrained(f"gs://{MODEL_PATH}")
model.eval()

# ---------- LOAD EMBEDDINGS ----------
with fs.open(f"{EMBEDDING_PATH}/MetaData.csv") as f:
    meta_df = pd.read_csv(f)

with fs.open(f"{EMBEDDING_PATH}/image_embeddings.npy", "rb") as f:
    image_embs = np.load(f)
with fs.open(f"{EMBEDDING_PATH}/caption_embeddings.npy", "rb") as f:
    caption_embs = np.load(f)

image_embs = F.normalize(torch.tensor(image_embs), dim=-1).cpu().numpy()
caption_embs = F.normalize(torch.tensor(caption_embs), dim=-1).cpu().numpy()

# ---------- UTILS ----------
def fuzzy_match(target, candidates, threshold=0.7):
    return any(SequenceMatcher(None, target.lower(), c.lower()).ratio() >= threshold for c in candidates)

# ---------- ROUTES ----------
@app.get("/")
def root():
    return {"message": "FashionCLIP API is running with GCS integration!"}

@app.post("/search")
def search(
    caption: str = Form(None),
    image: UploadFile = File(None),
    user_id: str = Form("guest@user.com")  # can be passed from frontend
):
    top_k = 2

    if caption:
        inputs = processor(text=[caption], return_tensors="pt", padding=True)
        with torch.no_grad():
            query_emb = model.get_text_features(**inputs)
        query_emb = F.normalize(query_emb, dim=-1).cpu().numpy()
        similarities = cosine_similarity(query_emb, image_embs)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

    elif image:
        image_bytes = image.file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_img = process_image_cv(pil_img)
        inputs = processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            query_emb = model.get_image_features(**inputs)
        query_emb = F.normalize(query_emb, dim=-1).cpu().numpy()
        similarities = cosine_similarity(query_emb, caption_embs)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

    else:
        return {"error": "No input provided."}

    results = []
    for idx in top_indices:
        row = meta_df.iloc[idx]
        image_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{row.get('Downloaded Image Path', '')}"

        results.append({
            "caption": row.get("Caption", "N/A"),
            "url": row.get("URL", "N/A"),
            "color": row.get("Color", "N/A"),
            "fabric": row.get("Fabric", "N/A"),
            "brand": row.get("Brand", "N/A"),
            "image_url": image_url
        })

    # Save to history
    search_type = "caption" if caption else "image"
    query_input = caption or image.filename
    save_search_history(user_id=user_id, search_type=search_type, query_input=query_input, results=results)

    return {"results": results}

@app.get("/user/history")
def history(user_id: str):
    user_logs = get_user_history(user_id)
    return {"history": user_logs}
