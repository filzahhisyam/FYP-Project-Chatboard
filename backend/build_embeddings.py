import os
os.environ["TRANSFORMERS_NO_TF"] = "1" 

from sentence_transformers import SentenceTransformer
import numpy as np, json

# ---------------------------
# Load corrected chunks from chunks.json
# ---------------------------
try:
    with open("data/chunks.json", "r", encoding="utf-8") as f:
        chunk_list = json.load(f)
    print(f"Loaded {len(chunk_list)} chunks from chunks.json")
except FileNotFoundError:
    print("chunks.json not found. Please create it first with your corrected chunks.")
    exit(1)

# ---------------------------
# Encode chunks into embeddings
# ---------------------------
print("Loading sentence transformer model...") # Added a print statement for you
texts = [c["text"] for c in chunk_list]
model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

print("Model loaded. Starting encoding...") # Added another print statement
embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

np.save("data/embeddings.npy", embs)
print(f"✅ Save {len(embs)} embeddings for {len(chunk_list)} chunks")
print(f"✅ Using your corrected chunks from chunks.json")