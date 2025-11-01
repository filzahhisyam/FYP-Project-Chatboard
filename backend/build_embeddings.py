from sentence_transformers import SentenceTransformer
import numpy as np, json, os

os.environ["USE_TF"] = "0"

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
texts = [c["text"] for c in chunk_list]
model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

np.save("data/embeddings.npy", embs)
print(f"✅ Saved {len(embs)} embeddings for {len(chunk_list)} chunks")
print(f"✅ Using your corrected chunks from chunks.json")