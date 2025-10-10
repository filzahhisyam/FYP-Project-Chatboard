import os
import json

# === CONFIG ===
HANDBOOK_DIR = "handbook"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def chunk_text(text, chunk_size=400, overlap=50):
    """Split text into overlapping chunks (for embeddings)."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def main():
    sections = []
    chunks = []

    for filename in os.listdir(HANDBOOK_DIR):
        if filename.endswith(".txt"):
            section_name = filename.replace(".txt", "")
            with open(os.path.join(HANDBOOK_DIR, filename), "r", encoding="utf-8") as f:
                text = f.read().strip()

            # Add full section to sections.json
            sections.append({
                "section": section_name,
                "text": text
            })

            # Split into chunks
            section_chunks = chunk_text(text)
            for idx, c in enumerate(section_chunks):
                chunks.append({
                    "section": section_name,
                    "chunk_id": f"{section_name}_{idx}",
                    "text": c
                })

    # Save outputs
    with open(os.path.join(DATA_DIR, "sections.json"), "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    with open(os.path.join(DATA_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(sections)} sections → data/sections.json")
    print(f"✅ Saved {len(chunks)} chunks → data/chunks.json")

if __name__ == "__main__":
    main()
