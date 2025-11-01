# backend/qa_pipeline.py

import numpy as np, json, re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sentence_transformers.cross_encoder import CrossEncoder

# Load data
embs = np.load("data/embeddings.npy")
with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load models
embedder = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Retrieve Function
def retrieve(question, top_k=5, metadata_filter=None):
    """Retrieve the most relevant handbook chunks for a question."""
    q_emb = embedder.encode([question], convert_to_numpy=True)
    all_scores = cosine_similarity(q_emb, embs)[0]

    # List of (chunk_index, score)
    all_scores = list(enumerate(all_scores))

    # Filter by section if needed
    if metadata_filter:
        section = metadata_filter.get("section")
        target_scores = [(i, s) for i, s in all_scores if chunks[i]["section"] == section]
        if not target_scores:
            target_scores = all_scores  
    else:
        target_scores = all_scores

    # Sort and get top candidates
    target_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [i for i, _ in target_scores[:20]]

    # Rerank with cross-encoder
    query_pairs = [[question, chunks[i]["text"]] for i in top_candidates]
    rerank_scores = cross_encoder.predict(query_pairs)
    ranked = sorted(zip(rerank_scores, top_candidates), key=lambda x: x[0], reverse=True)

    final_idxs = [idx for _, idx in ranked[:top_k]]
    return [chunks[i] for i in final_idxs]


# Main QA Function
def answer_question(question):
    q_lower = question.lower().strip()
    
    # Small Talk 
    if q_lower.startswith("hello") or q_lower.startswith("hi") or q_lower.startswith("hey"):
        return "Hello! How can I help you with the handbook today?"
    if "thank" in q_lower or q_lower in ["thanks", "thx"]:
        return "You're very welcome! Anything else you'd like to know?"
        
    # Detect vague or list-type queries
    summary_keywords = [
        "benefits", "incentives", "leave", "policy", "policies",
        "rules", "list", "examples", "promotion", "appraisal", "overtime"
    ]
    summary_prefixes = (
        "what are", "list all", "list the", "show me", "tell me about",
        "can i know about", "what about", "explain"
    )

    is_vague = (
        len(q_lower.split()) <= 2 and any(k in q_lower for k in summary_keywords)
    ) or (
        q_lower.startswith(summary_prefixes) and any(k in q_lower for k in summary_keywords)
    )

    if is_vague:
        top_k_val = 6 
    else:
        top_k_val = 1

    # Section-based keyword routing
    metadata_filter = None
    section_map = {
        "employment_contract": [
            "probation", "confirmation", "employment", "contract", "confirmed employee"
        ],
        "rules_regulations": [
            "rules", "regulations", "misconduct", "discipline", "warning"
        ],
        "leave": [
            "leave", "vacation", "annual", "maternity", "paternity", "marriage", "emergency"
        ],
        "medical_benefits": [
            "benefit", "insurance", "dental", "hospital", "medical", "coverage"
        ],
        "incentives": [
            "incentive", "allowance", "bonus trip", "education assistance", "eap"
        ],
        "salary": [
            "salary", "epf", "socso", "bonus", "pay", "overtime rate"
        ],
        "performance": [
            "performance", "appraisal", "kpi", "evaluation"
        ],
        "work_hours": [
            "working hours", "work hours", "shift", "lunch", "schedule", "time in", "clock in"
        ],
        "overtime": [
            "overtime", "time off in lieu", "extra hours", "ot","meal allocation", "meal"
        ],
        "transfer_promotion": [
            "promotion", "transfer", "demotion", "career movement"
        ],
        "leaving": [
            "termination", "resign", "retire", "quit", "end of service", "leaving","fired"
        ]
    }

    for section, keywords in section_map.items():
        if any(k in q_lower for k in keywords):
            metadata_filter = {"section": section}
            break

    # Retrieve Relevant Chunks
    contexts = retrieve(question, top_k=top_k_val, metadata_filter=metadata_filter)
    
    if not contexts:
        return "Sorry, I couldn't find any relevant information in the handbook."
    
    q_emb = embedder.encode([question], convert_to_numpy=True)
    top_context = contexts[0]["text"]
    c_emb = embedder.encode([top_context], convert_to_numpy=True)
    similarity = cosine_similarity(q_emb, c_emb)[0][0]

    if similarity < 0.3:
        return (
        "That topic doesn’t seem to be covered in the employee handbook. "
        "Please check with HR for clarification."
    )

    context_text = " ".join([c["text"] for c in contexts])
    source_section = contexts[0]["section"].replace("_", " ").title()

    # --- 5. If vague → show summary list ---
    if is_vague:
        bullets = "\n\n".join([f"• {c['text']}" for c in contexts])
        return f"📘 Here's what I found about **{question}** (from {source_section}):\n\n{bullets}"

    # --- 6. Use QA Model for specific queries ---
    result = qa(question=question, context=context_text)
    answer, score = result["answer"], result["score"]
    CONF_THRESHOLD = 0.1

    # --- 7. Fallback logic ---
    if score < CONF_THRESHOLD or len(answer.split()) <= 3:
        # Find full sentence
        start_idx = result.get("start", 0)
        sentence_start = context_text.rfind('.', 0, start_idx) + 1
        
        if sentence_start <= 0:
            sentence_start = context_text.rfind('\n', 0, start_idx) + 1
        sentence_end = context_text.find('.', start_idx) + 1
        
        if sentence_end <= 0:
            sentence_end = len(context_text)
        full_sentence = context_text[sentence_start:sentence_end].strip()
        full_sentence = re.sub(r"^\W+", "", full_sentence)
        
        if not full_sentence:
            full_sentence = context_text
        return f"📘 From handbook section: **{source_section}**\n\n{full_sentence}"

    # --- 8. Confident direct answer ---
    return f"📘 From handbook section: **{source_section}** {answer}\n\n "


#--------------------------------------------------------------------------------------------
