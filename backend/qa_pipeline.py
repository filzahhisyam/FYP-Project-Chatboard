# backend/qa_pipeline.py

import numpy as np
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sentence_transformers.cross_encoder import CrossEncoder

# --- 1. LOAD ALL MODELS AND DATA (Lightweight) ---

# Load embeddings + chunks
embs = np.load("data/embeddings.npy")
with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load ONLY the models we need. This is light enough for deployment.
embedder = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


# --- 2. THE RETRIEVE FUNCTION (With 2 Return Values) ---

def retrieve(question, top_k=5, metadata_filter=None):
    
    q_emb = embedder.encode([question], convert_to_numpy=True)
    
    # 1. Get semantic scores for ALL chunks
    all_semantic_scores = cosine_similarity(q_emb, embs)[0]
    
    # 2. Create a list of (original_index, score)
    all_scores = list(enumerate(all_semantic_scores))
    
    # 3. If there's a filter, filter the list.
    if metadata_filter:
        target_scores = [
            (i, score) for i, score in all_scores 
            if chunks[i]["section"] == metadata_filter["section"]
        ]
        # Handle case where filter finds nothing
        if not target_scores:
            target_scores = all_scores # Fallback to all scores
    # 4. If no filter, use the full list.
    else:
        target_scores = all_scores
        
    # 5. Sort the target list by score
    target_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 6. Get the top 20 candidate *indexes*
    top_k_candidates = 50
    if len(target_scores) < top_k_candidates:
        top_k_candidates = len(target_scores)
        
    top_candidate_original_idxs = [i for i, score in target_scores[:top_k_candidates]]

    # --- Re-ranker ---
    candidate_texts = [chunks[i]["text"] for i in top_candidate_original_idxs]
    query_pairs = [[question, text] for text in candidate_texts]
    new_scores = cross_encoder.predict(query_pairs)
    
    ranked_results = sorted(zip(new_scores, top_candidate_original_idxs), key=lambda x: x[0], reverse=True)
    
    # --- HANDLE EDGE CASE: No results ---
    if not ranked_results:
        return [], 0.0

    final_top_idxs = [idx for score, idx in ranked_results[:top_k]]
    
    candidates = [chunks[i] for i in final_top_idxs]
    
    # --- THIS IS THE FIX for the ValueError ---
    # Get the score of the single best chunk
    top_score = ranked_results[0][0]
    return candidates, top_score
    # --- END OF FIX ---

# In backend/qa_pipeline.py

def answer_question(question):
    
    # --- 1. SMALL TALK HANDLER ---
    q_lower = question.lower().strip() 
    words = set(q_lower.split())
    if words.intersection(["hello", "hi", "hey"]):
        return "Hello! 👋 How can I help you with the handbook today?"
    if "thank" in q_lower or q_lower in ["thanks", "thx"]:
        return "You're very welcome! 😊 Anything else you'd like to know?"
    
    # --- 2. DYNAMIC TOP_K LOGIC ---
    summary_keywords = ["benefits", "incentives", "leave", "policy", "policies", 
                        "rules", "list", "examples", "promotion", "appraisal", "overtime",
                         "sick", "illness","code of conduct","grievance procedure","salary"]
    
    summary_prefixes = ("what are", "list all", "list the", "show me", 
                        "can i know about", "tell me about", "what about", "explain",
                        "what happens")

    is_short_vague = (len(q_lower.split()) <= 2 and any(word in q_lower for word in summary_keywords))
    is_summary_question = (q_lower.startswith(summary_prefixes) and any(word in q_lower for word in summary_keywords))
    
    is_vague = is_short_vague or is_summary_question
    
    if is_vague:
        top_k_val = 6
    else:
        top_k_val = 1
            
    # --- 3. METADATA FILTER ---
    metadata_filter = None
    
    section_map = {
        "employment_contract": ["probation", "confirmation", "appointment","trial period","job offer"],
        "rules_regulations": ["rules", "regulations", "misconduct", "disciplinary"],
        "leave": ["leave", "vacation", "paternity", "maternity", "compassionate", "exam","time off", "sick leave"],
        "medical_benefits": ["benefit", "insurance", "dental", "medical", "hospitalisation", "sick", "illness"],
        "incentives": ["incentive", "allowance", "trip", "education", "eap", "recreational", "phone", "training"],
        "salary": ["salary", "epf", "socso", "bonus", "increment", "tax"],
        "performance": ["performance", "appraisal", "kpi", "interview"],
        "work_hours": ["working hours", "work hours", "lunch","office hours"],
        "overtime": ["overtime", "time off in lieu", "meal allowance","ot","working hours","extra hours"],
        "transfer_promotion": ["promotion", "transfer","promoted",],
        "leaving": ["termination", "quit", "resign", "retire", "fired", "abandonment"],
        "it_policy": ["it policy","it policies", "computer", "software"],
        "grievance_procedure": ["Grievance Policy", "grievance", "grievance procedure", "complaint", "problem with manager", "issue"],
        "code_of_conduct": ["code of conduct", "ethics", "how to act", "behave", "gifts", "confidential"],
        "introduction":["vision","mission"]
    }

    for section, keywords in section_map.items():
        if any(k in q_lower for k in keywords):
            metadata_filter = {"section": section}
            break 
    
    # --- 4. RETRIEVE and GET THE SCORE ---
    contexts, top_retrieval_score = retrieve(question, top_k=top_k_val, metadata_filter=metadata_filter) 
    
    if not contexts:
        return "I'm sorry, I could not find an answer to that in the handbook."
        
    # --- 5. ANSWERING LOGIC ---
    context_text = " ".join([c["text"] for c in contexts])

    if is_vague:
        # --- Handle BROAD questions ---
        # We trust the metadata filter, so we don't check the score here.
        
        all_sections = [c["section"].replace("_", " ").title() for c in contexts]
        unique_sections = list(dict.fromkeys(all_sections))
        source_str = ", ".join(unique_sections)
        bullet_points = [f"• {c['text']}" for c in contexts]
        context_text = "\n\n".join(bullet_points)
        return f"📘 Here's what I found about **{question}** (from {source_str}):\n\n{context_text}"
    
    else:
        # --- Handle SPECIFIC questions ---
        
        # --- THIS IS THE NEW, SMARTER "I DON'T KNOW" LOGIC ---
        RETRIEVAL_THRESHOLD = 0.1
        
        # We only say "I don't know" IF
        # 1. No filter was found (e.g., "dress code")
        # AND
        # 2. The score is low
        if metadata_filter is None and top_retrieval_score < RETRIEVAL_THRESHOLD:
            return "I'm sorry, I could not find an answer to that in the handbook."
        # --- END OF FIX ---

        # If a filter *was* found (like for "training"), we proceed,
        # even if the score is low.
        
        context_text = " ".join([c["text"] for c in contexts])
        source_section = contexts[0]["section"].replace("_", " ").title()
        result = qa(question=question, context=context_text)
        answer = result["answer"]
        qa_score = result["score"]
        
        CONF_THRESHOLD = 0.1  
        
        # --- Fallback 1: Low QA Score ---
        if qa_score < CONF_THRESHOLD:
            return f"{context_text}\n\n(Source: Handbook Section: **{source_section}**)"

        # --- Fallback 2: "Short Answer" Expansion ---
        if len(answer.split()) <= 5:
            start_index = result['start']
            end_index = result['end']
            
            sentence_start = context_text.rfind('.', 0, start_index) + 1
            if sentence_start == 0:
                sentence_start = context_text.rfind('\n', 0, start_index) + 1
            
            sentence_end = context_text.find('.', end_index) + 1
            if sentence_end == 0: 
                sentence_end = context_text.find('\n', end_index) + 1
            if sentence_end == 0:
                sentence_end = len(context_text)
            
            full_sentence = context_text[sentence_start:sentence_end].strip()
            final_answer = re.sub(r"^\W+", "", full_sentence)

            if not final_answer: 
                final_answer = context_text
            return f"{final_answer}\n\n(Source: Handbook Section: **{source_section}**)"
        
        else:
            # --- Confident, good-length answer ---
            return f"{answer}\n\n(Source: Handbook Section: **{source_section}**)"

    