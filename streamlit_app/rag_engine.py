"""Evidence-grounded RAG: local trusted passages, vector retrieval, optional LLM.
No invented citations. Optional FAISS multilingual embeddings; TF-IDF fallback.
"""
from __future__ import annotations
import json, os, re
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / 'knowledge' / 'guidance.json'
INGESTED = ROOT / 'knowledge' / 'ingested_chunks.json'

@lru_cache(maxsize=1)
def documents():
    docs = json.loads(KB.read_text(encoding='utf-8'))
    if INGESTED.exists():
        docs.extend(json.loads(INGESTED.read_text(encoding='utf-8')))
    return [d for d in docs if all(d.get(k) for k in ('id','title','section','url','text'))]

@lru_cache(maxsize=1)
def tfidf_index():
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs=documents()
    vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words='english',sublinear_tf=True)
    matrix=vectorizer.fit_transform([f"{d['title']} {d['section']} {d.get('keywords','')} {d['text']}" for d in docs])
    return vectorizer,matrix

@lru_cache(maxsize=1)
def embedding_index():
    """Use real multilingual sentence embeddings + FAISS if optional packages installed."""
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    docs=documents()
    texts=[f"{d['title']}. {d['section']}. {d.get('keywords','')}. {d['text']}" for d in docs]
    embeddings=model.encode(texts,normalize_embeddings=True,convert_to_numpy=True).astype('float32')
    idx=faiss.IndexFlatIP(embeddings.shape[1]);idx.add(embeddings)
    return model,idx

def retrieve(question: str, k: int=3, use_embeddings: bool=False):
    if not question.strip():return []
    docs=documents()
    if use_embeddings:
        try:
            import numpy as np
            model,idx=embedding_index()
            q=model.encode([question],normalize_embeddings=True,convert_to_numpy=True).astype('float32')
            scores,indices=idx.search(q,min(k,len(docs)))
            return [dict(docs[int(i)],score=float(score)) for i,score in zip(indices[0],scores[0]) if i>=0 and score>=0.32]
        except (ImportError, OSError):
            pass
    from sklearn.metrics.pairwise import cosine_similarity
    vectorizer,matrix=tfidf_index()
    scores=cosine_similarity(vectorizer.transform([question]),matrix)[0]
    ranked=sorted(enumerate(scores),key=lambda x:x[1],reverse=True)
    return [dict(docs[i],score=float(score)) for i,score in ranked[:k] if score>=0.075]

def citations(passages):
    return '\n'.join(f"- **[{d['id']}]** {d['title']} — {d['section']}: {d['url']}" for d in passages)

def compose_answer(question: str, passages: list, api_key: str='', model: str='llama-3.1-8b-instant'):
    """Optional Groq LLM (key in Streamlit secrets); otherwise honest retrieval-only answer."""
    if not passages:
        return "I couldn't verify an answer from the available trusted guidance. Please consult official local guidance.", 'abstained'
    if api_key:
        try:
            import requests
            context='\n\n'.join(f"[{d['id']}] {d['title']} | {d['section']} | {d['text']}" for d in passages)
            response=requests.post('https://api.groq.com/openai/v1/chat/completions',headers={'Authorization':f'Bearer {api_key}'},json={
                'model':model,'temperature':0,'max_tokens':450,
                'messages':[{'role':'system','content':'You are HeatSafe Pakistan evidence assistant. Answer ONLY from supplied context. Cite document IDs like [WHO-HEAT-1] for each supported claim. If evidence is insufficient, say so. Do not fabricate data, official warnings, or medical diagnoses. Explain clearly in the language of the question.'},
                            {'role':'user','content':f'Question: {question}\n\nVerified context:\n{context}'}]
            },timeout=25)
            response.raise_for_status()
            answer=response.json()['choices'][0]['message']['content'].strip()
            allowed={d['id'] for d in passages}
            cited=set(re.findall(r'\[([A-Za-z0-9_-]+)\]',answer))
            if cited and cited.issubset(allowed):return answer,'llm_grounded'
        except Exception:
            pass
    summary='\n\n'.join(f"**[{d['id']}] {d['section']}**\n{d['text']}" for d in passages)
    return f"**Relevant evidence (retrieval summary; no generative LLM used):**\n\n{summary}", 'retrieval_only'

def answer_guidance(question, api_key='', use_embeddings=False):
    passages=retrieve(question,use_embeddings=use_embeddings)
    answer,method=compose_answer(question,passages,api_key=api_key)
    if passages:answer+='\n\n**Sources**\n'+citations(passages)
    return {'answer':answer,'method':method,'sources':[d['id'] for d in passages]}
