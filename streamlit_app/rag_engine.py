"""Grounded retrieval for HeatSafe; optional LLM, no fabricated citations."""
from __future__ import annotations
import json,re
from pathlib import Path
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[1]
@lru_cache(maxsize=1)
def documents():
    docs=[]
    for p in (ROOT/'knowledge'/'guidance.json',ROOT/'knowledge'/'ingested_chunks.json'):
        if p.exists():
            try: docs.extend(json.loads(p.read_text(encoding='utf-8')))
            except (ValueError,OSError):pass
    return [d for d in docs if all(d.get(k) for k in ('id','title','section','url','text'))]
@lru_cache(maxsize=1)
def index():
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs=documents()
    if not docs:return None,None
    v=TfidfVectorizer(ngram_range=(1,2),stop_words='english',sublinear_tf=True)
    return v,v.fit_transform([f"{d['title']} {d['section']} {d.get('keywords','')} {d['text']}" for d in docs])
def retrieve(question,k=3):
    from sklearn.metrics.pairwise import cosine_similarity
    v,m=index()
    if v is None or not question.strip():return []
    scores=cosine_similarity(v.transform([question]),m)[0]
    return [dict(documents()[i],score=float(scores[i])) for i in scores.argsort()[::-1][:k] if scores[i]>=0.065]
def answer_guidance(question,api_key=''):
    docs=retrieve(question)
    if not docs:return {'answer':"I couldn't find enough evidence in the current trusted knowledge base to answer that. Try a question about heat precautions, humidity or official warnings.",'sources':[],'method':'abstained'}
    citations='\n'.join(f"- **[{d['id']}]** {d['title']} — {d['section']} ([source]({d['url']}))" for d in docs)
    answer=''
    if api_key:
        try:
            import requests
            context='\n'.join(f"[{d['id']}] {d['text']}" for d in docs)
            r=requests.post('https://api.groq.com/openai/v1/chat/completions',headers={'Authorization':f'Bearer {api_key}'},json={'model':'llama-3.1-8b-instant','temperature':0,'max_tokens':400,'messages':[{'role':'system','content':'Answer only from the provided context. Cite the exact evidence IDs in brackets for factual statements. Abstain if insufficient. Never invent weather readings, source links or medical claims.'},{'role':'user','content':f'Question: {question}\nContext:\n{context}'}]},timeout=20)
            r.raise_for_status();candidate=r.json()['choices'][0]['message']['content'].strip()
            cited=set(re.findall(r'\[([A-Za-z0-9_-]+)\]',candidate))
            if cited and cited.issubset({d['id'] for d in docs}):answer=candidate
        except Exception:pass
    if not answer:
        answer='**Evidence from trusted guidance:**\n\n'+'\n\n'.join(f"**{d['section']} [{d['id']}]** — {d['text']}" for d in docs)
    return {'answer':answer+'\n\n**Sources**\n'+citations,'sources':[d['id'] for d in docs],'method':'llm' if api_key and answer and not answer.startswith('**Evidence') else 'retrieval'}
