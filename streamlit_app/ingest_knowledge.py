"""Validate and chunk permitted local source text for RAG.
Input: knowledge/source_documents/*.txt with matching .json metadata files.
Each metadata JSON must contain title, section, url, organization, license.
Only ingest content you have permission to reuse.
Run: python streamlit_app/ingest_knowledge.py
"""
from pathlib import Path
import json, re
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'knowledge'/'source_documents'
OUT=ROOT/'knowledge'/'ingested_chunks.json'

def chunk_text(text, max_words=170, overlap=30):
    words=re.split(r'\s+',text.strip());result=[];step=max_words-overlap
    for i in range(0,len(words),step):
        segment=' '.join(words[i:i+max_words]).strip()
        if segment:result.append(segment)
        if i+max_words>=len(words):break
    return result

def ingest():
    docs=[]
    for path in sorted(SRC.glob('*.txt')):
        meta_file=path.with_suffix('.json')
        if not meta_file.exists():raise ValueError(f'Missing metadata for {path.name}')
        meta=json.loads(meta_file.read_text(encoding='utf-8'))
        for key in ('title','section','url','organization','license'):
            if not meta.get(key):raise ValueError(f'Missing {key} in {meta_file.name}')
        for n,chunk in enumerate(chunk_text(path.read_text(encoding='utf-8'))):
            docs.append({'id':f'{path.stem}-{n+1:03}',**meta,'text':chunk,'keywords':meta.get('keywords','')})
    OUT.write_text(json.dumps(docs,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Indexed {len(docs)} permitted chunks in {OUT}')
    return docs
if __name__=='__main__':ingest()
