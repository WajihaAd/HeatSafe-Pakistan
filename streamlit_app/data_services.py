"""Persistent published Prefect results, with explicit freshness and provenance."""
from datetime import datetime, timezone
import requests

def read_published_results(base_url: str):
    """base_url is public raw content URL folder ending in /; never requires tokens."""
    if not base_url or not base_url.startswith('https://'):
        return {'available':False,'reason':'No public published-results URL configured'}
    try:
        results={}
        for filename in ('latest_prediction.json','live_quality_report.json'):
            r=requests.get(base_url.rstrip('/')+'/'+filename,timeout=12,headers={'Cache-Control':'no-cache'})
            r.raise_for_status();results[filename]=r.json()
        return {'available':True,'prediction':results['latest_prediction.json'],'quality':results['live_quality_report.json']}
    except (requests.RequestException,ValueError) as exc:
        return {'available':False,'reason':f'{type(exc).__name__}: {exc}'}
