"""HeatSafe Pakistan: five-page Streamlit weather intelligence dashboard.
Launch from repository root using Streamlit Community Cloud entrypoint streamlit_app/app.py.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import re
import sys
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from rag_engine import answer_guidance
from data_services import read_published_results

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed' / 'heatsafe_ml_dataset.csv'
KNOWLEDGE = ROOT / 'knowledge' / 'guidance.json'
MODEL = ROOT / 'models' / 'heatsafe_live_improved_model.pkl'
RISK = {0:'Low',1:'Moderate',2:'High',3:'Extreme'}
st.set_page_config(page_title='HeatSafe Pakistan',page_icon='🌤️',layout='wide')
st.markdown('''<style>
.stApp {background:#0b1526;color:#edf4ff}
[data-testid="stSidebar"] {background:#111f34}
[data-testid="stMetric"] {background:#17283e;border:1px solid #30425b;padding:16px;border-radius:14px}
h1,h2,h3 {color:#f3f8ff!important}
.smallnote {color:#b3c6d9;font-size:0.88rem}
</style>''',unsafe_allow_html=True)

@st.cache_data(ttl=600,show_spinner=False)
def live_weather():
    url='https://api.open-meteo.com/v1/forecast'
    params={'latitude':24.8607,'longitude':67.0011,'current':'temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,wind_speed_10m','timezone':'Asia/Karachi','wind_speed_unit':'kmh'}
    r=requests.get(url,params=params,timeout=15);r.raise_for_status()
    data=r.json()
    return data['current'],data.get('timezone','Asia/Karachi')

@st.cache_data(show_spinner=False)
def historical():
    df=pd.read_csv(DATA)
    df['time']=pd.to_datetime(df['time'],utc=True,errors='coerce')
    df=df.dropna(subset=['time']).sort_values('time')
    df['row_id']='KHI_'+df['time'].dt.strftime('%Y%m%dT%H%M%SZ')
    return df

@st.cache_data(show_spinner=False)
def guidance():
    return json.loads(KNOWLEDGE.read_text(encoding='utf-8'))

def risk_prediction(current):
    try:
        import joblib
        saved=joblib.load(MODEL)
        model,features=saved['model'],saved['features']
        local=pd.Timestamp(current['time'])
        if local.tzinfo is None: local=local.tz_localize('Asia/Karachi')
        else: local=local.tz_convert('Asia/Karachi')
        row={'meteo_temperature':current['temperature_2m'],'meteo_humidity':current['relative_humidity_2m'],'dew_point_2m':current['dew_point_2m'],'apparent_temperature':current['apparent_temperature'],'meteo_wind_speed':current['wind_speed_10m'],'hour_karachi':local.hour,'month':local.month}
        pred=int(model.predict(pd.DataFrame([row])[features])[0])
        return RISK.get(pred,'Unknown'),None
    except Exception as exc:
        return None,f'Model unavailable or incompatible: {type(exc).__name__}: {exc}'

def freshness(current):
    try:
        t=pd.Timestamp(current['time'])
        if t.tzinfo is None:t=t.tz_localize('Asia/Karachi')
        age=(pd.Timestamp.now(tz='UTC')-t.tz_convert('UTC')).total_seconds()/3600
        return age,('PASS' if -0.25<=age<=2 else 'REVIEW_REQUIRED')
    except Exception:return None,'REVIEW_REQUIRED'

def weather_panel():
    try:
        w,tz=live_weather()
    except Exception as exc:
        st.error(f'Live weather unavailable: {exc}');return None
    age,status=freshness(w)
    cols=st.columns(4)
    for col,(label,key,unit) in zip(cols,[('Temperature','temperature_2m','°C'),('Feels like','apparent_temperature','°C'),('Humidity','relative_humidity_2m','%'),('Wind speed','wind_speed_10m','km/h')]):
        col.metric(label,f'{w[key]} {unit}')
    st.caption(f"Observed {w['time']} ({tz}) · Open-Meteo · Quality: {status} · age: {age:.1f} hours" if age is not None else 'Observation timestamp unavailable')
    if status!='PASS':st.warning('Live observation did not pass freshness checks. Prediction withheld.');return w
    label,error=risk_prediction(w)
    if error:st.warning(error)
    else:st.info(f'Experimental model — predicted next-hour heat-risk category: **{label}**. Not a medical safety clearance.')
    return w

def search_knowledge(question):
    """Retrieve supporting passages using TF-IDF vector similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    docs=guidance()
    texts=[d['title']+' '+d['section']+' '+d['keywords']+' '+d['text'] for d in docs]
    try:
        vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words='english')
        matrix=vectorizer.fit_transform(texts)
        query=vectorizer.transform([question])
        scores=cosine_similarity(query,matrix)[0]
        ranked=sorted(enumerate(scores),key=lambda p:p[1],reverse=True)
        return [docs[i] for i,score in ranked[:2] if score>=0.075]
    except ValueError:
        return []

def answer_question(question):
    q=question.lower()
    if any(s in q for s in ['current temperature','temperature now','weather now','current weather','humidity now','feels like now']):
        try:
            w,_=live_weather();return f"Open-Meteo reports Karachi at {w['time']} (Asia/Karachi): temperature {w['temperature_2m']} °C, feels-like {w['apparent_temperature']} °C, humidity {w['relative_humidity_2m']}%, wind {w['wind_speed_10m']} km/h. Source: Open-Meteo current weather API. These are observations, not forecasts."
        except Exception:return 'I cannot verify the current weather right now. Please try again later.'
    if any(s in q for s in ['risk now','current risk','risk prediction','heat risk today']):
        try:
            w,_=live_weather();_,status=freshness(w)
            if status!='PASS':return 'The current weather did not pass freshness checks, so I cannot give a reliable model prediction.'
            label,err=risk_prediction(w)
            if err:return 'The experimental model is unavailable. I cannot verify a prediction.'
            return f'Experimental next-hour heat-risk classification: {label}. Observation time: {w["time"]} Asia/Karachi. Source: Open-Meteo observation + HeatSafe trained model. This is not a medically validated warning.'
        except Exception:return 'I cannot verify a current prediction right now.'
    if any(s in q for s in ['historical','history','average temperature','hottest','highest temperature','past temperature']):
        try:
            d=historical();col='meteo_temperature'
            if 'highest' in q or 'hottest' in q:
                row=d.loc[d[col].idxmax()]
                return f"Highest Open-Meteo temperature in the committed dataset: {row[col]:.1f} °C at {row['time']} (UTC). Gold-derived row ID: {row['row_id']}. This is the maximum in this dataset, not an official all-time record."
            return f"Mean Open-Meteo temperature across {len(d):,} committed hourly records ({d['time'].min()} to {d['time'].max()}): {d[col].mean():.1f} °C. Calculation: arithmetic mean of nonmissing meteo_temperature values. Example supporting row IDs: {', '.join(d['row_id'].head(3))}. For full audit, see the historical dataset."
        except Exception:return 'Historical dataset is unavailable; I cannot verify the requested number.'
    result=answer_guidance(question,api_key=st.secrets.get("GROQ_API_KEY", ""),use_embeddings=st.secrets.get("USE_EMBEDDINGS",False))
    return result["answer"]

st.sidebar.title('🌤️ HeatSafe Pakistan')
st.sidebar.caption('Karachi climate intelligence')
page=st.sidebar.radio('Navigation',['Overview','Weather Trends','Ask HeatSafe AI','Data Quality','About'])
st.sidebar.divider()
st.sidebar.caption('Experimental research prototype · not official weather or medical advice')

if page=='Overview':
    st.title('🌤️ HeatSafe Pakistan')
    st.write('**Karachi weather intelligence and heat-risk decision support**')
    weather_panel()
    st.subheader('Historical temperature trend')
    try:
        d=historical();monthly=d.set_index('time')['meteo_temperature'].resample('MS').mean().reset_index()
        st.plotly_chart(px.line(monthly,x='time',y='meteo_temperature',title='Monthly average temperature (historical Open-Meteo)',labels={'meteo_temperature':'Temperature (°C)','time':'UTC month'}),use_container_width=True)
    except Exception as exc:st.warning(f'Historical chart unavailable: {exc}')
    if st.button('💬 Ask HeatSafe AI →',use_container_width=True):st.session_state['open_ai_hint']=True
    if st.session_state.get('open_ai_hint'):st.info('Select “Ask HeatSafe AI” in the left sidebar to open the integrated assistant.')

elif page=='Weather Trends':
    st.title('📈 Weather Trends')
    try:
        d=historical();st.caption(f"{len(d):,} records · {d['time'].min()} to {d['time'].max()} · timestamps UTC")
        opts={'Temperature':'meteo_temperature','Feels like':'apparent_temperature','Humidity':'meteo_humidity','Wind speed':'meteo_wind_speed'}
        choice=st.selectbox('Measure',list(opts));freq=st.selectbox('Aggregation',['Daily','Monthly'],index=1)
        unit={'Temperature':'°C','Feels like':'°C','Humidity':'%','Wind speed':'km/h'}[choice]
        agg=d.set_index('time')[opts[choice]].resample('D' if freq=='Daily' else 'MS').mean().reset_index()
        st.plotly_chart(px.line(agg,x='time',y=opts[choice],title=f'{freq} average {choice.lower()}',labels={opts[choice]:f'{choice} ({unit})','time':'UTC'}),use_container_width=True)
        st.metric('Historical records',f'{len(d):,}')
        st.caption('Source: committed processed HeatSafe dataset. Charts are descriptive, not forecasts.')
    except Exception as exc:st.error(f'Unable to read historical dataset: {exc}')

elif page=='Ask HeatSafe AI':
    st.title('💬 Ask HeatSafe AI')
    st.caption('Evidence-grounded weather and heat-health assistant. Optional LLM with citations; otherwise transparent retrieval summaries.')
    st.write('Try: **What precautions should outdoor workers take?** · **What is the current weather?** · **What is the highest historical temperature?**')
    if 'messages' not in st.session_state:st.session_state.messages=[]
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):st.markdown(msg['content'])
    if st.button('Clear conversation'):st.session_state.messages=[];st.rerun()
    question=st.chat_input('Ask about Karachi weather, heat safety, or historical data...')
    if question:
        st.session_state.messages.append({'role':'user','content':question})
        with st.chat_message('user'):st.markdown(question)
        response=answer_question(question)
        st.session_state.messages.append({'role':'assistant','content':response})
        with st.chat_message('assistant'):st.markdown(response)

elif page=='Data Quality':
    st.title('🧪 Data Quality & Pipeline')
    try:
        d=historical();flag=d.get('temperature_quality_flag')
        st.metric('Historical records',f'{len(d):,}')
        if flag is not None:
            count=int(flag.astype(str).str.lower().isin(['true','1','yes']).sum())
            st.metric('Flagged temperature disagreements',count)
        st.caption('Source: committed processed weather data. Historical checks are not the same as live validation.')
    except Exception as exc:st.warning(f'Historical quality summary unavailable: {exc}')
    try:
        w,_=live_weather();age,status=freshness(w)
        st.metric('Current weather freshness',status)
        st.write(f"Latest Open-Meteo observation: **{w['time']} Asia/Karachi**")
    except Exception as exc:st.warning(f'Current weather unavailable: {exc}')
    published=read_published_results(st.secrets.get('PUBLISHED_RESULTS_URL',''))
    if published['available']:
        st.success('Published Prefect result is accessible (verify timestamp and status below)')
        st.json(published['prediction'])
        with st.expander('Published quality report'):st.json(published['quality'])
    else:
        st.warning('Prefect published results unavailable: '+published['reason']+'; live weather above is fetched separately.')
    st.caption('Do not interpret this page as proof of successful hourly Prefect execution.')

else:
    st.title('ℹ️ About HeatSafe Pakistan')
    st.markdown('''HeatSafe Pakistan is an experimental climate and heat-risk decision-support prototype focused on Karachi.

**Data:** Open-Meteo and NASA POWER historical hourly weather (2022–September 2026); OpenAQ PM2.5 observations from October 2026. These datasets have different date ranges and are not presented as simultaneous observations.

**Model:** An existing HistGradientBoostingClassifier predicts a proxy heat-risk category using weather features. Categories were derived from apparent-temperature cutoffs, not validated medical outcomes. The rare extreme category had weaker recall in previous project evaluation. Predictions are not official heat warnings or a guarantee of safety.

**AI assistant:** Uses TF-IDF vector retrieval over a small, curated guidance collection, plus live weather and historical dataset queries. An optional API LLM and optional multilingual FAISS embedding search are supported; without configuration it uses TF-IDF retrieval and evidence summaries. Document summaries are paraphrased and linked to sources.

**Pipeline:** Prefect Cloud Serverless schedules the weather flow, but durable result publishing is not connected to this dashboard yet. Live dashboard weather is fetched independently from Open-Meteo.

**Safety:** For official alerts, follow the Pakistan Meteorological Department and relevant local authorities. Seek professional medical advice for health concerns.''')
    st.markdown('**Source links:** [Open-Meteo](https://open-meteo.com/) · [NASA POWER](https://power.larc.nasa.gov/) · [OpenAQ](https://openaq.org/) · [WHO](https://www.who.int/news-room/fact-sheets/detail/climate-change-heat-and-health) · [CDC/NIOSH](https://www.cdc.gov/niosh/heat-stress/about/index.html)')
