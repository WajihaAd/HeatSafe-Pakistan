"""HeatSafe Pakistan | premium Streamlit weather intelligence and grounded assistant."""
from pathlib import Path
import json,re
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from rag_engine import answer_guidance
from data_services import read_published_results

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'processed'/'heatsafe_ml_dataset.csv'
MODEL=ROOT/'models'/'heatsafe_live_improved_model.pkl'
RISK={0:'Low',1:'Moderate',2:'High',3:'Extreme'}
st.set_page_config(page_title='HeatSafe Pakistan | Karachi',page_icon='🌤️',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif!important}
.stApp{background:radial-gradient(ellipse at 70% 0%,#25234b 0%,#0a1225 47%,#08101e 100%);color:#edf4ff}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#171637,#0c1830);border-right:1px solid #343b61}
[data-testid="stSidebar"] *{color:#e6edff!important}
h1,h2,h3{letter-spacing:-.035em!important;color:#f4f6ff!important}
h1{font-size:2.7rem!important} h2{font-size:1.7rem!important}
[data-testid="stMetric"]{background:linear-gradient(135deg,#1a2b4a,#171b3a);border:1px solid #344668;padding:20px;border-radius:20px;box-shadow:0 12px 30px #0002}
[data-testid="stMetricLabel"]{color:#b7c7e3!important}
[data-testid="stMetricValue"]{color:#fff!important;font-weight:800!important}
[data-testid="stChatMessage"]{background:#18233c;border:1px solid #33425c;border-radius:18px;padding:14px;margin:10px 0}
.stButton>button{background:linear-gradient(90deg,#5754d9,#7754c8);color:white;border:0;border-radius:13px;padding:12px 22px;font-weight:700}
.stButton>button:hover{background:#6961e5;color:white;border:0}
.hero{padding:27px 30px;border-radius:25px;background:linear-gradient(110deg,#20254c,#233b5e 65%,#46326a);border:1px solid #4b4e82;margin:10px 0 23px}
.eyebrow{color:#b9b7ff;font-weight:700;letter-spacing:.12em;font-size:.8rem}
.hero-title{font-size:2.4rem;font-weight:800;letter-spacing:-.045em;line-height:1.12;color:white;margin:10px 0}
.hero-desc{color:#d1dafa;max-width:760px}
.insight{background:#182943;border-left:4px solid #b4a2ff;padding:17px 21px;border-radius:13px;margin:14px 0;color:#edf4ff}
.source{font-size:.86rem;color:#aebdd5}
[data-testid="stAlert"]{border-radius:14px}
</style>''',unsafe_allow_html=True)

@st.cache_data(ttl=600,show_spinner=False)
def live_weather():
    r=requests.get('https://api.open-meteo.com/v1/forecast',params={'latitude':24.8607,'longitude':67.0011,'current':'temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,wind_speed_10m,precipitation,weather_code','hourly':'temperature_2m,apparent_temperature,relative_humidity_2m','forecast_days':2,'timezone':'Asia/Karachi'},timeout=18)
    r.raise_for_status();return r.json()
@st.cache_data(show_spinner=False)
def historical():
    d=pd.read_csv(DATA);d['time']=pd.to_datetime(d['time'],utc=True,errors='coerce');d=d.dropna(subset=['time']).sort_values('time')
    d['row_id']='KHI_'+d['time'].dt.strftime('%Y%m%dT%H%M%SZ')
    return d
@st.cache_resource(show_spinner=False)
def model_bundle():
    import joblib
    return joblib.load(MODEL)
def live_status(w):
    t=pd.Timestamp(w['time'])
    t=t.tz_localize('Asia/Karachi') if t.tzinfo is None else t.tz_convert('Asia/Karachi')
    hours=(pd.Timestamp.now(tz='UTC')-t.tz_convert('UTC')).total_seconds()/3600
    return ('PASS' if -.25<=hours<=2 else 'REVIEW_REQUIRED'),hours

def predict(w):
    try:
        status,_=live_status(w)
        if status!='PASS':return None,'Prediction withheld: live observation failed freshness check.'
        bundle=model_bundle();features=bundle['features'];t=pd.Timestamp(w['time']);t=t.tz_localize('Asia/Karachi') if t.tzinfo is None else t.tz_convert('Asia/Karachi')
        vals={'meteo_temperature':w['temperature_2m'],'meteo_humidity':w['relative_humidity_2m'],'dew_point_2m':w['dew_point_2m'],'apparent_temperature':w['apparent_temperature'],'meteo_wind_speed':w['wind_speed_10m'],'hour_karachi':t.hour,'month':t.month}
        label=int(bundle['model'].predict(pd.DataFrame([vals])[features])[0]);return RISK.get(label,'Unknown'),None
    except Exception as exc:return None,f'Model unavailable ({type(exc).__name__}); no risk prediction displayed.'

def get_live():
    try:return live_weather(),None
    except Exception as exc:return None,f'Live weather unavailable: {type(exc).__name__}. Please retry later.'

def hero(kicker,title,description):
    st.markdown(f'<div class="hero"><div class="eyebrow">{kicker}</div><div class="hero-title">{title}</div><div class="hero-desc">{description}</div></div>',unsafe_allow_html=True)

def chart(x,series,title,ylabel):
    fig=go.Figure()
    for i,(name,values) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=x,y=values,name=name,mode='lines',line={'color':['#80b9ff','#ffad78','#9e91ff','#67d7d1'][i%4],'width':2.5},fill='tozeroy' if len(series)==1 else None,fillcolor='rgba(105,100,220,0.12)'))
    fig.update_layout(title={'text':title,'font':{'size':17}},height=340,margin=dict(l=15,r=10,t=50,b=25),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font={'color':'#cdd9ee'},xaxis={'gridcolor':'#263653'},yaxis={'title':ylabel,'gridcolor':'#263653'},legend={'orientation':'h','y':1.13})
    st.plotly_chart(fig,use_container_width=True)

def answer(question):
    q=question.lower().strip()
    weather_words=['weather','temperature','temp','humidity','wind','feels like','hot right now','karachi now','current conditions','mausam','موسم','درجہ حرارت']
    current_words=['current','now','today','right now','at the moment','karachi is','karachi weather','latest','present','abhi','ابھی','آج']
    history_words=['historical','history','highest','hottest','average','mean','past','since 2022','trend','recorded','maximum','minimum','dataset','2022','2023','2024','2025','2026']
    risk_words=['risk prediction','risk level','risk category','predicted risk','heat risk now','current risk','next hour risk']
    if any(s in q for s in risk_words):
        result,error=get_live()
        if error:return error
        w=result['current'];label,err=predict(w)
        if err:return err+' Source: Open-Meteo live weather and HeatSafe experimental model.'
        return f'**Experimental next-hour risk category: {label}.**\n\nObservation: {w["time"]} Pakistan time. This is a model classification based on weather features, not an official heat alert or medical safety determination.\n\n**Evidence:** Open-Meteo live weather API; HeatSafe trained model.'
    if (any(s in q for s in weather_words) and any(s in q for s in current_words)) or re.search(r'(?:temp|temperature|weather|humidity|wind)\s+(?:of|in|at)\s+karachi',q) or re.search(r'karachi\s+(?:temperature|weather|humidity|temp)',q):
        result,error=get_live()
        if error:return error
        w=result['current'];status,age=live_status(w)
        return f'**Karachi weather** (Open-Meteo, {w["time"]} Pakistan time)\n\n🌡️ Temperature: **{w["temperature_2m"]} °C**  \n☀️ Feels like: **{w["apparent_temperature"]} °C**  \n💧 Humidity: **{w["relative_humidity_2m"]}%**  \n🌬️ Wind: **{w["wind_speed_10m"]} km/h**\n\nFreshness check: **{status}** ({age:.1f} hours old). These are API weather estimates, not readings from an on-site sensor.\n\n**Source:** [Open-Meteo](https://open-meteo.com/).'
    if any(s in q for s in history_words) and any(s in q for s in ['temperature','temp','weather','heat','hottest','highest','average','mean','trend','history','historical','dataset']):
        try:
            d=historical();v=d['meteo_temperature'].dropna();row=d.loc[v.idxmax()];start=d['time'].min().date();end=d['time'].max().date()
            if any(s in q for s in ['highest','hottest','maximum']):return f'**Highest temperature in the HeatSafe dataset: {row["meteo_temperature"]:.1f} °C** on {row["time"]} UTC.\n\nSupporting gold-derived row ID: `{row["row_id"]}`. This is a dataset maximum, not Karachi’s official all-time record. Source: Open-Meteo historical data.'
            monthly=d.set_index('time')['meteo_temperature'].resample('MS').mean().dropna();peak=monthly.idxmax()
            return f'Across **{len(d):,} hourly rows** ({start} to {end}), the mean Open-Meteo temperature is **{v.mean():.1f} °C**. The warmest monthly mean is **{monthly.max():.1f} °C** in {peak:%B %Y}.\n\nMethod: arithmetic mean of nonmissing hourly observations; monthly means grouped by UTC month. Example supporting row IDs: `{d.iloc[0]["row_id"]}`, `{d.iloc[len(d)//2]["row_id"]}`, `{d.iloc[-1]["row_id"]}`. The full dataset is required to reproduce the aggregate.'
        except Exception:return 'I cannot load the historical dataset to verify this number.'
    try: key=st.secrets.get('GROQ_API_KEY','')
    except Exception:key=''
    return answer_guidance(question,api_key=key)['answer']

st.sidebar.markdown('## ☀️ HeatSafe')
st.sidebar.caption('PAKISTAN  /  KARACHI')
page=st.sidebar.radio('Explore',['🏠 Overview','📈 Weather Trends','💬 Ask HeatSafe AI','🛡️ Data Quality','ℹ️ About'],label_visibility='collapsed')
st.sidebar.divider()
st.sidebar.caption('Weather observations: Open-Meteo  •  Research prototype')

if page.startswith('🏠'):
    hero('KARACHI · WEATHER INTELLIGENCE','Weather, explained.','Live weather conditions, experimental heat-risk intelligence and evidence-based guidance — in one place.')
    data,error=get_live()
    if error:st.warning(error)
    else:
        w=data['current'];status,age=live_status(w)
        cols=st.columns(4)
        for c,(label,value,unit) in zip(cols,[('🌡️ Temperature',w.get('temperature_2m'),'°C'),('☀️ Feels like',w.get('apparent_temperature'),'°C'),('💧 Humidity',w.get('relative_humidity_2m'),'%'),('🌬️ Wind',w.get('wind_speed_10m'),'km/h')]):c.metric(label,f'{value} {unit}')
        st.caption(f'Open-Meteo • {w["time"]} Pakistan time • Quality: {status} • {age:.1f} hours old')
        label,err=predict(w)
        if err:st.warning(err)
        else:st.info(f'**Experimental next-hour heat-risk category: {label}.** This is not an official warning or a guarantee of safety.')
        if status=='PASS':
            diff=w['apparent_temperature']-w['temperature_2m']
            st.markdown(f'<div class="insight"><b>✦ What the readings mean</b><br>The feels-like temperature is <b>{abs(diff):.1f} °C {"higher" if diff>=0 else "lower"}</b> than the air temperature. Humidity is <b>{w["relative_humidity_2m"]}%</b>. Heat strain depends on exposure, activity, hydration and other factors; these readings alone cannot determine individual safety.</div>',unsafe_allow_html=True)
        hourly=data.get('hourly',{})
        if hourly.get('time'):
            times=pd.to_datetime(hourly['time']);chart(times[:36],{'Air temperature':hourly['temperature_2m'][:36],'Feels like':hourly['apparent_temperature'][:36]},'Next 36 hours · Open-Meteo forecast','°C')
            st.caption('Forecast estimates, not measured future conditions.')
    try:
        d=historical();month=d.set_index('time')['meteo_temperature'].resample('MS').mean().dropna();last=month.iloc[-1];overall=month.mean()
        st.markdown(f'<div class="insight"><b>✦ Historical perspective</b><br>Across the available dataset, the average of monthly mean temperatures is <b>{overall:.1f} °C</b>. The final complete month in the dataset averages <b>{last:.1f} °C</b>. Compare equivalent seasons before inferring climate trends.</div>',unsafe_allow_html=True)
    except Exception:pass
    st.subheader('💬 Ask HeatSafe AI')
    st.write('Ask about Karachi weather, heat-risk predictions, heat precautions or historical records.')
    with st.form('overview_ask'):
        question=st.text_input('Your question',placeholder='What is the temperature in Karachi now?');submitted=st.form_submit_button('Ask HeatSafe AI →')
    if submitted and question:st.markdown(answer(question))
elif page.startswith('📈'):
    hero('HISTORICAL ANALYTICS','Weather Trends','Explore hourly historical weather from January 2022 through September 2026.')
    try:
        d=historical();a,b,c=st.columns(3);a.metric('Hourly observations',f'{len(d):,}');b.metric('Mean temperature',f'{d["meteo_temperature"].mean():.1f} °C');c.metric('Maximum in dataset',f'{d["meteo_temperature"].max():.1f} °C')
        choice=st.selectbox('Explore a measure',{'Temperature':'meteo_temperature','Feels like':'apparent_temperature','Humidity':'meteo_humidity','Wind speed':'meteo_wind_speed'}.keys());mapping={'Temperature':'meteo_temperature','Feels like':'apparent_temperature','Humidity':'meteo_humidity','Wind speed':'meteo_wind_speed'};freq=st.selectbox('Time resolution',['Monthly','Daily']);v=d.set_index('time')[mapping[choice]].resample('MS' if freq=='Monthly' else 'D').mean().dropna()
        chart(v.index,{choice:v.values},f'{freq} average {choice.lower()}','°C' if choice in ('Temperature','Feels like') else '%' if choice=='Humidity' else 'km/h')
        peak=v.idxmax();st.markdown(f'<div class="insight"><b>✦ Data insight</b><br>Highest {freq.lower()} mean in this view: <b>{v.max():.1f}</b> during <b>{peak:%B %Y}</b>. Lowest: <b>{v.min():.1f}</b>. These are descriptive statistics from the committed dataset, not a prediction.</div>',unsafe_allow_html=True)
        st.caption('Source: HeatSafe processed dataset from Open-Meteo / NASA POWER. Chart series uses the selected Open-Meteo measure.')
    except Exception as exc:st.error(f'Historical data could not be loaded: {exc}')
elif page.startswith('💬'):
    hero('EVIDENCE-BASED ASSISTANT','Ask HeatSafe AI','Ask about real Karachi weather, historical measurements and trusted heat-health guidance. Answers show evidence and limitations.')
    if 'messages' not in st.session_state:st.session_state.messages=[]
    samples=['What is the temperature in Karachi now?','What is the current heat risk prediction?','What precautions should outdoor workers take?','What is the highest historical temperature?']
    st.caption('Suggested questions')
    cols=st.columns(2)
    chosen=None
    for i,q in enumerate(samples):
        if cols[i%2].button(q,key=f'sample{i}',use_container_width=True):chosen=q
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):st.markdown(msg['content'])
    q=st.chat_input('Ask about Karachi weather or heat safety...') or chosen
    if q:
        st.session_state.messages.append({'role':'user','content':q})
        with st.chat_message('user'):st.markdown(q)
        with st.chat_message('assistant'):
            with st.spinner('Checking data and trusted sources...'):reply=answer(q)
            st.markdown(reply)
        st.session_state.messages.append({'role':'assistant','content':reply})
    if st.button('Clear conversation'):st.session_state.messages=[];st.rerun()
    st.caption('Without an optional language-model API key, document answers are retrieved evidence summaries, not generative AI responses.')
elif page.startswith('🛡️'):
    hero('TRANSPARENCY','Data Quality & Pipeline','See what is validated, what is stale, and what has not been connected yet.')
    try:
        d=historical();a,b=st.columns(2);a.metric('Historical hourly rows',f'{len(d):,}');b.metric('Flagged temperature disagreements',int(pd.to_numeric(d['temperature_quality_flag'],errors='coerce').fillna(0).sum()))
        st.caption('Historical flags are from the committed processed dataset; they are not live quality checks.')
    except Exception as exc:st.warning(f'Historical quality unavailable: {exc}')
    data,error=get_live()
    if error:st.warning(error)
    else:
        status,age=live_status(data['current']);st.metric('Live observation freshness',status);st.caption(f'Observed {data["current"]["time"]} Pakistan time • age {age:.1f} h')
    try:url=st.secrets.get('PUBLISHED_RESULTS_URL','')
    except Exception:url=''
    pub=read_published_results(url)
    if pub['available']:
        st.success('Persistent published Prefect results are available');st.json(pub['prediction'])
        with st.expander('Published quality report'):st.json(pub['quality'])
    else:st.warning('Prefect-to-dashboard publishing not connected: '+pub['reason'])
    st.caption('A successful manual Prefect run does not prove scheduled hourly runs or durable publishing.')
else:
    hero('PROJECT INFORMATION','About HeatSafe Pakistan','A transparent research prototype that turns climate and weather data into understandable insights.')
    st.markdown('''### Data sources
- **Open-Meteo:** historical weather and independent current weather API.
- **NASA POWER:** historical climate/weather cross-checks.
- **OpenAQ:** October 2026 PM2.5 measurements (a different date range; not merged as concurrent observations).

### Model and limitations
The trained HistGradientBoostingClassifier predicts a **proxy next-hour heat-risk category** based on weather features. Risk labels were derived from apparent-temperature thresholds rather than health outcomes. Rare extreme-risk cases had weaker recall. This is not a validated medical or official meteorological alert.

### AI assistant
Current weather answers come from Open-Meteo; historical numeric answers come from the processed dataset; guidance answers come from a small source-linked document collection. The knowledge base is not yet a complete verified corpus, and generative AI is optional.

### Automation
Prefect Cloud Serverless is scheduled hourly. Durable publishing to this dashboard is **not yet configured**. This dashboard's current observations are fetched independently.

**Trusted links:** [Open-Meteo](https://open-meteo.com/) · [NASA POWER](https://power.larc.nasa.gov/) · [OpenAQ](https://openaq.org/) · [WHO](https://www.who.int/news-room/fact-sheets/detail/climate-change-heat-and-health) · [CDC/NIOSH](https://www.cdc.gov/niosh/heat-stress/about/index.html)''')
