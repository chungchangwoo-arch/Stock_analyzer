import streamlit as st
import requests
from groq import Groq
import pandas as pd
import FinanceDataReader as fdr
import plotly.graph_objects as go
import plotly.express as px 
from plotly.subplots import make_subplots
import re
import html
import json
import time
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import numpy as np

# --------------------------------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------------------------------
st.set_page_config(page_title="AlphaView: Deep Dive", page_icon="⚡", layout="wide")

try:
    NAVER_ID = st.secrets["naver"]["client_id"]
    NAVER_SECRET = st.secrets["naver"]["client_secret"]
    GROQ_KEY = st.secrets["groq"]["api_key"]
except:
    st.error("🚨 secrets.toml 설정이 필요합니다.")
    st.stop()

client = Groq(api_key=GROQ_KEY)

# --------------------------------------------------------------------------
# 2. 유틸리티 & 데이터 수집
# --------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def get_krx_code_map():
    try:
        df = fdr.StockListing('KRX')
        return dict(zip(df['Name'], df['Code']))
    except:
        return {}

def find_ticker(name, code_map):
    if name in code_map: return code_map[name]
    if name.isdigit() and len(name) == 6: return name
    return name.upper()

def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def summarize_title(text, max_length=60):
    """
    긴 기사 제목을 간단히 요약
    - 괄호 제거 (기사 출처 등)
    - 길이 초과 시 핵심만 추출
    """
    # 괄호 안 내용 제거 (예: [기사출처], (분석), 등)
    text = re.sub(r'[\(\[].*?[\)\]]', '', text).strip()
    
    # 길이가 길면 '-', '|' 등으로 첫 번째 절만 추출
    if len(text) > max_length:
        for delimiter in [' - ', ' | ', ' / ', '...']:
            if delimiter in text:
                text = text.split(delimiter)[0].strip()
                break
    
    # 여전히 길면 max_length로 자르고 '...' 추가
    if len(text) > max_length:
        text = text[:max_length] + '...'
    
    return text

@st.cache_data(ttl=600)
def get_stock_data(ticker, start_date, end_date):
    try:
        df = fdr.DataReader(ticker, start_date, end_date)
        return df
    except:
        return None

@st.cache_data(ttl=3600)
def get_naver_datalab_trend(keyword, start_date, end_date):
    url = "https://openapi.naver.com/v1/datalab/search"
    headers = {
        "X-Naver-Client-Id": NAVER_ID,
        "X-Naver-Client-Secret": NAVER_SECRET,
        "Content-Type": "application/json"
    }
    body = {
        "startDate": start_date.strftime("%Y-%m-%d"),
        "endDate": end_date.strftime("%Y-%m-%d"),
        "timeUnit": "date",
        "keywordGroups": [{"groupName": keyword, "keywords": [keyword]}]
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(body))
        if response.status_code == 200:
            result = response.json()
            if not result['results']: return pd.DataFrame()
            data_list = result['results'][0]['data']
            df_trend = pd.DataFrame(data_list)
            df_trend['period'] = pd.to_datetime(df_trend['period'])
            df_trend.set_index('period', inplace=True)
            df_trend.rename(columns={'ratio': 'search_volume'}, inplace=True)
            return df_trend
        else: return pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=600)
def get_naver_news_content(keyword, start_date, end_date):
    """
    뉴스 검색 (keyword 정확도 필터만 적용)
    과거 데이터의 뉴스 부족 문제 해결을 위해 재무 키워드 필터 제거
    """
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": NAVER_ID, "X-Naver-Client-Secret": NAVER_SECRET}
    all_items = []
    curr = start_date.replace(day=1)
    end = end_date.replace(day=1)
    
    # 정규식: 정확한 단어 경계로 keyword 포함 필터
    import re as re_module
    keyword_pattern = re_module.compile(rf'\b{re_module.escape(keyword)}\b')
    
    while curr <= end:
        year_month = curr.strftime("%Y년 %m월")
        query = f"{keyword} {year_month}"
        # display 100으로 증가, sort='sim'(유사도)로 변경해 관련성 높은 것 우선
        params = {"query": query, "display": 100, "sort": "sim"}
        try:
            res = requests.get(url, headers=headers, params=params, timeout=5)
            if res.status_code == 200:
                items = res.json().get('items', [])
                for item in items:
                    try:
                        pub_date = datetime.strptime(item['pubDate'], "%a, %d %b %Y %H:%M:%S %z").replace(tzinfo=None)
                        if start_date <= pub_date.date() <= end_date:
                            title = clean_text(item.get('title', ''))
                            # 필터: 정확한 keyword 포함 여부만 체크
                            if not keyword_pattern.search(title):
                                continue
                            item['clean_date'] = pub_date
                            item['date_str'] = pub_date.strftime("%Y-%m-%d")
                            item['summary_title'] = summarize_title(title)  # 요약된 제목도 저장
                            all_items.append(item)
                    except: continue
        except: pass
        curr += relativedelta(months=1)
        time.sleep(0.1)
    
    all_items.sort(key=lambda x: x['clean_date'])
    # 중복제거
    unique = []
    seen = set()
    for i in all_items:
        if i['title'] not in seen:
            unique.append(i)
            seen.add(i['title'])
    return unique

# --------------------------------------------------------------------------
# 3. AI 분석 함수들 (일반 분석 + [NEW] 급등락 분석)
# --------------------------------------------------------------------------

def analyze_general_trend(keyword, trend_df, news_items):
    """전체 기간 트렌드 분석"""
    max_date = trend_df['search_volume'].idxmax().strftime("%Y-%m-%d") if not trend_df.empty else "N/A"
    step = max(1, len(news_items) // 15)
    context = "\n".join([f"- [{i['date_str']}] {clean_text(i['title'])}" for i in news_items[::step][:15]])

    prompt = f"""
    Target: {keyword}, Peak Date: {max_date}
    News: {context}
    Output JSON: {{ "summary": "Period summary", "peak_reason": "Reason for peak interest", "sentiment": "Sentiment" }}
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"system","content":"Analyze financial trend. JSON only."},{"role":"user","content":prompt}],
            temperature=0.1, response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except: return None

def analyze_top_volatility(keyword, events_df, news_items):
    """
    [NEW] 이벤트(기간) 단위로 급등락의 원인을 분석 (뉴스 기반)
    events_df: DataFrame with columns ['Start','End','PeakDate','PeakChange','PeakChangeAbs','Trigger','NewsList']
    뉴스가 충분한 이벤트만 분석
    """
    if events_df is None or events_df.empty:
        return None

    # 뉴스가 최소 2개 이상인 이벤트만 필터링
    events_with_news = events_df[events_df['NewsList'].apply(lambda x: len(x) >= 2 if isinstance(x, list) else False)]
    
    if events_with_news.empty:
        # 뉴스가 없으면 분석 불가
        return {"events": [], "note": "뉴스 데이터 부족"}

    events_context = ""
    for _, ev in events_with_news.iterrows():
        start = ev['Start'].strftime("%Y-%m-%d")
        end = ev['End'].strftime("%Y-%m-%d")
        peak = ev['PeakDate'].strftime("%Y-%m-%d")
        change = ev['PeakChange']
        trigger = ev.get('Trigger', '')

        # 뉴스 포함 (이미 필터링되어 있음)
        related_news = ev.get('NewsList', [])
        if related_news:
            news_str = " | ".join([f"[{n.get('date', '')}] {n.get('title', '')}" for n in related_news[:5]])
        else:
            news_str = "(뉴스 없음)"
        
        events_context += f"【이벤트】\n기간: {start} ~ {end}\n피크: {peak}\n변동폭: {change:.2f}%\n뉴스: {news_str}\n\n"

    prompt = f"""
    당신은 금융 분석 전문가입니다. 다음 주식 급등락 이벤트들을 뉴스 기반으로 분석하세요. 
    뉴스에서 명확한 원인을 찾아 그것을 중심으로 분석하고, 뉴스와 직접적인 연관이 없는 추측은 하지 마세요.

    분석 가이드:
    1. 주어진 뉴스 내용을 기반으로만 분석
    2. 뉴스에 명시된 사건/공시/결과 등을 직접 인용
    3. 뉴스 없이 추측하거나 지어내지 말 것
    4. 분석이 불가능한 경우 "뉴스 기반 분석 불가능" 명시
    5. 해당 기간의 주가 움직임에 대한 정성적 평가 추가 (호재/악재, 기대감, 조정 등)

    이벤트 데이터 (기간, 피크 변동폭, 관련 뉴스):
    {events_context}

    응답 형식 (JSON으로만 응답):
    {{
      "events": [
        {{
          "date_range": "YYYY-MM-DD ~ YYYY-MM-DD",
          "peak": "YYYY-MM-DD",
          "change": "+10.5%",
          "reason": "뉴스 기반 원인 분석 (2-3문장)",
          "news_summary": "주요 뉴스 요약 (1문장)",
          "sentiment": "긍정적/부정적/중립적 및 시장 평가"
        }}
      ]
    }}
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"system","content":"You are a financial expert. Analyze stock volatility ONLY based on provided news. Do not speculate. Return ONLY valid JSON."},{"role":"user","content":prompt}],
            temperature=0.1, response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except:
        return {"events": [], "note": "분석 실패"}

# --------------------------------------------------------------------------
# 4. 메인 UI
# --------------------------------------------------------------------------

krx_map = get_krx_code_map()

with st.sidebar:
    st.header("⚡ AlphaView: Deep Dive")
    name_input = st.text_input("종목명", "카카오")
    ticker = find_ticker(name_input, krx_map)
    st.divider()
    s_date = st.date_input("시작일", date(2024, 11, 19))
    e_date = st.date_input("종료일", date(2025, 11, 19))
    run_btn = st.button("심층 분석 시작 🚀", type="primary")

st.title(f"AlphaView: {name_input} ({ticker})")

if run_btn:
    with st.status("데이터 수집 및 정밀 분석 중...", expanded=True) as status:
        st.write("1. 주가 & 트렌드 데이터 수집...")
        df_stock = get_stock_data(ticker, s_date, e_date)
        df_trend = get_naver_datalab_trend(name_input, s_date, e_date)
        
        st.write("2. 뉴스 아카이브 검색...")
        news_items = get_naver_news_content(name_input, s_date, e_date)
        
        # 데이터 병합 및 변동성 계산
        top_events_df = pd.DataFrame()
        merged_df = pd.DataFrame()
        if df_stock is not None and not df_trend.empty:
            aligned_trend = df_trend.reindex(df_stock.index).fillna(0)
            df_stock['DailyReturn'] = df_stock['Close'].pct_change() * 100
            df_stock['AbsReturn'] = df_stock['DailyReturn'].abs()

            merged_df = pd.DataFrame({
                'Close': df_stock['Close'],
                'DailyReturn': df_stock['DailyReturn'],
                'AbsReturn': df_stock['AbsReturn'],
                'SearchVolume': aligned_trend['search_volume'],
                'Volume': df_stock.get('Volume', pd.Series(index=df_stock.index))
            })

            # 보간/결측 처리 (필요시)
            merged_df = merged_df.dropna(subset=['Close'])

            # 롤링 기반 z-score(이상치 탐지)로 이벤트(기간) 추출
            # 데이터 크기에 맞춰 rolling window 동적 조정
            n_data = len(merged_df)
            if n_data < 30:
                window_size = max(5, n_data // 6)
                min_periods = max(3, window_size // 2)
            else:
                window_size = 30
                min_periods = 10
            
            roll = merged_df[['DailyReturn', 'SearchVolume', 'Volume']].rolling(window=window_size, min_periods=min_periods)
            mean = roll.mean()
            std = roll.std().replace(0, np.nan)

            merged_df['ret_z'] = (merged_df['DailyReturn'] - mean['DailyReturn']) / std['DailyReturn']
            merged_df['search_z'] = (merged_df['SearchVolume'] - mean['SearchVolume']) / std['SearchVolume']
            merged_df['vol_z'] = (merged_df['Volume'] - mean['Volume']) / std['Volume']

            # 이벤트 플래그: 수익률 또는 검색량 또는 거래량에서 유의한 이상치
            # z-score 임계값도 데이터량에 맞춰 조정
            z_threshold = 1.8 if n_data < 100 else 2.0
            merged_df['evt_flag'] = (merged_df['ret_z'].abs() > z_threshold) | (merged_df['search_z'] > z_threshold) | (merged_df['vol_z'] > z_threshold)

            # 연속된 True 구간을 이벤트로 묶기
            groups = []
            in_group = False
            start = None
            end = None
            for idx, row in merged_df.iterrows():
                if row['evt_flag'] and not in_group:
                    in_group = True
                    start = idx
                    end = idx
                elif row['evt_flag'] and in_group:
                    end = idx
                elif (not row['evt_flag']) and in_group:
                    groups.append((start, end))
                    in_group = False
            if in_group and start is not None:
                groups.append((start, end))

            events = []
            for s, e in groups:
                window = merged_df.loc[s:e]
                if window.empty: continue
                
                event_duration = (e.date() - s.date()).days + 1
                if event_duration < 1:
                    continue
                
                # 피크 날짜: 절대 변동폭 최대값 기준
                peak_idx = window['AbsReturn'].idxmax()
                peak_change = window.loc[peak_idx, 'DailyReturn'] if not pd.isna(peak_idx) else 0.0
                peak_abs = abs(peak_change)
                
                # 필터: 변동폭 최소값 (너무 작으면 의미 없음)
                if peak_abs < 1.0:  # 최소 1% 이상 변동
                    continue
                
                # 트리거 힌트: 어떤 지표가 가장 크었는지
                trig_vals = {
                    'return': window['ret_z'].abs().max(),
                    'search': window['search_z'].max(),
                    'volume': window['vol_z'].max()
                }
                trigger = max(trig_vals, key=lambda k: -np.nan_to_num(-trig_vals[k]))
                
                # 【NEW】 의미성 점수 계산 (기간 + 변동폭 + 평균 변동성)
                # 단일 일의 큰 변동 vs 장기 추세 변화를 모두 고려
                avg_abs_return = window['AbsReturn'].mean()
                cumulative_return = window['DailyReturn'].sum()
                
                # 점수 구성:
                # 1. 피크 단일 변동폭 (단기 급등락)
                peak_score = peak_abs
                
                # 2. 기간 평균 변동성 (추세의 일관성)
                duration_factor = min(event_duration / 30, 1.0)  # 최대 30일 기준
                trend_score = avg_abs_return * duration_factor * 2
                
                # 3. 누적 변동폭 (장기 추세)
                cumulative_score = abs(cumulative_return) * 0.5
                
                # 종합 의미성 점수 (가중 평균)
                significance_score = (peak_score * 0.5 + trend_score * 0.3 + cumulative_score * 0.2)
                
                # 수집된 뉴스(구간 ±2일로 확대, 요약 제목 포함)
                related_news = []
                for item in news_items:
                    try:
                        nd = item['clean_date'].date()
                        if (nd >= s.date() - timedelta(days=2)) and (nd <= e.date() + timedelta(days=2)):
                            related_news.append({
                                'date': item.get('date_str'), 
                                'title': clean_text(item.get('title')),
                                'summary_title': item.get('summary_title', clean_text(item.get('title'))),  # 요약 제목
                                'link': item.get('link')
                            })
                    except:
                        continue

                events.append({
                    'Start': s,
                    'End': e,
                    'PeakDate': peak_idx,
                    'PeakChange': peak_change,
                    'PeakChangeAbs': peak_abs,
                    'Duration': event_duration,
                    'AvgReturn': avg_abs_return,
                    'CumulativeReturn': cumulative_return,
                    'SignificanceScore': significance_score,
                    'Trigger': trigger,
                    'NewsList': related_news
                })

            if events:
                top_events_df = pd.DataFrame(events).sort_values('SignificanceScore', ascending=False)
                
                # 뉴스 보유 여부로 필터링: 최소 2개 이상의 뉴스를 가진 이벤트만 선택
                events_with_news = top_events_df[top_events_df['NewsList'].apply(len) >= 2]
                
                if not events_with_news.empty:
                    # 뉴스 충분한 이벤트 중 상위 3개 선택 (의미성 점수순)
                    top_events_df = events_with_news.head(3)
                    st.write(f"✅ {len(events)}개의 이벤트 탐지, 뉴스 기반 분석 가능한 TOP3 선별 (의미성 점수순)")
                else:
                    # 뉴스 부족 시 모든 이벤트를 의미성 점수 기준으로 정렬하되, 분석 시 경고
                    top_events_df = top_events_df.head(3)
                    st.warning(f"⚠️ {len(events)}개 이벤트 탐지되었으나, 관련 뉴스 부족으로 시장 데이터 기반 분석입니다.")

        st.write("3. AI 종합 분석 (트렌드 + 급등락 원인)...")
        ai_general = analyze_general_trend(name_input, df_trend, news_items)
        ai_volatility = analyze_top_volatility(name_input, top_events_df if not top_events_df.empty else pd.DataFrame(), news_items)
        
        if top_events_df.empty:
            st.warning("⚠️ 탐지된 급등락 이벤트가 없습니다. 기간을 변경해보세요.")

        status.update(label="분석 완료!", state="complete", expanded=False)

    # ------------------------------------------------
    # [시각화] 시계열 차트 (선 + 선)
    # ------------------------------------------------
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 주가 vs 대중 관심도")
        
        if not merged_df.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 1. 검색량 (얇은 선, 배경에 깔림 — 가격을 가리지 않도록 아래에 그림)
            fig.add_trace(
                go.Scatter(
                    x=merged_df.index,
                    y=merged_df['SearchVolume'],
                    name="검색량(Trend)",
                    mode='lines',
                    line=dict(color='#ff7f0e', width=1.6, dash='dot'),
                    opacity=0.55,
                ),
                secondary_y=True
            )

            # 2. 주가 (진한 선, 위에 그림)
            fig.add_trace(
                go.Scatter(x=merged_df.index, y=merged_df['Close'], name="주가",
                           line=dict(color='#1f77b4', width=3)),
                secondary_y=False
            )
            
            # Top 3 이벤트만 시각적으로 구간 표시 (vrect + 번호 라벨)
            if not top_events_df.empty:
                top3 = top_events_df.head(3)
                colors = ['rgba(255, 0, 0, 0.15)', 'rgba(255, 165, 0, 0.15)', 'rgba(0, 0, 255, 0.15)']  # 빨강, 주황, 파랑
                for rank, (_, ev) in enumerate(top3.iterrows()):
                    try:
                        # 구간 표시 (vrect)
                        fig.add_vrect(
                            x0=ev['Start'], x1=ev['End'],
                            fillcolor=colors[rank], opacity=1.0,
                            layer='below', line_width=2,
                            annotation_text=f"#{rank+1}",
                            annotation_position='top left',
                            annotation_font=dict(size=14, color='black')
                        )
                    except:
                        continue

            fig.update_yaxes(title_text="주가", secondary_y=False)
            fig.update_yaxes(title_text="트렌드 지수", secondary_y=True, showgrid=False) # 오른쪽 그리드 제거
            fig.update_layout(hovermode="x unified", height=450)
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption("※ 주황색 점선은 검색 관심도입니다. 색상 구간(#1,#2,#3)은 상위 3대 급등락 기간입니다.")

    # ------------------------------------------------
    # [AI 리포트] 일반 요약 + 급등락 원인
    # ------------------------------------------------
    with col2:
        st.subheader("🤖 AI 심층 리포트")
        
        # 1. 일반 요약
        if ai_general:
            with st.expander("📌 전체 흐름 요약", expanded=True):
                st.info(ai_general.get('summary'))
                st.write(f"**심리:** {ai_general.get('sentiment')}")
        
        # 2. 급등락 원인 분석 (Top 3 뉴스 기반) + 관련 뉴스 + 이벤트 특성
        st.markdown("#### 🚨 급등락 원인 분석 (뉴스 기반 TOP3)")
        if ai_volatility and isinstance(ai_volatility, dict):
            # 뉴스 부족 경고
            if ai_volatility.get('note'):
                st.warning(f"⚠️ {ai_volatility.get('note')} - 분석이 제한됩니다.")
            
            events = ai_volatility.get('events', [])
            if events:
                for i, event in enumerate(events[:3]):
                    date_range = event.get('date_range') or event.get('date') or event.get('peak') or 'N/A'
                    change = event.get('change') or ''
                    reason = event.get('reason') or event.get('reasoning') or event.get('detail') or ''
                    news_summary = event.get('news_summary', '')
                    sentiment = event.get('sentiment', '')

                    icon = ""
                    try:
                        if isinstance(change, str) and "+" in change:
                            icon = "📈"
                        elif isinstance(change, str) and "-" in change:
                            icon = "📉"
                        else:
                            chv = float(str(change).replace('%', ''))
                            icon = "📈" if chv > 0 else "📉"
                    except:
                        icon = ""

                    with st.container(border=True):
                        st.markdown(f"**#{i+1}. {date_range} {icon} {change}**")
                        
                        # 이벤트 특성 정보 (기간, 점수 등)
                        if not top_events_df.empty and i < len(top_events_df):
                            ev = top_events_df.iloc[i]
                            duration = ev.get('Duration', 0)
                            significance = ev.get('SignificanceScore', 0)
                            avg_return = ev.get('AvgReturn', 0)
                            cumulative = ev.get('CumulativeReturn', 0)
                            
                            # 이벤트 타입 판단
                            if duration <= 3:
                                event_type = "🔴 단기 급등락"
                            elif duration <= 14:
                                event_type = "🟠 중기 추세 변화"
                            else:
                                event_type = "🟡 장기 추세 변화"
                            
                            st.caption(f"{event_type} | 기간: {duration}일 | 일평균 변동: {avg_return:.2f}% | 누적: {cumulative:.2f}%")
                        
                        # 뉴스 요약 (AI가 제공한 핵심 요약)
                        if news_summary:
                            st.info(f"📰 핵심 뉴스: {news_summary}")
                        
                        # 원인 분석
                        if reason:
                            st.write(f"**분석:** {reason}")
                        
                        # 시장 평가 및 감정
                        if sentiment:
                            st.write(f"**평가:** {sentiment}")
                        
                        # 참고: 관련 뉴스 제목들 (짧은 버전)
                        if not top_events_df.empty and i < len(top_events_df):
                            ev = top_events_df.iloc[i]
                            if ev.get('NewsList') and len(ev.get('NewsList', [])) > 0:
                                st.caption("📑 관련 뉴스 제목:")
                                for n in ev['NewsList'][:3]:
                                    # 요약된 제목 우선 표시
                                    display_title = n.get('summary_title') or n.get('title')
                                    st.caption(f"• {display_title}")
            else:
                st.info("뉴스 기반 분석이 가능한 이벤트가 없습니다. 다른 기간을 시도해보세요.")

    # ------------------------------------------------
    # [상관관계] 탭으로 분리
    # ------------------------------------------------
    st.divider()
    st.subheader("🔗 상관관계 심화 분석")
    
    corr_vol = merged_df[['SearchVolume', 'AbsReturn']].corr().iloc[0, 1]
    
    c1, c2 = st.columns(2)
    with c1:
        fig_vol = px.scatter(
            merged_df, x='SearchVolume', y='AbsReturn',
            title=f"검색량 vs 주가 변동폭 (R={corr_vol:.2f})",
            labels={'SearchVolume': '검색량 (0-100)', 'AbsReturn': '변동폭 (|%|)'},
            trendline='ols', color_discrete_sequence=['#d62728']
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with c2:
        st.markdown("#### 💡 통계적 해석")
        st.write(f"현재 상관계수 **R = {corr_vol:.2f}** 입니다.")
        if corr_vol > 0.3:
            st.success("대중의 관심이 높아질수록 **주가가 크게 요동치는(변동성 확대)** 경향이 뚜렷합니다.")
        elif corr_vol > 0.1:
            st.info("관심이 높아지면 변동성이 **약간 커지는** 경향이 있습니다.")
        else:
            st.warning("검색량과 변동성 사이에 뚜렷한 관계가 발견되지 않았습니다.")

    # 뉴스 타임라인
    if news_items:
        with st.expander("🗞️ 전체 뉴스 타임라인 보기"):
            for item in news_items:
                st.write(f"**{item['date_str']}** | [{clean_text(item['title'])}]({item['link']})")




def plot_correlation_heatmap(portfolio_data):
    """종목간 상관관계 히트맵"""
    try:
        if not portfolio_data:
            return None
        
        if "stock_data" not in portfolio_data:
            return None
        
        stock_data_dict = portfolio_data.get("stock_data", {})
        
        if not stock_data_dict or len(stock_data_dict) < 2:
            return None
        
        # 각 종목의 종가 데이터 추출
        price_data = {}
        for ticker, data in stock_data_dict.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                if 'Close' in data.columns:
                    price_data[ticker] = data['Close']
        
        if len(price_data) < 2:
            return None
        
        # DataFrame으로 통합
        prices_df = pd.DataFrame(price_data)
        prices_df = prices_df.dropna()
        
        if prices_df.empty or prices_df.shape[0] < 2:
            return None
        
        # 수익률로 변환
        returns_df = prices_df.pct_change().dropna()
        
        if returns_df.empty or returns_df.shape[0] < 2 or returns_df.shape[1] < 2:
            return None
        
        # 상관계수 계산
        try:
            corr_matrix = returns_df.corr()
        except:
            return None
        
        if corr_matrix is None or corr_matrix.empty:
            return None
        
        # 히트맵 생성
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.columns),
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 11},
            colorbar=dict(title="상관계수"),
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title='📊 종목간 상관관계 (분산 효과 분석)',
            height=450,
            xaxis_title='종목',
            yaxis_title='종목',
            template='plotly_white',
            hovermode='closest'
        )
        
        return fig
    except Exception as e:
        return None
