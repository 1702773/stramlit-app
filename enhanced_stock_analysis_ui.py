import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from datetime import datetime, timedelta
import json
import numpy as np
import base64
from io import BytesIO

# 設置頁面配置
st.set_page_config(
    page_title="AI 股票趨勢分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂CSS樣式
def load_custom_css(theme="light"):
    """載入自訂CSS樣式"""
    if theme == "dark":
        st.markdown("""
        <style>
        /* 深色模式樣式 */
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stMarkdown {
            color: #fafafa;
        }
        .stMetric {
            background-color: #1e2130;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        /* 載入動畫 */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .loading {
            animation: pulse 1.5s ease-in-out infinite;
        }
        /* 卡片樣式 */
        .info-card {
            background: linear-gradient(135deg, #1e2130 0%, #2a2d3e 100%);
            padding: 20px;
            border-radius: 12px;
            margin: 10px 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            border-left: 4px solid #667eea;
        }
        /* 分隔線動畫 */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
            margin: 20px 0;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        /* 淺色模式樣式 */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .stMetric {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .stMetric:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        /* 載入動畫 */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .loading {
            animation: pulse 1.5s ease-in-out infinite;
        }
        /* 標題動畫 */
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        h1, h2, h3 {
            animation: slideIn 0.6s ease-out;
        }
        /* 卡片樣式 */
        .info-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin: 10px 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border-left: 4px solid #667eea;
            transition: all 0.3s ease;
        }
        .info-card:hover {
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
            transform: translateX(5px);
        }
        /* 分隔線樣式 */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
            margin: 20px 0;
        }
        /* 數據表格樣式 */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

# 配色主題配置
COLOR_THEMES = {
    "專業藍": {
        "bullish": "#26a69a",  # 上漲K線
        "bearish": "#ef5350",  # 下跌K線
        "ma5": "#2196F3",
        "ma10": "#4CAF50",
        "ma20": "#FF9800",
        "ma60": "#9C27B0",
        "volume": "#64B5F6",
        "background": "white"
    },
    "經典黑": {
        "bullish": "#00ff00",
        "bearish": "#ff0000",
        "ma5": "#FFD700",
        "ma10": "#00CED1",
        "ma20": "#FF69B4",
        "ma60": "#9370DB",
        "volume": "#4169E1",
        "background": "#000000"
    },
    "清新綠": {
        "bullish": "#48bb78",
        "bearish": "#f56565",
        "ma5": "#38b2ac",
        "ma10": "#4299e1",
        "ma20": "#ed8936",
        "ma60": "#9f7aea",
        "volume": "#68d391",
        "background": "white"
    },
    "深色模式": {
        "bullish": "#26a69a",
        "bearish": "#ef5350",
        "ma5": "#42a5f5",
        "ma10": "#66bb6a",
        "ma20": "#ffa726",
        "ma60": "#ab47bc",
        "volume": "#7e57c2",
        "background": "#1e1e1e"
    }
}

def get_stock_data(symbol, api_key, start_date, end_date):
    """從Financial Modeling Prep API獲取股票歷史數據"""
    try:
        url = f"https://financialmodelingprep.com/stable/historical-price-eod/full"
        params = {
            'symbol': symbol,
            'apikey': api_key,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not isinstance(data, list) or len(data) == 0:
            st.error(f"無法獲取股票 {symbol} 的數據，請檢查股票代碼是否正確。")
            return None
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"API請求失敗：{str(e)}")
        return None
    except Exception as e:
        st.error(f"數據處理錯誤：{str(e)}")
        return None

def filter_by_date_range(df, start_date, end_date):
    """根據日期範圍過濾數據"""
    if df is None:
        return None
    
    mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
    filtered_df = df.loc[mask].copy()
    
    return filtered_df.reset_index(drop=True)

def get_moving_averages(df):
    """計算移動平均線（MA5, MA10, MA20, MA60）"""
    if df is None or len(df) == 0:
        return None
    
    df = df.copy()
    
    df['MA5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['MA10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['MA60'] = df['close'].rolling(window=60, min_periods=1).mean()
    
    return df

def create_candlestick_chart(df, symbol, color_theme="專業藍", chart_height=700):
    """創建K線圖和移動平均線圖表（支援自訂配色和高度）"""
    theme = COLOR_THEMES[color_theme]
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} 價格與移動平均線', '成交量'),
        row_heights=[0.7, 0.3]
    )
    
    # K線圖
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K線圖',
            increasing_line_color=theme['bullish'],
            decreasing_line_color=theme['bearish'],
            increasing_fillcolor=theme['bullish'],
            decreasing_fillcolor=theme['bearish']
        ),
        row=1, col=1
    )
    
    # 移動平均線
    ma_config = {
        'MA5': {'color': theme['ma5'], 'width': 2},
        'MA10': {'color': theme['ma10'], 'width': 2},
        'MA20': {'color': theme['ma20'], 'width': 2},
        'MA60': {'color': theme['ma60'], 'width': 2}
    }
    
    for ma, config in ma_config.items():
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[ma],
                mode='lines',
                name=ma,
                line=dict(color=config['color'], width=config['width']),
                hovertemplate=f'{ma}: %{{y:.2f}}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 成交量柱狀圖
    colors = [theme['bullish'] if df['close'].iloc[i] >= df['open'].iloc[i] 
              else theme['bearish'] for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='成交量',
            marker_color=colors,
            opacity=0.6,
            hovertemplate='成交量: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 更新佈局
    fig.update_layout(
        title={
            'text': f'<b>{symbol} 股價技術分析圖表</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#333'}
        },
        xaxis_title='日期',
        yaxis_title='價格 (USD)',
        height=chart_height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        template='plotly_white' if theme['background'] == 'white' else 'plotly_dark',
        hovermode='x unified',
        plot_bgcolor=theme['background'],
        paper_bgcolor=theme['background']
    )
    
    # 更新x軸
    fig.update_xaxes(
        rangeslider_visible=False,
        row=1, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)'
    )
    
    # 更新y軸
    fig.update_yaxes(
        title_text="價格 (USD)", 
        row=1, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)'
    )
    fig.update_yaxes(
        title_text="成交量", 
        row=2, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)'
    )
    
    return fig

def generate_ai_insights(symbol, stock_data, openai_api_key, start_date, end_date):
    """使用OpenAI進行技術分析"""
    try:
        client = OpenAI(api_key=openai_api_key)
        
        first_date = stock_data['date'].iloc[0].strftime('%Y-%m-%d')
        last_date = stock_data['date'].iloc[-1].strftime('%Y-%m-%d')
        start_price = stock_data['close'].iloc[0]
        end_price = stock_data['close'].iloc[-1]
        price_change = ((end_price - start_price) / start_price) * 100
        
        data_json = stock_data.to_json(orient='records', date_format='iso')
        
        system_message = """你是一位專業的技術分析師，專精於股票技術分析和歷史數據解讀。你的職責包括：

1. 客觀描述股票價格的歷史走勢和技術指標狀態
2. 解讀歷史市場數據和交易量變化模式
3. 識別技術面的歷史支撐阻力位
4. 提供純教育性的技術分析知識

重要原則：
- 僅提供歷史數據分析和技術指標解讀，絕不提供任何投資建議或預測
- 保持完全客觀中立的分析態度
- 使用專業術語但保持易懂
- 所有分析僅供教育和研究目的
- 強調技術分析的局限性和不確定性
- 使用繁體中文回答

嚴格的表達方式要求：
- 使用「歷史數據顯示」、「技術指標反映」、「過去走勢呈現」等客觀描述
- 避免「可能性」、「預期」、「建議」、「關注」等暗示性用詞
- 禁用「如果...則...」的假設句型，改用「歷史上當...時，曾出現...現象」
- 不提供具體價位的操作參考點，僅描述技術位階的歷史表現
- 強調「歷史表現不代表未來結果」
- 避免任何可能被解讀為操作指引的表達

免責聲明：所提供的分析內容純粹基於歷史數據的技術解讀，僅供教育和研究參考，不構成任何投資建議或未來走勢預測。歷史表現不代表未來結果。"""
        
        user_prompt = f"""請基於以下股票歷史數據進行深度技術分析：

### 基本資訊
- 股票代號：{symbol}
- 分析期間：{first_date} 至 {last_date}
- 期間價格變化：{price_change:.2f}% (從 ${start_price:.2f} 變化到 ${end_price:.2f})

### 完整交易數據
以下是該期間的完整交易數據，包含日期、開盤價、最高價、最低價、收盤價、成交量和移動平均線：
{data_json}

### 分析架構：技術面完整分析

#### 1. 趨勢分析
- 整體趨勢方向（上升、下降、盤整）
- 關鍵支撐位和阻力位識別
- 趨勢強度評估

#### 2. 技術指標分析
- 移動平均線分析（短期與長期MA的關係）
- 價格與移動平均線的相對位置
- 成交量與價格變動的關聯性

#### 3. 價格行為分析
- 重要的價格突破點
- 波動性評估
- 關鍵的轉折點識別

#### 4. 風險評估
- 當前價位的風險等級
- 潛在的支撐和阻力區間
- 市場情緒指標

#### 5. 市場觀察
- 短期技術面觀察（1-2週）
- 中期技術面觀察（1-3個月）
- 關鍵價位觀察點
- 技術面風險因子

### 綜合評估要求
#### 輸出格式要求
- 條理清晰，分段論述
- 提供具體的數據支撐
- 避免過於絕對的預測，強調分析的局限性
- 在適當位置使用表格或重點標記

分析目標：{symbol}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"AI分析失敗：{str(e)}")
        return "AI分析暫時無法使用，請檢查API金鑰或稍後再試。"

def fig_to_png(fig):
    """將Plotly圖表轉換為PNG格式"""
    img_bytes = fig.to_image(format="png", width=1200, height=800)
    return img_bytes

def dataframe_to_excel(df):
    """將DataFrame轉換為Excel格式"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='股票數據')
    return output.getvalue()

def create_download_button(data, filename, label, file_format="png"):
    """創建下載按鈕"""
    if file_format == "png":
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}" style="text-decoration:none;"><button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; padding: 10px 20px; cursor: pointer; font-weight: 600;">{label}</button></a>'
    elif file_format == "excel":
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" style="text-decoration:none;"><button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; padding: 10px 20px; cursor: pointer; font-weight: 600;">{label}</button></a>'
    
    return href

# ========== 主程式開始 ==========

# 側邊欄設置
with st.sidebar:
    st.markdown("## 🔧 分析設定")
    st.divider()
    
    # 主題選擇
    st.markdown("### 🎨 界面主題")
    ui_theme = st.selectbox(
        "選擇界面主題",
        ["淺色模式", "深色模式"],
        help="切換淺色或深色界面主題"
    )
    
    # 圖表配色選擇
    color_theme = st.selectbox(
        "選擇圖表配色",
        ["專業藍", "經典黑", "清新綠", "深色模式"],
        help="選擇K線圖和技術指標的配色方案"
    )
    
    # 圖表高度調整
    chart_height = st.slider(
        "圖表高度",
        min_value=500,
        max_value=1000,
        value=700,
        step=50,
        help="調整圖表顯示高度"
    )
    
    st.divider()
    
    # 股票輸入
    st.markdown("### 📊 股票資訊")
    symbol = st.text_input(
        "股票代碼",
        value="AAPL",
        help="輸入美股股票代碼，例如：AAPL, MSFT, GOOGL, TSLA"
    )
    
    # API金鑰
    st.markdown("### 🔑 API 設定")
    fmp_api_key = st.text_input(
        "FMP API Key",
        type="password",
        help="請輸入您的Financial Modeling Prep API金鑰"
    )
    
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        type="password",
        help="請輸入您的OpenAI API金鑰"
    )
    
    # 日期選擇
    st.markdown("### 📅 時間範圍")
    
    # 快速時間選擇
    time_preset = st.selectbox(
        "快速選擇",
        ["自訂", "最近1個月", "最近3個月", "最近6個月", "最近1年", "最近2年"],
        help="快速選擇常用的時間範圍"
    )
    
    if time_preset == "自訂":
        default_start_date = datetime.now() - timedelta(days=90)
        default_end_date = datetime.now()
    elif time_preset == "最近1個月":
        default_start_date = datetime.now() - timedelta(days=30)
        default_end_date = datetime.now()
    elif time_preset == "最近3個月":
        default_start_date = datetime.now() - timedelta(days=90)
        default_end_date = datetime.now()
    elif time_preset == "最近6個月":
        default_start_date = datetime.now() - timedelta(days=180)
        default_end_date = datetime.now()
    elif time_preset == "最近1年":
        default_start_date = datetime.now() - timedelta(days=365)
        default_end_date = datetime.now()
    else:  # 最近2年
        default_start_date = datetime.now() - timedelta(days=730)
        default_end_date = datetime.now()
    
    start_date = st.date_input(
        "起始日期",
        value=default_start_date,
        help="選擇分析的起始日期"
    )
    
    end_date = st.date_input(
        "結束日期", 
        value=default_end_date,
        help="選擇分析的結束日期"
    )
    
    st.divider()
    
    # 分析按鈕
    analyze_button = st.button("🚀 開始分析", type="primary", use_container_width=True)
    
    st.divider()
    
    # 免責聲明
    with st.expander("📢 免責聲明", expanded=False):
        st.markdown("""
        本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。
        
        請使用者自行判斷投資決策，並承擔相關風險。本系統作者不對任何投資行為負責，亦不承擔任何損失責任。
        """)

# 載入自訂CSS
load_custom_css("dark" if ui_theme == "深色模式" else "light")

# 主標題
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='font-size: 3em; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;'>
        📈 AI 股票趨勢分析系統
    </h1>
    <p style='font-size: 1.2em; color: #666; margin-top: 0;'>專業技術分析 · 智能數據洞察 · 視覺化呈現</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# 主要分析邏輯
if analyze_button:
    # 輸入驗證
    if not symbol.strip():
        st.error("❌ 請輸入股票代碼")
    elif not fmp_api_key.strip():
        st.error("❌ 請輸入FMP API Key")
    elif not openai_api_key.strip():
        st.error("❌ 請輸入OpenAI API Key")
    elif start_date >= end_date:
        st.error("❌ 起始日期不能晚於或等於結束日期")
    else:
        # 開始分析流程
        with st.spinner("🔄 正在獲取股票數據..."):
            stock_data = get_stock_data(symbol.upper(), fmp_api_key, start_date, end_date)
            
            if stock_data is not None and len(stock_data) > 0:
                st.success(f"✅ 成功獲取 {len(stock_data)} 筆交易數據")
                
                # 過濾數據
                filtered_data = filter_by_date_range(stock_data, start_date, end_date)
                
                if filtered_data is not None and len(filtered_data) > 0:
                    # 計算移動平均線
                    with st.spinner("📊 正在計算技術指標..."):
                        data_with_ma = get_moving_averages(filtered_data)
                    
                    if data_with_ma is not None:
                        # 顯示K線圖
                        st.markdown("### 📊 股價K線圖與技術指標")
                        
                        # 創建圖表
                        chart = create_candlestick_chart(
                            data_with_ma, 
                            symbol.upper(), 
                            color_theme,
                            chart_height
                        )
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # 匯出功能區
                        st.markdown("### 💾 匯出選項")
                        col_exp1, col_exp2, col_exp3 = st.columns(3)
                        
                        with col_exp1:
                            # 匯出圖表為PNG
                            try:
                                png_data = fig_to_png(chart)
                                st.markdown(
                                    create_download_button(
                                        png_data,
                                        f"{symbol}_chart.png",
                                        "📷 下載圖表 (PNG)",
                                        "png"
                                    ),
                                    unsafe_allow_html=True
                                )
                            except:
                                st.info("💡 圖表匯出需要安裝 kaleido 套件