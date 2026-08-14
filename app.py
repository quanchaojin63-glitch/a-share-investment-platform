import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="A股价值投资终端 V2.1",
    page_icon="📈",
    layout="wide",
)

# -----------------------------
# Demo universe / fallback data
# -----------------------------
DEMO = pd.DataFrame([
    ["601138", "工业富联", 46.50, 0.82, 3.75, 25.0, 20.0, 0.20, 0.18, 0.12],
    ["300502", "新易盛", 145.00, 4.20, 6.10, 32.0, 25.0, 0.25, 0.22, 0.15],
    ["300308", "中际旭创", 205.00, 5.90, 8.20, 30.0, 24.0, 0.22, 0.20, 0.14],
    ["GOOG", "谷歌", 190.00, 8.20, 10.20, 24.0, 21.0, 0.18, 0.15, 0.12],
    ["601899", "紫金矿业", 27.80, 1.10, 1.45, 18.0, 16.0, 0.12, 0.10, 0.08],
], columns=[
    "code", "name", "price", "eps", "eps_next",
    "fair_pe", "pe_next", "g26", "g27", "terminal_g"
])

# -----------------------------
# Helpers
# -----------------------------
def fmt_num(x, digits=2):
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):,.{digits}f}"
    except Exception:
        return "-"

def safe_float(x, default=np.nan):
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return default
        return float(x)
    except Exception:
        return default

def normalize_code(x):
    s = str(x).strip()
    if s.isdigit():
        return s.zfill(6)
    return s.upper()

@st.cache_data(ttl=300, show_spinner=False)
def load_spot():
    """Try AkShare realtime A-share quotes. Return a normalized dataframe."""
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return pd.DataFrame(), "DEMO"
        rename = {
            "代码": "code",
            "名称": "name",
            "最新价": "price",
            "涨跌幅": "change_pct",
            "市盈率-动态": "pe",
            "市净率": "pb",
            "总市值": "market_cap",
            "成交额": "turnover",
        }
        df = df.rename(columns=rename)
        needed = ["code", "name", "price"]
        if not all(c in df.columns for c in needed):
            return pd.DataFrame(), "DEMO"
        for c in ["price", "change_pct", "pe", "pb", "market_cap", "turnover"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["code"] = df["code"].map(normalize_code)
        return df, "LIVE"
    except Exception:
        return pd.DataFrame(), "DEMO"

@st.cache_data(ttl=600, show_spinner=False)
def load_history(code, days=180):
    """Try daily history from AkShare; return normalized dataframe."""
    try:
        import akshare as ak
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(
            symbol=normalize_code(code),
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
        })
        for c in ["open", "close", "high", "low", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date", "close"]).sort_values("date")
    except Exception:
        return pd.DataFrame()

def demo_history(price):
    dates = pd.date_range(end=datetime.now(), periods=120, freq="B")
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.018, len(dates))
    series = price * np.cumprod(1 + returns)
    return pd.DataFrame({"date": dates, "close": series})

def get_stock_row(code, live_df):
    code = normalize_code(code)
    if live_df is not None and not live_df.empty:
        hit = live_df[live_df["code"].astype(str) == code]
        if not hit.empty:
            r = hit.iloc[0].to_dict()
            demo = DEMO[DEMO.code == code]
            if not demo.empty:
                d = demo.iloc[0]
                for k in ["eps", "eps_next", "fair_pe", "pe_next", "g26", "g27", "terminal_g"]:
                    r.setdefault(k, d[k])
            return r, "LIVE"
    hit = DEMO[DEMO.code == code]
    if not hit.empty:
        return hit.iloc[0].to_dict(), "DEMO"
    return None, "DEMO"

def valuation(price, eps, eps_next, fair_pe, growth26, growth27,
              margin, discount, terminal_g):
    eps = safe_float(eps, 0)
    eps_next = safe_float(eps_next, eps)
    fair_pe = safe_float(fair_pe, 20)
    growth26 = safe_float(growth26, 0)
    growth27 = safe_float(growth27, 0)
    discount = safe_float(discount, 0.09)
    terminal_g = safe_float(terminal_g, 0.03)

    eps26 = eps * (1 + growth26)
    eps27 = eps26 * (1 + growth27)

    pe_value = eps26 * fair_pe

    # Simple 3-stage DCF using EPS as a proxy for owner earnings.
    e1 = eps26
    e2 = eps27
    e3 = e2 * (1 + growth27 * 0.65)
    dcf = (
        e1 / (1 + discount)
        + e2 / (1 + discount) ** 2
        + e3 / (1 + discount) ** 3
    )
    terminal = e3 * (1 + terminal_g) / max(discount - terminal_g, 0.01)
    dcf += terminal / (1 + discount) ** 3

    intrinsic = 0.55 * pe_value + 0.45 * dcf
    safety_price = intrinsic * (1 - margin)
    current_pe = price / eps26 if eps26 > 0 else np.nan
    upside = intrinsic / price - 1 if price else np.nan

    return {
        "eps26": eps26,
        "eps27": eps27,
        "pe_value": pe_value,
        "dcf": dcf,
        "intrinsic": intrinsic,
        "safety_price": safety_price,
        "current_pe": current_pe,
        "upside": upside,
    }

# -----------------------------
# Header
# -----------------------------
st.title("📈 A股价值投资终端 V2.1")
st.caption("巴菲特 + 芒格框架：实时行情 × 盈利预测 × 多模型估值 × 安全边际")

spot, data_status = load_spot()

if data_status == "LIVE":
    st.success("🟢 已连接实时行情接口：当前行情来自 AkShare。估值参数仍需根据财报/研究假设调整。")
else:
    st.warning("🟡 当前环境暂未成功连接实时行情接口，使用 Demo 数据。平台本身可以正常运行；若 Cloud 网络访问恢复，会自动切换到 LIVE。")

# -----------------------------
# Sidebar - every widget has unique key
# -----------------------------
st.sidebar.header("全局估值假设")

g = st.sidebar.slider("2026E 盈利增长 (%)", -20, 80, 25, 1, key="growth_2026")
g27 = st.sidebar.slider("2027E 盈利增长 (%)", -20, 80, 22, 1, key="growth_2027")
fair_pe = st.sidebar.slider("合理 PE", 8.0, 60.0, 25.0, 0.5, key="fair_pe")
margin_pct = st.sidebar.slider("安全边际 (%)", 0, 50, 25, 5, key="safety_margin")
discount_pct = st.sidebar.slider("DCF 折现率 (%)", 6.0, 15.0, 9.0, 0.5, key="discount_rate")
terminal_g_pct = st.sidebar.slider("终值增长率 (%)", 1.0, 5.0, 3.0, 0.5, key="terminal_growth")

margin = margin_pct / 100
discount = discount_pct / 100
terminal_g = terminal_g_pct / 100

# -----------------------------
# Market radar
# -----------------------------
st.header("市场雷达")

if data_status == "LIVE" and not spot.empty:
    radar = spot.copy()
    up = int((radar["change_pct"] > 0).sum()) if "change_pct" in radar else 0
    down = int((radar["change_pct"] < 0).sum()) if "change_pct" in radar else 0
    pe_mean = radar["pe"].replace([np.inf, -np.inf], np.nan).dropna().mean() if "pe" in radar else np.nan
    count = len(radar)
else:
    radar = DEMO.copy()
    count = len(radar)
    up = 3
    down = 2
    pe_mean = 24.7

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("股票数量", f"{count:,}")
c2.metric("上涨", f"{up:,}")
c3.metric("下跌", f"{down:,}")
c4.metric("平均动态PE", f"{pe_mean:.1f}x" if pd.notna(pe_mean) else "-")
c5.metric("数据状态", data_status)

# -----------------------------
# Search
# -----------------------------
st.subheader("🔎 搜索股票")

query = st.text_input(
    "代码 / 名称",
    placeholder="例如：601138、工业富联、新易盛、中际旭创",
    key="stock_search",
)

if data_status == "LIVE" and not spot.empty:
    candidates = spot.copy()
else:
    candidates = DEMO.copy()

if query:
    q = query.strip().lower()
    candidates = candidates[
        candidates["code"].astype(str).str.lower().str.contains(q, na=False)
        | candidates["name"].astype(str).str.lower().str.contains(q, na=False)
    ]

if candidates.empty:
    st.info("没有找到匹配股票。可以输入 601138 / 工业富联 / 新易盛 等。")
    st.stop()

# Use a stable, unique selectbox key.
options = [
    f"{r.code} | {r.name}"
    for r in candidates[["code", "name"]].head(30).itertuples(index=False)
]
selected = st.selectbox("选择股票", options, key="selected_stock")
selected_code = selected.split("|")[0].strip()

row, row_status = get_stock_row(selected_code, spot)

if row is None:
    st.error("无法取得股票数据。")
    st.stop()

# Fill model inputs from DEMO when live market data does not contain fundamentals.
demo_row = DEMO[DEMO.code == selected_code]
d = demo_row.iloc[0].to_dict() if not demo_row.empty else {}

price = safe_float(row.get("price"), safe_float(d.get("price"), np.nan))
eps = safe_float(row.get("eps"), safe_float(d.get("eps"), 1.0))
eps_next = safe_float(row.get("eps_next"), safe_float(d.get("eps_next"), eps * 1.2))

result = valuation(
    price, eps, eps_next, fair_pe,
    g / 100, g27 / 100, margin, discount, terminal_g
)

# -----------------------------
# Stock overview
# -----------------------------
st.header(f"{row.get('name', selected_code)}（{selected_code}）")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("当前价格", fmt_num(price))
m2.metric("2026E EPS", fmt_num(result["eps26"]))
m3.metric("2026E PE", f"{result['current_pe']:.1f}x" if pd.notna(result["current_pe"]) else "-")
m4.metric("综合内在价值", fmt_num(result["intrinsic"]))
m5.metric("安全边际价格", fmt_num(result["safety_price"]))

if pd.notna(result["upside"]):
    if price <= result["safety_price"]:
        st.success(f"🟢 当前价格低于安全边际价格，模型安全边际约 {result['upside']:.1%}。")
    elif price <= result["intrinsic"]:
        st.info(f"🟡 当前价格低于模型内在价值，但尚未达到 {margin_pct}% 安全边际。")
    else:
        st.error(f"🔴 当前价格高于模型内在价值，模型隐含上行空间 {result['upside']:.1%}。")

# -----------------------------
# Valuation breakdown
# -----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("估值拆分")
    val_df = pd.DataFrame({
        "模型": ["PE估值", "DCF估值", "综合内在价值", f"{margin_pct}%安全边际价格"],
        "价格": [
            result["pe_value"],
            result["dcf"],
            result["intrinsic"],
            result["safety_price"],
        ],
    })
    st.dataframe(
        val_df.style.format({"价格": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("估值敏感性：合理 PE")
    pes = [15, 20, 25, 30, 35]
    sens = pd.DataFrame({
        "合理PE": pes,
        "对应价格": [result["eps26"] * p for p in pes],
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sens["合理PE"],
        y=sens["对应价格"],
        mode="lines+markers",
        name="PE估值"
    ))
    fig.add_hline(y=price, line_dash="dash", annotation_text="当前价格")
    fig.update_layout(
        height=350,
        xaxis_title="合理 PE",
        yaxis_title="估值价格",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# History
# -----------------------------
st.subheader("📉 股价走势")

hist = load_history(selected_code)
if hist.empty:
    hist = demo_history(price)
    hist_status = "DEMO"
else:
    hist_status = "LIVE"

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=hist["date"],
    y=hist["close"],
    mode="lines",
    name="收盘价",
))
fig2.update_layout(
    height=400,
    xaxis_title="日期",
    yaxis_title="价格",
    margin=dict(l=20, r=20, t=20, b=20),
)
st.plotly_chart(fig2, use_container_width=True)
st.caption(f"历史价格数据：{hist_status}；实时行情状态：{row_status}。")

# -----------------------------
# Quality checklist
# -----------------------------
st.subheader("🧠 芒格 / 巴菲特质量检查框架")

quality = pd.DataFrame([
    ["盈利增长", f"{g}% / {g27}%", "需要结合公司订单、行业周期和资本开支验证"],
    ["估值纪律", f"{fair_pe:.1f}x", "不要为了高增长无限提高合理PE"],
    ["安全边际", f"{margin_pct}%", "价格越低，错误空间越大"],
    ["现金流", "待接入财报", "V3 将加入 FCF / ROIC / ROE"],
    ["护城河", "待研究", "规模、客户黏性、技术、成本优势"],
    ["管理层", "待研究", "资本配置、股东回报、诚信与执行力"],
], columns=["维度", "当前状态", "投资含义"])
st.dataframe(quality, use_container_width=True, hide_index=True)

st.divider()
st.caption("V2.1 稳定修正版：统一 widget key、避免重复组件 ID、对行情/历史数据采用 LIVE → DEMO 自动降级。")
