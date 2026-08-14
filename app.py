import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="A股价值投资终端 V2", page_icon="📈", layout="wide")

# ---------- Data ----------
@st.cache_data(ttl=60)
def market_data():
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        ren = {"代码":"code","名称":"name","最新价":"price","涨跌幅":"change_pct",
               "成交额":"amount","换手率":"turnover","市盈率-动态":"pe","市净率":"pb","总市值":"market_cap"}
        df = df.rename(columns=ren)
        cols = ["code","name","price","change_pct","amount","turnover","pe","pb","market_cap"]
        for c in cols:
            if c not in df: df[c] = np.nan
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df[cols], "LIVE"
    except Exception:
        rows = [
            ["601138","工业富联",100,1.25,28.6,4.8,2100000,3.50,22,24,28],
            ["300308","中际旭创",600,-0.85,31.5,10.2,4800000,19.00,32,35,38],
            ["300502","新易盛",620,2.10,29.8,12.0,3500000,20.80,35,38,42],
            ["600900","长江电力",30.5,0.45,18.2,2.1,720000,1.40,8,9,10],
            ["601899","紫金矿业",25.8,-1.10,15.5,3.0,680000,1.60,12,14,16],
        ]
        return pd.DataFrame(rows, columns=["code","name","price","change_pct","pe","pb","market_cap","eps","roe","g2026","g2027"]), "DEMO"

@st.cache_data(ttl=300)
def history(code):
    try:
        import akshare as ak
        h = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20240101", adjust="")
        if h is not None and len(h):
            h = h.rename(columns={"日期":"date","收盘":"close","成交额":"amount"})
            h["date"] = pd.to_datetime(h["date"])
            return h[["date","close","amount"]]
    except Exception:
        pass
    # deterministic synthetic history for demo
    dates = pd.date_range("2025-01-01", periods=380, freq="B")
    seed = sum(ord(x) for x in code)
    rng = np.random.default_rng(seed)
    base = 80 + (seed % 200)
    ret = rng.normal(0.0004, 0.018, len(dates))
    close = base * np.exp(np.cumsum(ret))
    return pd.DataFrame({"date":dates, "close":close, "amount":rng.uniform(1e8,5e9,len(dates))})

# ---------- Models ----------
def dcf_value(eps, growth, discount, terminal_growth, years=5, payout=0.0):
    if not np.isfinite(eps): return np.nan
    fcf = eps * max(0.0, 1-payout)
    pv = 0
    for y in range(1, years+1):
        fcf_y = fcf * (1+growth)**y
        pv += fcf_y / (1+discount)**y
    terminal = fcf * (1+growth)**years * (1+terminal_growth) / (discount-terminal_growth)
    pv_terminal = terminal / (1+discount)**years
    return pv + pv_terminal

def score_quality(roe, growth, margin, pe, net_debt, moat, management):
    vals = [
        np.clip(roe/25*100,0,100),
        np.clip(growth/35*100,0,100),
        np.clip(margin/30*100,0,100),
        np.clip(100-pe/60*100,0,100) if np.isfinite(pe) else 50,
        np.clip(100-net_debt/80*100,0,100),
        moat, management
    ]
    return float(np.mean(vals))

market, source = market_data()

st.title("📈 A股价值投资终端 V2")
st.caption("巴菲特 + 芒格框架：实时行情 × 盈利预测 × 多模型估值 × 安全边际")

if source == "LIVE":
    st.success("行情接口已连接")
else:
    st.warning("当前运行环境未连接行情接口，使用内置 Demo 数据；部署到可联网环境后自动尝试实时接口。")

# ---------- Sidebar ----------
st.sidebar.header("全局估值假设")

g = st.sidebar.slider(
    "2026E 盈利增长 (%)",
    min_value=-20,
    max_value=80,
    value=25,
    step=1
)

g27 = st.sidebar.slider(
    "2027E 盈利增长 (%)",
    min_value=-20,
    max_value=80,
    value=22,
    step=1
)

fair_pe = st.sidebar.slider(
    "合理 PE",
    min_value=8.0,
    max_value=60.0,
    value=25.0,
    step=0.5
)

margin_pct = st.sidebar.slider(
    "安全边际 (%)",
    min_value=0,
    max_value=50,
    value=25,
    step=5
)

margin = margin_pct / 100

discount_pct = st.sidebar.slider(
    "DCF 折现率 (%)",
    min_value=6.0,
    max_value=15.0,
    value=9.0,
    step=0.5
)

discount = discount_pct / 100

terminal_g_pct = st.sidebar.slider(
    "终值增长率 (%)",
    min_value=1.0,
    max_value=5.0,
    value=3.0,
    step=0.5
)

terminal_g = terminal_g_pct / 100
margin_pct = st.sidebar.slider(
    "安全边际 (%)",
    min_value=0,
    max_value=50,
    value=25,
    step=5
)

margin = margin_pct / 100
discount = st.sidebar.slider("DCF 折现率", 6.0, 15.0, 9.0, 0.5) / 100
terminal_g = st.sidebar.slider("终值增长率", 1.0, 5.0, 3.0, 0.5) / 100

# ---------- Dashboard ----------
st.subheader("市场雷达")
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("股票数量", f"{len(market):,}")
c2.metric("上涨", f"{int((pd.to_numeric(market.change_pct, errors='coerce')>0).sum()):,}")
c3.metric("下跌", f"{int((pd.to_numeric(market.change_pct, errors='coerce')<0).sum()):,}")
c4.metric("平均动态PE", f"{pd.to_numeric(market.pe, errors='coerce').mean():.1f}x")
c5.metric("数据状态", source)

query = st.text_input("🔎 搜索股票", placeholder="代码 / 名称")
if query:
    view = market[market.code.str.contains(query, na=False) | market.name.str.contains(query, na=False)]
else:
    view = market

st.dataframe(
    view[["code","name","price","change_pct","pe","pb","turnover","market_cap"]]
    .rename(columns={"code":"代码","name":"名称","price":"现价","change_pct":"涨跌幅%",
                     "pe":"动态PE","pb":"PB","turnover":"换手率","market_cap":"总市值"}),
    use_container_width=True, hide_index=True
)

# ---------- Stock analysis ----------
st.subheader("🔬 个股深度估值")
codes = market.code.tolist()
default = codes.index("601138") if "601138" in codes else 0
code = st.selectbox("选择股票", codes, index=default, format_func=lambda x: f"{x}  {market.loc[market.code==x,'name'].iloc[0]}")

r = market[market.code==code].iloc[0]
price = float(r.price)
pe_now = float(r.pe) if pd.notna(r.pe) else np.nan
eps = float(r.eps) if "eps" in r.index and pd.notna(r.eps) else (price/pe_now if np.isfinite(pe_now) and pe_now>0 else np.nan)
roe = float(r.roe) if "roe" in r.index and pd.notna(r.roe) else 15
g26 = g/100
g27v = g27/100

eps26 = eps*(1+g26)
eps27 = eps26*(1+g27v)
pe26 = price/eps26 if eps26 else np.nan
pe27 = price/eps27 if eps27 else np.nan
pe_value = eps26*fair_pe
pe_value27 = eps27*fair_pe/(1+discount)
dcf = dcf_value(eps, g26, discount, terminal_g)
intrinsic = np.nanmean([pe_value, pe_value27, dcf])
safe_price = intrinsic*(1-margin)
upside = intrinsic/price-1

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("现价", f"¥{price:.2f}")
k2.metric("TTM/基准EPS", f"¥{eps:.2f}")
k3.metric("2026E EPS", f"¥{eps26:.2f}")
k4.metric("2026E PE", f"{pe26:.1f}x")
k5.metric("内在价值", f"¥{intrinsic:.2f}")
k6.metric("安全边际价", f"¥{safe_price:.2f}")

if price <= safe_price:
    st.success("🟢 当前价格低于安全边际价格：进入重点研究区")
elif price <= intrinsic:
    st.warning("🟡 当前价格低于估算内在价值，但安全边际不足")
else:
    st.error("🔴 当前价格高于估算内在价值")

# ---------- Valuation table ----------
st.markdown("### 💰 多模型估值")
valuation = pd.DataFrame({
    "模型":["当前价格","2026E PE × 合理PE","2027E PE折现","DCF","综合内在价值","25%安全边际"],
    "价格":[price,pe_value,pe_value27,dcf,intrinsic,intrinsic*(1-margin)],
    "解释":[
        "市场报价",
        f"EPS26 ¥{eps26:.2f} × {fair_pe:.1f}x",
        f"EPS27 ¥{eps27:.2f} × {fair_pe:.1f}x，再折现",
        f"折现率 {discount:.1%}，终值增长 {terminal_g:.1%}",
        "三种方法等权平均",
        f"内在价值 × (1-{margin:.0%})"
    ]
})
st.dataframe(valuation.style.format({"价格":"¥{:.2f}"}), use_container_width=True, hide_index=True)

# ---------- Price history ----------
st.markdown("### 📉 股价动态")
h = history(code)
fig = px.line(h, x="date", y="close", title=f"{r['name']} 历史收盘价")
st.plotly_chart(fig, use_container_width=True)

# ---------- Quality ----------
st.markdown("### 🧠 芒格多维度企业质量")
q1,q2,q3 = st.columns(3)
moat = q1.slider("护城河",0,100,75)
management = q2.slider("管理层/资本配置",0,100,75)
margin_profit = q3.slider("盈利质量",0,100,70)
net_debt = st.slider("净负债率假设 (%)", -20, 100, 20)

quality_score = score_quality(roe, g, margin_profit/100*30, pe_now, net_debt, moat, management)
qcols = st.columns(4)
qcols[0].metric("ROE", f"{roe:.1f}%")
qcols[1].metric("增长", f"{g:.1f}%")
qcols[2].metric("质量评分", f"{quality_score:.0f}/100")
qcols[3].metric("理论上涨空间", f"{upside:.1%}")

st.progress(int(np.clip(quality_score,0,100))/100)

# ---------- Sensitivity ----------
st.markdown("### 🎯 估值敏感性矩阵")
pe_range = np.arange(max(10, fair_pe-10), fair_pe+11, 5)
growth_range = np.arange(max(-10,g-15), min(80,g+16), 5)
matrix = pd.DataFrame(index=[f"{x:.0f}%" for x in growth_range],
                      columns=[f"{x:.0f}x" for x in pe_range], dtype=float)
for gg in growth_range:
    for pp in pe_range:
        matrix.loc[f"{gg:.0f}%",f"{pp:.0f}x"] = eps*(1+gg/100)*pp
st.dataframe(matrix.style.format("¥{:.0f}"), use_container_width=True)

# ---------- Watchlist ----------
st.markdown("### ⭐ 安全边际扫描")
scan = market.copy()
scan["eps26"] = pd.to_numeric(scan.get("eps"), errors="coerce") * (1+g26)
scan["fair_value"] = scan["eps26"] * fair_pe
scan["safety_price"] = scan["fair_value"] * (1-margin)
scan["margin"] = scan["fair_value"]/pd.to_numeric(scan.price,errors="coerce")-1
scan = scan.replace([np.inf,-np.inf],np.nan).dropna(subset=["margin"])
scan["状态"] = np.where(scan["price"]<=scan["safety_price"],"🟢安全边际",
                 np.where(scan["price"]<=scan["fair_value"],"🟡合理区","🔴高估区"))
st.dataframe(
    scan.sort_values("margin", ascending=False)[
        ["code","name","price","eps26","fair_value","safety_price","margin","状态"]
    ].rename(columns={"code":"代码","name":"名称","price":"现价","eps26":"2026E EPS",
                      "fair_value":"内在价值","safety_price":"安全边际价","margin":"理论空间"}),
    use_container_width=True, hide_index=True
)

st.caption(f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 本工具用于研究与估值演示，不构成投资建议。")
