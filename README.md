# A股价值投资终端 V2

## 功能

- A股行情接口
- 股票搜索与市场雷达
- 个股历史价格
- 2026E / 2027E EPS
- Forward PE
- PE估值
- DCF估值
- 综合内在价值
- 安全边际价格
- 估值敏感性矩阵
- 芒格多维度企业质量评分
- 全市场安全边际扫描
- LIVE / DEMO 数据降级

## 本地运行

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## 部署

可以部署到 Streamlit Community Cloud、Docker 或其他 Python 云服务器。

## 重要

V2中的盈利增长率、合理PE、DCF参数是研究模型输入，不应被理解为真实分析师一致预期。
实时行情/财报接口可能受第三方数据源可用性影响。

下一阶段 V3 应建立数据库和真正的财报预测引擎，避免用单一增长率直接外推EPS。
