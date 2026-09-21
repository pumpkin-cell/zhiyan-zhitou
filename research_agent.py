# research_agent.py
# =====================================================================
# 面向金融量化投研工作全流程的智能体（主程序）
# 能力：
#   1. 理解自然语言指令 / 研报观点（调用 DeepSeek 大模型）
#   2. 提取「要复现什么因子 + 标的 + 参数」
#   3. 自动调用因子复现/回测引擎（复用项目已有逻辑）
#   4. 输出结构化结论 + 自动撰写研究报告
#
# 运行前设置环境变量（勿硬编码到代码）：
#   PowerShell: $env:DEEPSEEK_API_KEY = "sk-xxx"
#   python research_agent.py "复现20日动量因子，在沪深300ETF上"
# =====================================================================
import os
import re
import json
import hashlib
import numpy as np
import pandas as pd
from metrics import calc_return_metrics

RAW = "./data/raw"
EQUITY = ['159915.SZ', '510300.SH', '510500.SH', '510880.SH', '512000.SH',
          '512010.SH', '512480.SH', '512660.SH', '516160.SH', '588000.SH']
BENCH = '510300.SH'


# ---------------------------------------------------------------------
# 1. 大模型调用（多模型，OpenAI 兼容接口，可切换）
# ---------------------------------------------------------------------
MODELS = {
    "deepseek": {"base": "https://api.deepseek.com/chat/completions", "model": "deepseek-chat", "key_env": "DEEPSEEK_API_KEY"},
    "qwen": {"base": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "model": "qwen-plus", "key_env": "DASHSCOPE_API_KEY"},
    "glm": {"base": "https://open.bigmodel.cn/api/paas/v4/chat/completions", "model": "glm-4", "key_env": "ZHIPU_API_KEY"},
}


def call_llm(prompt, system=None, temperature=0.0, provider="deepseek"):
    cfg = MODELS.get(provider, MODELS["deepseek"])
    api_key = os.environ.get(cfg["key_env"])
    if not api_key:
        raise ValueError(f"未设置环境变量 {cfg['key_env']}，请先设置后再运行")
    import requests
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(
        cfg["base"],
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": cfg["model"], "messages": messages, "temperature": temperature},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------
# 2. 意图理解：自然语言 → 结构化指令
# ---------------------------------------------------------------------
def understand_intent(user_text, provider="deepseek"):
    system = "你是量化投研指令解析器。只输出合法 JSON，不要输出任何多余文字。"
    prompt = f'''请从下面这句话里提取复现指令，输出 JSON（字段固定）：
"{user_text}"

JSON 格式：
{{
  "factor": "动量/反转/波动率/情绪/均线/RSI 之一（无法判断填 未知）",
  "asset": "标的代码，默认 510300.SH",
  "lookback": 回看窗口天数整数（默认20）
}}'''
    raw = call_llm(prompt, system, provider=provider)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {"factor": "未知", "asset": "510300.SH", "lookback": 20}


def extract_pdf_text(pdf_path):
    """用 pypdf 提取 PDF 文本。"""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_pdf_factors(pdf_text, provider="deepseek"):
    """从研报 PDF 文本里提取要复现的因子（大模型理解 → 映射到已有因子库）。"""
    system = "你是量化研报复现解析器。只输出合法 JSON，不要输出任何多余文字。"
    prompt = f'''请从下面的研报内容里，提取要复现的因子，输出 JSON：
{pdf_text[:3000]}

JSON 格式：
{{
  "factor": "动量/反转/波动率/情绪/均线/RSI 之一（无法判断填 未知）",
  "lookback": 回看窗口天数整数（默认20）
}}'''
    raw = call_llm(prompt, system, provider=provider)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {"factor": "未知", "lookback": 20}


def summarize_report(pdf_text, provider="deepseek", hint=""):
    """总结研报：核心观点、关键点、主题、地域、情绪倾向（标签+分数）、是否含量化因子。
    hint：可选的用户提示方向，引导大模型总结重点。"""
    system = "你是资深行业研究员。总结研报核心观点，判断是否包含可复现的量化因子或策略。只输出合法 JSON。"
    hint_text = f"\n用户提示（优先按这个方向总结）：{hint}" if hint else ""
    prompt = f'''请总结下面的研报{hint_text}，输出 JSON：
{pdf_text[:3000]}

JSON 格式：
{{
  "title": "研报主题（一句话）",
  "publisher": "发布机构/作者（研报署名机构，无法判断填 未知）",
  "publish_date": "研报发布年份（如 2024；从研报标题/封面/落款推断，无法判断填 未知）",
  "core_view": "核心观点（2-3句）",
  "key_points": ["关键点1", "关键点2", "关键点3"],
  "topics": ["从以下标签中选1-3个：医药、军工、半导体、软件、新能源、消费、金融、地产、外贸、宏观政策、科技、其他"],
  "region": "国内/外贸/全球 之一",
  "sentiment": "乐观/中性/悲观 之一",
  "sentiment_score": 情绪分数（-1到1，-1最悲观，1最乐观，保留一位小数）,
  "has_quant_factor": true/false（是否包含可复现的量化因子或策略）
}}'''
    raw = call_llm(prompt, system, temperature=0.0, provider=provider)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {"title": "未知", "publisher": "未知", "publish_date": "未知", "core_view": "未能总结", "key_points": [], "topics": [],
            "region": "全球", "sentiment": "中性", "sentiment_score": 0.0, "has_quant_factor": False}


def summarize_report_multi(pdf_text, provider="deepseek", hint="", n=3):
    """多视角汇总：从 3 个固定视角分别总结（每次 temperature=0，确定性输出），
    关键点取并集、情绪取众数、情绪分取平均——覆盖更全面，且整体可复现。
    用于解决「单次总结可能不全面」的顾虑：确定性不牺牲覆盖度。"""
    from collections import Counter
    views = ["从行业趋势与景气度角度", "从数据、证据与量化因子角度", "从风险与不确定性角度"][:n]
    summaries = []
    for v in views:
        combined = (hint + "；" if hint else "") + v
        try:
            summaries.append(summarize_report(pdf_text, provider=provider, hint=combined))
        except Exception:
            continue
    if not summaries:
        return summarize_report(pdf_text, provider=provider, hint=hint)
    seen, key_points = set(), []
    for s in summaries:
        for kp in s.get("key_points", []):
            if kp and kp not in seen:
                seen.add(kp)
                key_points.append(kp)
    sentiment = Counter(s.get("sentiment", "中性") for s in summaries).most_common(1)[0][0]
    scores = [s.get("sentiment_score", 0.0) for s in summaries]
    base = summaries[0]
    base["key_points"] = key_points
    base["sentiment"] = sentiment
    base["sentiment_score"] = round(sum(scores) / len(scores), 1)
    base["multi_view"] = True
    return base


CACHE_PATH = "./output/summary_cache.json"


def _load_summary_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_summary_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def summarize_report_cached(pdf_text, provider="deepseek", hint=""):
    """带缓存的研报总结（默认多视角 + 确定性）：
    - 同一 PDF 内容（hash）只总结一次，之后复用缓存 → 谁跑、何时跑结果都一致（可复现）；
    - 默认 3 个固定视角 + temperature=0（确定性），覆盖更全面且不引入随机性。"""
    key = hashlib.md5(pdf_text.encode('utf-8')).hexdigest()
    cache = _load_summary_cache()
    if key in cache:
        result = dict(cache[key])
        result["_cached"] = True
        return result
    result = summarize_report_multi(pdf_text, provider=provider, hint=hint)
    cache[key] = result
    _save_summary_cache(cache)
    return result


def understand_code(code_text, filename="", provider="deepseek"):
    """大模型理解上传的代码：作用、类别、输入输出、是否可复现量化因子、依赖。
    用于「页面直接上传 .py 文件 → 大模型理解 → 入库」的轻量代码入库通道。"""
    system = "你是量化策略代码解析器。只输出合法 JSON，不要输出任何多余文字。"
    prompt = f'''请理解下面的 Python 代码（文件：{filename}），输出 JSON（字段固定）：
```python
{code_text[:4000]}
```

JSON 格式：
{{
  "name": "代码/因子名称（一句话，简洁）",
  "purpose": "代码作用（2-3句：它做什么、用于投研哪个环节）",
  "category": "截面因子/时序因子/另类因子/策略回测/数据工具/其他 之一",
  "inputs": "输入数据（如：价格 DataFrame、参数 lookback）",
  "outputs": "输出结果（如：因子值 Series、净值曲线）",
  "is_quant_factor": true/false（是否是可复现的量化因子函数）,
  "mapped_factor": "若能对应已有因子（动量/反转/波动率/情绪/均线/RSI）填名称，否则填 null",
  "entry_func": "入口函数名（代码里可被调用执行策略的函数名，如 run_strategy、factor、backtest；没有填 null）",
  "params": {{"参数名": 默认值, ...}}（可调参数字典，供复现时只改参数，如 {{"lookback": 20, "top_k": 3}}；没有填 {{}}）,
  "deps": ["依赖库，如 numpy、pandas"]
}}'''
    raw = call_llm(prompt, system, temperature=0.0, provider=provider)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {"name": filename or "未知代码", "purpose": "未能理解", "category": "其他",
            "inputs": "", "outputs": "", "is_quant_factor": False, "mapped_factor": None,
            "entry_func": None, "params": {}, "deps": []}


def generate_factor_code(pdf_text, factor_name, provider="deepseek"):
    """大模型基于研报原文生成因子值计算代码（受严格约束，可审计、无未来函数）。
    约束：因子逻辑必须来自研报原文、白名单库、确定性、无未来函数、时间对齐。"""
    system = "你是量化因子代码生成器。只输出合法 JSON。必须严格遵守所有约束，宁可保守也不可引入未来函数。"
    prompt = f'''研报内容（因子出处只能来自这里，禁止凭空捏造）：
{pdf_text[:3000]}

要复现的因子：{factor_name}

请生成该因子的「因子值计算」Python 代码，严格遵循以下约束：
1. 因子逻辑必须来自研报原文，禁止引入研报未提到的因子或逻辑；附上"出处（研报原句）"
2. 白名单 import：只允许 numpy/pandas/math/typing；禁止 os/sys/subprocess/requests/random/open/eval/exec/__import__ 及一切文件、网络读写
3. 确定性：禁止随机种子、禁止 datetime.now()、禁止全局可变状态；同一输入必同一输出
4. 无未来函数：信号只用 t 时刻及以前数据；禁止 shift(负数)；去极值/标准化只能在 rolling 窗口内做，禁止全样本（会偷用未来均值）
5. 时间对齐（开盘收盘确认）：因子用 t 日收盘价计算，信号在 t+1 日生效——若生成择时信号，必须 signal.shift(1) 再乘收益；成本/手续费由内置回测引擎计入，因子值代码内不做交易
6. 参数用研报给的值或默认值，禁止搜索最优参数
7. 函数签名固定：def factor(close: pd.DataFrame, **params) -> pd.Series，返回逐日因子值（长度与 close 对齐）
8. pandas 兼容性（pandas 3.x）：禁止 stack(dropna=...)、DataFrame.append、Series.iteritems 等已移除 API；stack() 一律不传 dropna 参数；去 NaN 用 .dropna() 链式调用

输出 JSON：
{{
  "source_quote": "研报原句（因子出处）",
  "factor_logic": "因子逻辑（一句话）",
  "code": "完整 Python 代码（仅白名单库 import + factor 函数）",
  "params": {{"参数名": 默认值}},
  "entry_func": "factor"
}}'''
    raw = call_llm(prompt, system, temperature=0.0, provider=provider)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {"source_quote": "", "factor_logic": "", "code": "", "params": {}, "entry_func": "factor"}


def fix_factor_code(code, error, factor_name, provider="deepseek"):
    """让大模型根据报错修正代码：保持因子逻辑不变，只修复导致报错的代码问题。
    用于「执行失败后自动兜底」，属于修 bug，不涉及改参数优化收益（不违反防过拟合约束）。"""
    system = "你是量化因子代码修正器。只输出合法 JSON。只修复代码错误，绝不改变因子逻辑。"
    prompt = f'''以下因子代码执行报错，请修正代码（保持因子逻辑不变，只修复导致报错的问题）。

因子：{factor_name}
当前代码：
```python
{code}
```

报错信息：
{error}

修正时同样遵守：
- 白名单库：只 numpy/pandas/math/typing
- 无未来函数：禁 shift(负数)、禁全样本标准化
- 时间对齐：t 收盘算信号，t+1 生效
- pandas 3.x 兼容：禁 stack(dropna=...)、DataFrame.append、iteritems

输出 JSON：
{{"code": "修正后的完整代码", "params": {{"参数名": 默认值}}, "entry_func": "factor"}}'''
    raw = call_llm(prompt, system, temperature=0.0, provider=provider)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {"code": code, "params": {}, "entry_func": "factor"}


# ---------------------------------------------------------------------
# 3. 因子执行引擎（复用项目已有的复现/回测逻辑）
# ---------------------------------------------------------------------
def load_prices():
    prices = pd.read_csv(f"{RAW}/etf_prices/all_close.csv", parse_dates=['date'])
    return prices.sort_values('date').reset_index(drop=True)


def cross_sectional(prices, lookback=20, top_k=3, reverse=False):
    """截面动量/反转：按过去 lookback 日收益排序，赢家-输家组合日收益序列。
    reverse=True 时返回「输家-赢家」= 反转。"""
    close = prices[EQUITY].astype(float)
    ret = close.pct_change()
    n = len(prices)
    idx_map = {c: i for i, c in enumerate(EQUITY)}
    hold_w, hold_l = [], []
    w_ret = np.zeros(n)
    l_ret = np.zeros(n)
    for t in range(n):
        if hold_w:
            v = ret.iloc[t, hold_w].values
            v = v[~np.isnan(v)]
            w_ret[t] = v.mean() if len(v) else 0.0
        if hold_l:
            v = ret.iloc[t, hold_l].values
            v = v[~np.isnan(v)]
            l_ret[t] = v.mean() if len(v) else 0.0
        if t % max(lookback, 5) == 0 and t >= lookback:
            mom = close.iloc[t] / close.iloc[t - lookback] - 1.0
            mom = mom.dropna()
            if len(mom) >= 6:
                order = mom.sort_values(ascending=False)
                hold_w = [idx_map[c] for c in order.index[:top_k]]
                hold_l = [idx_map[c] for c in order.index[-top_k:]]
    ls = (l_ret - w_ret) if reverse else (w_ret - l_ret)
    return ls


def ic_of_factor(factor_series, prices, horizon=5):
    """时序 IC：因子值 vs 基准未来 horizon 日收益的 Spearman 相关。"""
    px = prices[BENCH]
    fr = px.pct_change(horizon).shift(-horizon)
    df = pd.DataFrame({'f': factor_series, 'r': fr}).dropna()
    if len(df) < 30:
        return 0.0
    return df['f'].corr(df['r'], method='spearman')


def volatility_factor(prices):
    return prices[BENCH].pct_change().rolling(20).std() * np.sqrt(252)


def sentiment_factor(prices):
    sent = pd.read_csv(f"{RAW}/sentiment_indices.csv", parse_dates=['date'])
    s_raw = sent.set_index('date').reindex(prices['date'])['daily_sentiment'].ffill()
    s = s_raw.fillna(s_raw.iloc[:int(len(s_raw) * 0.7)].mean())  # 训练集均值中性化
    return s.values


def ma_factor(prices, window=20):
    """均线择时：价格站上均线持有、跌破空仓，返回策略收益序列。"""
    px = prices[BENCH]
    ma = px.rolling(window).mean()
    signal = (px > ma).astype(int)
    ret = px.pct_change()
    return (signal.shift(1).fillna(0) * ret).values


def rsi_factor(prices, window=14):
    """RSI 指标：返回 RSI 序列。"""
    px = prices[BENCH]
    delta = px.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return (100 - 100 / (1 + rs)).values


def run_factor(intent):
    """按意图执行因子复现，返回结果 dict。"""
    factor = intent.get("factor", "未知")
    lookback = int(intent.get("lookback", 20) or 20)
    prices = load_prices()

    if factor == "动量":
        ls = cross_sectional(prices, lookback, reverse=False)
        m = calc_return_metrics(ls)
        return {"factor": "动量", "lookback": lookback, "annual": m['annual_return_pct'],
                "sharpe": m['sharpe_ratio'], "mdd": m['max_drawdown_pct'],
                "conclusion": f"赢家-输家年化 {m['annual_return_pct']:.2f}%，"
                              f"{'动量有效' if m['annual_return_pct'] > 2 else '动量在A股ETF短期失效，呈反转'}"}

    if factor == "反转":
        ls = cross_sectional(prices, lookback, reverse=True)
        m = calc_return_metrics(ls)
        return {"factor": "反转", "lookback": lookback, "annual": m['annual_return_pct'],
                "sharpe": m['sharpe_ratio'], "mdd": m['max_drawdown_pct'],
                "conclusion": f"输家-赢家年化 {m['annual_return_pct']:.2f}%，"
                              f"{'短期反转有效' if m['annual_return_pct'] > 2 else '反转不显著'}"}

    if factor == "波动率":
        ic = ic_of_factor(volatility_factor(prices), prices)
        return {"factor": "波动率", "lookback": 20, "ic": ic,
                "annual": None, "sharpe": None, "mdd": None,
                "conclusion": f"波动率因子时序 IC={ic:.4f}，"
                              f"{'波动率有预测力' if abs(ic) > 0.03 else '波动率因子预测力较弱'}"}

    if factor == "情绪":
        ic = ic_of_factor(sentiment_factor(prices), prices)
        return {"factor": "情绪", "lookback": 0, "ic": ic,
                "annual": None, "sharpe": None, "mdd": None,
                "conclusion": f"情绪因子时序 IC={ic:.4f}，"
                              f"{'情绪为反向指标（情绪过热→未来回落）' if ic < -0.03 else '情绪因子预测力较弱'}"}

    if factor == "均线":
        sr = ma_factor(prices, lookback)
        m = calc_return_metrics(sr)
        return {"factor": "均线", "lookback": lookback, "annual": m['annual_return_pct'],
                "sharpe": m['sharpe_ratio'], "mdd": m['max_drawdown_pct'],
                "conclusion": f"均线择时年化 {m['annual_return_pct']:.2f}%（基准约 7.3%），"
                              f"{'跑赢基准' if m['annual_return_pct'] > 7.3 else '未跑赢基准'}"}

    if factor == "RSI":
        ic = ic_of_factor(rsi_factor(prices), prices)
        return {"factor": "RSI", "lookback": 14, "ic": ic,
                "annual": None, "sharpe": None, "mdd": None,
                "conclusion": f"RSI 时序 IC={ic:.4f}，"
                              f"{'RSI 有预测力' if abs(ic) > 0.03 else 'RSI 预测力较弱'}"}

    return {"factor": "未知", "lookback": lookback, "annual": None, "sharpe": None,
            "mdd": None, "ic": None,
            "conclusion": "无法识别因子，请指定 动量/反转/波动率/情绪/均线/RSI 之一"}


# ---------------------------------------------------------------------
# 4. 研究报告生成（调用大模型把结构化结果写成文字报告）
# ---------------------------------------------------------------------
def generate_report(intent, result, provider="deepseek"):
    system = "你是资深量化研究员。基于给定数据写一段严谨、有机制解释、含风险提示的研究结论。"
    prompt = f'''请根据以下复现结果，写一份 150 字以内的量化研究结论（含因子机制解释和风险提示）：

复现指令：{json.dumps(intent, ensure_ascii=False)}
复现结果：{json.dumps(result, ensure_ascii=False)}'''
    return call_llm(prompt, system, temperature=0.5, provider=provider)


# ---------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------
def main():
    import sys
    user_text = sys.argv[1] if len(sys.argv) > 1 else "复现20日动量因子，在沪深300ETF上"

    print("=" * 78)
    print("投研智能体 · 全流程自动化")
    print("=" * 78)
    print(f"[1/4] 用户指令：{user_text}")

    intent = understand_intent(user_text)
    print(f"[2/4] 意图理解：{json.dumps(intent, ensure_ascii=False)}")

    result = run_factor(intent)
    print(f"[3/4] 因子复现完成：")
    for k, v in result.items():
        if v is not None:
            print(f"      {k}: {v}")

    print(f"[4/4] 生成研究报告...")
    report = generate_report(intent, result)
    print("\n" + "─" * 78)
    print("📄 自动生成的研究报告：")
    print("─" * 78)
    print(report)
    print("=" * 78)


if __name__ == '__main__':
    main()
