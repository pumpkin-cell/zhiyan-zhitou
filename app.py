# app.py
# =====================================================================
# 智研智投 · 量化投研智能体 —— Streamlit 交互界面
# 运行：streamlit run app.py  （或 python -m streamlit run app.py）
# 三个任务：因子复现 / 因子卡片评估 / 自然语言智能体（需 DEEPSEEK_API_KEY）
# =====================================================================
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 加载项目自带中文字体，解决云端(Linux)无中文字体导致图表中文乱码的问题
import matplotlib.font_manager as _fm
_font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "NotoSansSC-Regular.otf")
if os.path.exists(_font_path):
    _fm.fontManager.addfont(_font_path)
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Microsoft YaHei', 'SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="智研智投 · 量化投研智能体", page_icon="🤖", layout="wide")
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 700;}
    .stButton > button {border-radius: 10px; font-weight: 600; border: none;
        background: linear-gradient(135deg,#2b5797,#3a7bd5); color: #fff;
        box-shadow: 0 2px 8px rgba(43,87,151,.25);}
    .stButton > button:hover {background: linear-gradient(135deg,#1a2a6c,#2b5797); color: #fff;}
    .stExpander {border-radius: 10px; border: 1px solid rgba(49,51,63,0.15);}
    div[data-testid="stSidebar"] {background-color: #f7f8fa;}
    div[data-testid="stHeader"] {background: transparent;}
    h1, h2, h3 {color: #1a2a6c;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#1a2a6c 0%,#2b5797 55%,#3a7bd5 100%);
            padding:1.6rem 2rem;border-radius:14px;margin-bottom:1rem;box-shadow:0 4px 16px rgba(43,87,151,.25);">
  <h1 style="color:#ffffff;margin:0;font-size:1.9rem;font-weight:700;">🤖 智研智投</h1>
  <p style="color:#d6e4ff;margin:0.5rem 0 0 0;font-size:0.98rem;line-height:1.5;">
  面向金融量化投研的全流程智能体 —— 大模型理解 · 确定性引擎复现 · 全程可审计、无未来函数</p>
</div>
""", unsafe_allow_html=True)

_c1, _c2, _c3, _c4 = st.columns(4)
_c1.metric("可复现因子", "6")
_c2.metric("资产池 ETF", "13")
_c3.metric("数据跨度", "2019–2026")
_c4.metric("未来函数防线", "5 道")
st.markdown("---")

task = st.sidebar.radio("选择任务", ["① 因子复现", "② 因子卡片评估", "③ 自然语言智能体", "④ 研究记忆库", "⑤ 策略代码库"])

from reproduce_batch import load_prices, cross_sectional, ic_of, rsi, ma_trend
from metrics import calc_return_metrics

EQUITY = ['159915.SZ', '510300.SH', '510500.SH', '510880.SH', '512000.SH',
          '512010.SH', '512480.SH', '512660.SH', '516160.SH', '588000.SH']
BENCH = '510300.SH'


def plot_nav(returns, title):
    nav = np.cumprod(1 + returns)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(nav, color='#2b5797', linewidth=1.5)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    st.pyplot(fig)


# ---------------------------------------------------------------------
# 任务①：因子复现
# ---------------------------------------------------------------------
if task == "① 因子复现":
    st.header("因子复现（截面动量/反转）")
    col1, col2, col3 = st.columns(3)
    factor = col1.selectbox("因子", ["动量", "反转"])
    lookback = col2.slider("回看窗口(交易日)", 5, 120, 20)
    top_k = col3.slider("组合持仓数", 1, 5, 3)

    if st.button("🚀 运行复现"):
        prices = load_prices()
        ls = cross_sectional(prices, lookback, reverse=(factor == "反转"), hold=20)
        m = calc_return_metrics(ls)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("年化收益", f"{m['annual_return_pct']:.2f}%")
        c2.metric("夏普比率", f"{m['sharpe_ratio']:.3f}")
        c3.metric("最大回撤", f"{m['max_drawdown_pct']:.2f}%")
        c4.metric("日胜率", f"{m['daily_win_rate_pct']:.1f}%")
        plot_nav(ls, f"{factor}因子（lookback={lookback}, top{top_k}）净值曲线")

# ---------------------------------------------------------------------
# 任务②：因子卡片评估
# ---------------------------------------------------------------------
elif task == "② 因子卡片评估":
    st.header("因子库")
    st.caption("复现结果自动入库 · 状态判定 · 复现前自动避坑")
    from factor_library import FactorLibrary
    # 手动录入通道（用户自我上传因子）
    with st.expander("➕ 手动录入因子"):
        with st.form("manual_factor"):
            f_name = st.text_input("因子名")
            c1, c2 = st.columns(2)
            f_type = c1.selectbox("类型", ["截面", "时序", "另类"])
            f_source = c2.text_input("来源", placeholder="如 经典文献 / 自研")
            f_def = st.text_area("定义 / 逻辑说明")
            f_submit = st.form_submit_button("录入")
        if f_submit:
            if not f_name.strip():
                st.error("因子名不能为空")
            else:
                lib = FactorLibrary()
                lib.register(f_name.strip(), {}, {}, source=f_source.strip(),
                             definition=f_def.strip(), ftype=f_type)
                st.success(f"✅ 已录入因子「{f_name.strip()}」（状态：待验证）")
                st.rerun()
    if st.button("📊 查看因子库"):
        lib = FactorLibrary()
        factors = lib.get_all()
        if factors:
            def fmt(v):
                return None if v is None else round(float(v), 4)
            rows = []
            for f in factors:
                m = f.get("metrics", {}) or {}
                rows.append({
                    "因子": f.get("name", ""),
                    "类型": f.get("type", ""),
                    "IC": fmt(m.get("ic")),
                    "ICIR": fmt(m.get("icir")),
                    "夏普": fmt(m.get("sharpe")),
                    "状态": f.get("status", ""),
                    "来源": f.get("source", ""),
                    "版本": f.get("version", 1),
                    "更新时间": f.get("updated", ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.caption(f"共 {len(factors)} 个因子 · 有效可复用 · 已证伪自动避坑 · 待验证待复现")
        else:
            st.warning("因子库为空，请先复现因子或运行 factor_cards.py。")

# ---------------------------------------------------------------------
# 任务③：自然语言智能体
# ---------------------------------------------------------------------
elif task == "③ 自然语言智能体":
    st.header("自然语言智能体")
    st.caption("AI 理解 → 确定性引擎计算 → 人工确认，每步可追溯")

    from research_agent import MODELS
    model_names = {"deepseek": "DeepSeek", "qwen": "通义千问", "glm": "智谱 GLM"}
    provider = st.selectbox("大模型（可切换）", list(model_names.keys()), format_func=lambda p: model_names[p])

    # API key 页面输入（替代终端输入，只存内存不落盘）
    key_env = MODELS[provider]["key_env"]
    api_key = st.text_input(f"🔑 {key_env}", type="password",
                            value=os.environ.get(key_env, ""),
                            help="只存内存，不写入文件")
    if api_key:
        os.environ[key_env] = api_key

    input_mode = st.radio("输入方式", ["输入指令", "上传研报 PDF"])
    text = None
    uploaded = None
    hint = ""
    if input_mode == "输入指令":
        text = st.text_input("输入指令", "复现20日动量因子，在沪深300ETF上")
    else:
        uploaded = st.file_uploader("上传研报 PDF（可多选，最多3个）", type="pdf", accept_multiple_files=True)
        hint = st.text_input("提示方向（可选）", placeholder="留空 = 通用总结（核心观点/关键点/情绪/行业）",
                             help="可选：引导大模型聚焦某行业/方向总结")
        st.caption("多视角汇总 · 确定性输出（可复现）")

    if st.button("🤖 运行智能体"):
        from research_agent import (understand_intent, extract_pdf_text, extract_pdf_factors,
                                    summarize_report, summarize_report_cached)
        if not os.environ.get(key_env):
            st.error(f"请先在上方输入 {key_env}")
            st.stop()

        if input_mode == "输入指令":
            st.subheader("① 意图理解（大模型）")
            intent = understand_intent(text, provider=provider)
            st.json(intent)
            st.session_state["agent_intent"] = intent
            st.session_state["agent_summary"] = None
        else:
            if not uploaded:
                st.error("请先上传 PDF")
                st.stop()
            files = list(uploaded) if isinstance(uploaded, list) else [uploaded]
            files = files[:3]

            # ---- 上传后立即查重（MD5 内容指纹 + 文件名，无需等大模型解析）----
            from research_memory import ResearchMemory
            import hashlib
            _mem = ResearchMemory()
            _md5_map = {up.name: hashlib.md5(up.getvalue()).hexdigest() for up in files}
            _fresh, _dup = [], []
            for _up in files:
                _hit = _mem.find_report_by_file(_up.name, _md5_map[_up.name])
                if _hit:
                    _dup.append((_up.name, _hit[0].get("title", "")))
                else:
                    _fresh.append(_up)
            if _dup:
                _dup_lines = "；".join(f"{n}（已在库：{t}）" for n, t in _dup)
                st.warning(f"⚡ 检测到 {len(_dup)} 份已录入研报，已跳过重复分析：{_dup_lines}")
            files = _fresh
            if not files:
                st.info("本次上传的研报都已存在于记忆库，无需重复分析。")
                st.stop()
            if len(files) == 1:
                # 单个 PDF：总结 + 有因子复现 / 无因子存库
                up = files[0]
                st.subheader("① PDF 文本提取")
                with st.spinner("正在解析 PDF 并调用大模型总结（约需 30-90 秒）..."):
                    try:
                        pdf_text = extract_pdf_text(up)
                        st.session_state["agent_pdf_text"] = pdf_text
                        st.caption(f"已提取 {len(pdf_text)} 字，预览前 200 字：")
                        st.text(pdf_text[:200] + ("..." if len(pdf_text) > 200 else ""))
                        st.subheader("② 研报理解与总结（大模型）")
                        summary = summarize_report_cached(pdf_text, provider=provider, hint=hint)
                        summary["source_file"] = up.name
                        summary["md5"] = _md5_map.get(up.name, "")
                        st.json(summary)
                        st.session_state["agent_summary"] = summary
                        st.session_state["agent_intent"] = (extract_pdf_factors(pdf_text, provider=provider)
                                                            if summary.get("has_quant_factor") else None)
                    except Exception as e:
                        st.error(f"总结失败：{e}")
            else:
                # 多个 PDF：批量总结，逐个确认入库
                st.session_state["batch_summaries"] = []
                ok_count = 0
                for i, up in enumerate(files):
                    with st.spinner(f"正在总结研报 {i+1}/{len(files)}：{up.name}（大模型调用中，约需 30-90 秒）..."):
                        try:
                            pdf_text = extract_pdf_text(up)
                            summary = summarize_report_cached(pdf_text, provider=provider, hint=hint)
                            summary["source_file"] = up.name
                            summary["md5"] = _md5_map.get(up.name, "")
                            st.session_state["batch_summaries"].append(summary)
                            ok_count += 1
                        except Exception as e:
                            st.error(f"研报 {i+1}「{up.name}」总结失败：{e}")
                st.success(f"已总结 {ok_count}/{len(files)} 份研报，请在下方逐个人工确认入库。")
                st.session_state["agent_intent"] = None
                st.session_state["agent_summary"] = None

    # ---- 批量总结：逐个确认入库（不自动入库）----
    if st.session_state.get("batch_summaries"):
        from research_memory import ResearchMemory
        st.subheader("批量研报：逐个人工确认后入库")
        for i, summary in enumerate(st.session_state["batch_summaries"]):
            score = summary.get("sentiment_score", 0.0)
            emoji = "🟢" if score > 0.2 else ("🔴" if score < -0.2 else "⚪")
            with st.expander(f"研报 {i+1}：{emoji} {summary.get('title', '')}"):
                st.write(summary.get("core_view", ""))
                st.caption(f"主题：{', '.join(summary.get('topics', []))} | 地域：{summary.get('region', '')} | "
                           f"情绪：{summary.get('sentiment', '')} {score:+.1f} | 来源：{summary.get('publisher', '') or summary.get('source_file', '')}")
                if st.button(f"✅ 确认入库 #{i+1}", key=f"confirm_report_{i}"):
                    mem = ResearchMemory()
                    dup = mem.find_report_by_title(summary.get("title", ""))
                    if dup:
                        st.warning(f"⚠️ 记忆库中已存在相似研报「{dup[0].get('title', '')}」，未重复录入。")
                    else:
                        mem.record_report(summary)
                        st.success(f"已入库：{summary.get('title', '')}")

    # ---- 有量化因子：人工确认（可修改）→ 复现 ----
    if st.session_state.get("agent_intent"):
        intent = st.session_state["agent_intent"]
        factor_options = ["动量", "反转", "波动率", "情绪", "均线", "RSI"]
        default_idx = factor_options.index(intent["factor"]) if intent["factor"] in factor_options else 0
        st.subheader("人工确认（可修改因子）")
        st.caption("因子可在此修改，确认后才进入计算")
        intent["factor"] = st.selectbox("确认/修改因子", factor_options, index=default_idx)
        # 复现前查库避坑：命中「已证伪」历史结论则提示
        from factor_library import FactorLibrary
        lib = FactorLibrary()
        warns = lib.warn_if_failed(intent["factor"])
        if warns:
            st.warning(f"⚠️ 历史经验：「{intent['factor']}」已在 A 股上被复现为「已证伪」，"
                       f"建议调整参数或更换标的，无需重复探索。")

        # ---- 代码库策略匹配：命中则可用同事上传的代码自动复现 ----
        from code_library import CodeLibrary, run_code_strategy
        clib = CodeLibrary()
        matched = [c for c in clib.get_all()
                   if str(c.get("summary", {}).get("mapped_factor", "")) == intent["factor"]]
        if matched:
            st.markdown("---")
            st.subheader("🔗 代码库匹配到策略（可自动复现）")
            st.caption("以下策略来自「⑤ 策略代码库」页面同事上传，确认参数后自动执行复现")
            for ci, c in enumerate(matched):
                s = c.get("summary", {})
                with st.expander(f"📄 {c.get('file', '')}｜{s.get('name', '')}"):
                    st.write(f"**作用**：{s.get('purpose', '')}")
                    st.code(c.get("content", ""), language="python")
                    _params = s.get("params", {}) or {}
                    _user_params = {}

                    def _to_num(v):
                        try:
                            f = float(v)
                            return int(f) if f == int(f) else f
                        except (TypeError, ValueError):
                            return v

                    for _pk, _pv in _params.items():
                        _n = _to_num(_pv)
                        _user_params[_pk] = st.number_input(
                            f"参数 {_pk}", value=_n,
                            step=1 if isinstance(_n, int) else 0.1,
                            key=f"code_param_{ci}_{_pk}")
                    if st.button(f"🚀 用此代码库策略复现 #{ci+1}", key=f"run_code_{ci}"):
                        with st.spinner("执行代码库策略..."):
                            _res = run_code_strategy(c.get("content", ""), s.get("entry_func"), _user_params)
                        st.json(_res)
                        if "error" not in _res:
                            from research_memory import ResearchMemory
                            _mem2 = ResearchMemory()
                            _mem2.record(f"代码库策略复现 {s.get('name', '')}", intent, _res,
                                          str(_res.get("conclusion", _res)), [intent.get("factor", "")])
                            st.caption("✅ 复现结果已存入记忆库")

        # ---- 代码库无匹配 → 大模型基于研报原文生成因子代码（严格约束）----
        if not matched:
            st.markdown("---")
            st.subheader("🧠 代码库无匹配，可由大模型基于研报生成因子代码")
            st.caption("约束：研报原文 · 白名单库 · 无未来函数 · 时间对齐 · 计入成本")
            from research_agent import generate_factor_code
            from code_library import safe_code_checks, run_generated_factor
            if st.button("🤖 生成因子代码"):
                _pdf = st.session_state.get("agent_pdf_text", "")
                if not _pdf:
                    st.error("缺少研报原文，请重新上传研报。")
                else:
                    with st.spinner("大模型基于研报生成因子代码（受严格约束）..."):
                        st.session_state["gen_code"] = generate_factor_code(_pdf, intent["factor"], provider=provider)
            _gen = st.session_state.get("gen_code")
            if _gen:
                st.markdown(f"**因子出处（研报原句）**：{_gen.get('source_quote', '')}")
                st.markdown(f"**因子逻辑**：{_gen.get('factor_logic', '')}")
                st.code(_gen.get("code", ""), language="python")
                _ok, _issues = safe_code_checks(_gen.get("code", ""))
                if not _ok:
                    st.error("⚠️ 生成代码未通过安全检查：" + "；".join(_issues))
                else:
                    st.success("✅ 通过静态检查（白名单库 + 无未来函数 + 确定性）")
                    _gparams = _gen.get("params", {}) or {}
                    _user_params = {}

                    def _to_num2(v):
                        try:
                            f = float(v)
                            return int(f) if f == int(f) else f
                        except (TypeError, ValueError):
                            return v

                    for _pk, _pv in _gparams.items():
                        _n = _to_num2(_pv)
                        _user_params[_pk] = st.number_input(
                            f"参数 {_pk}", value=_n,
                            step=1 if isinstance(_n, int) else 0.1,
                            key=f"gen_param_{_pk}")
                    if st.button("✅ 人工确认，执行因子值计算 + IC 验证"):
                        from research_agent import load_prices, ic_of_factor, fix_factor_code
                        import pandas as _pd
                        _prices = load_prices()
                        _close = _prices[EQUITY].astype(float)
                        _cur_code = _gen.get("code", "")
                        _fv, _err = None, None
                        # 自动修正循环：执行报错 → 大模型修 bug → 重试（最多 2 次修正，不改因子逻辑）
                        for _attempt in range(3):
                            with st.spinner(f"执行生成的因子代码（第 {_attempt+1} 次）..."):
                                _fv, _err = run_generated_factor(_cur_code, _close, _user_params)
                            if _err is None:
                                break
                            st.warning(f"⚠️ 执行报错（第 {_attempt+1} 次）：{_err}")
                            if _attempt < 2:
                                with st.spinner(f"大模型自动修正代码（第 {_attempt+1} 次）..."):
                                    _fix = fix_factor_code(_cur_code, _err, intent["factor"], provider=provider)
                                    _cur_code = _fix.get("code", _cur_code)
                        if _err is not None:
                            st.error(f"多次修正后仍失败：{_err}。请检查环境依赖版本（requirements.txt 已锁版本，可 pip install -r requirements.txt 对齐）。")
                        else:
                            _gen["code"] = _cur_code
                            _ic = ic_of_factor(_pd.Series(_fv).reset_index(drop=True), _prices)
                            st.metric("因子时序 IC（未来5日收益）", f"{_ic:.4f}")
                            st.caption("|IC|>0.03 判有效；完整回测含成本/手续费")
                            from research_memory import ResearchMemory
                            _mem3 = ResearchMemory()
                            _mem3.record(f"大模型生成因子 {intent['factor']}（研报驱动）", intent,
                                          {"ic": _ic, "code": _gen.get("code", "")},
                                          f"{intent['factor']} 因子 IC={_ic:.4f}，{'有效' if abs(_ic) > 0.03 else '证伪'}",
                                          [intent.get("factor", ""), "大模型生成"])
                            from factor_library import FactorLibrary
                            _flib = FactorLibrary()
                            _flib.register(intent["factor"], _user_params, {"ic": _ic},
                                           source=f"大模型生成（研报驱动）：{_gen.get('source_quote', '')[:30]}",
                                           definition=_gen.get("factor_logic", ""))
                            st.caption("✅ 因子值 + IC + 出处 + 代码已入库（可审计）")
                            clib.register(f"gen_{intent['factor']}.py", _gen.get("code", ""), _gen)
                            st.caption("✅ 生成代码已存入代码库")

        if st.button("✅ 确认，继续复现"):
            from research_agent import run_factor, generate_report
            from research_memory import ResearchMemory
            result = run_factor(intent)
            st.subheader("因子复现结果（确定性引擎执行）")
            st.json(result)
            st.subheader("自动生成的研究报告")
            st.write(generate_report(intent, result, provider=provider))
            mem = ResearchMemory()
            mem.record(f"复现 {intent['factor']} 因子", intent, result,
                       result.get("conclusion", ""), [intent.get("factor", "")])
            # 复现结果自动写入因子库（去重更新，版本 +1）
            params = {k: v for k, v in intent.items() if k != "factor"}
            metrics = {k: result[k] for k in ["ic", "annual", "sharpe", "mdd"] if result.get(k) is not None}
            lib.register(intent["factor"], params, metrics, source="用户指令复现")
            st.caption(f"✅ 该因子已写入因子库（去重更新，版本 +1）")
            st.subheader("研究记录（已存入记忆库）")
            st.json({"因子": intent["factor"], "结果": result})

    # ---- 无量化因子（单个 PDF）：存为研究素材 ----
    elif st.session_state.get("agent_summary") is not None and st.session_state.get("agent_intent") is None:
        from research_memory import ResearchMemory
        summary = st.session_state["agent_summary"]
        st.info("该研报不包含可复现的量化因子，可作为「研究素材」存入。")
        if st.button("💾 存入研究记忆库"):
            mem = ResearchMemory()
            dup = mem.find_report_by_title(summary.get("title", ""))
            if dup:
                st.warning(f"⚠️ 记忆库中已存在相似研报「{dup[0].get('title', '')}」，未重复录入。")
            else:
                mem.record_report(summary)
                st.success(f"✅ 已存入：{summary.get('title', '')}（情绪：{summary.get('sentiment', '')}）")
            st.caption("已沉淀为可复用研究素材")

# ---------------------------------------------------------------------
# 任务④：研究记忆库（知识沉淀 · 可复查 · 多维度筛选）
# ---------------------------------------------------------------------
elif task == "④ 研究记忆库":
    st.header("研究记忆库")
    st.caption("支持按情绪 / 地域 / 行业 / 年份筛选")
    from research_memory import ResearchMemory
    mem = ResearchMemory()

    tab1, tab2 = st.tabs(["📄 研报素材", "🧪 实验结论"])

    with tab1:
        reports = mem.data.get("reports", [])
        TOPIC_LABELS = ["医药", "军工", "半导体", "软件", "新能源", "消费",
                        "金融", "地产", "外贸", "宏观政策", "科技", "其他"]
        # 手动录入通道（用户自我上传）
        with st.expander("➕ 手动录入研报素材"):
            with st.form("manual_report"):
                m_title = st.text_input("标题")
                c1, c2 = st.columns(2)
                m_publisher = c1.text_input("发布机构")
                m_publish_date = c2.text_input("发布年份", placeholder="如 2024")
                m_core = st.text_area("核心观点")
                m_topics = st.multiselect("行业标签", TOPIC_LABELS)
                c3, c4 = st.columns(2)
                m_region = c3.selectbox("地域", ["国内", "外贸", "全球"])
                m_sentiment = c4.selectbox("情绪", ["乐观", "中性", "悲观"])
                m_submit = st.form_submit_button("录入")
            if m_submit:
                if not m_title.strip():
                    st.error("标题不能为空")
                else:
                    dup = mem.find_report_by_title(m_title)
                    if dup:
                        st.warning(f"⚠️ 已存在相似研报「{dup[0].get('title', '')}」，未重复录入。")
                    else:
                        mem.record_report({
                            "title": m_title.strip(),
                            "publisher": m_publisher.strip(),
                            "publish_date": m_publish_date.strip(),
                            "core_view": m_core.strip(),
                            "key_points": [],
                            "topics": m_topics,
                            "region": m_region,
                            "sentiment": m_sentiment,
                            "sentiment_score": {"乐观": 0.5, "中性": 0.0, "悲观": -0.5}[m_sentiment],
                            "has_quant_factor": False,
                            "source_file": "手动录入",
                        })
                        st.success("✅ 已录入")
                        st.rerun()
        if not reports:
            st.info("暂无研报素材。去「③ 自然语言智能体」上传研报后，会沉淀到这里。")
        else:
            st.subheader(f"已沉淀 {len(reports)} 份研报素材")
            col1, col2, col3, col4 = st.columns(4)
            all_sentiments = sorted(set(r.get("sentiment", "中性") for r in reports))
            all_regions = sorted(set(r.get("region", "全球") for r in reports))
            # 行业：规范标签 + 自动追加新标签
            discovered = sorted(set(t for r in reports for t in r.get("topics", []) if t) - set(TOPIC_LABELS))
            all_topics = TOPIC_LABELS + discovered
            # 年份：优先用「研报发布年份」，回退到录入年份
            def _year(r):
                s = (r.get("publish_date") or r.get("time", "") or "")[:4]
                return s if s.isdigit() else ""
            all_years = sorted(set(_year(r) for r in reports if _year(r)), reverse=True)
            sent_filter = col1.multiselect("按情绪", all_sentiments)
            region_filter = col2.multiselect("按地域", all_regions)
            topic_filter = col3.multiselect("按行业", all_topics)
            year_filter = col4.multiselect("按年份", all_years)

            indexed = [(i, r) for i, r in enumerate(reports)]
            filtered = sorted(indexed, key=lambda x: x[1].get("time", ""), reverse=True)
            if sent_filter:
                filtered = [x for x in filtered if x[1].get("sentiment") in sent_filter]
            if region_filter:
                filtered = [x for x in filtered if x[1].get("region") in region_filter]
            if topic_filter:
                filtered = [x for x in filtered if any(t in x[1].get("topics", []) for t in topic_filter)]
            if year_filter:
                filtered = [x for x in filtered if _year(x[1]) in year_filter]

            st.caption(f"筛选结果：{len(filtered)} 份")
            for idx, r in filtered:
                score = r.get("sentiment_score", 0.0)
                emoji = "🟢" if score > 0.2 else ("🔴" if score < -0.2 else "⚪")
                with st.expander(f"{emoji} {r.get('title', '')}（{r.get('sentiment', '')} {score:+.1f}）"):
                    st.write(r.get("core_view", ""))
                    if r.get("key_points"):
                        st.write("**关键点：**")
                        for kp in r["key_points"]:
                            st.write(f"· {kp}")
                    st.caption(f"主题：{', '.join(r.get('topics', [])) or '无'} | "
                               f"地域：{r.get('region', '')} | 发布：{r.get('publish_date') or '未知'} | "
                               f"来源：{r.get('publisher', '') or r.get('source_file', '')}")
                    if st.button("🗑️ 删除", key=f"del_report_{idx}"):
                        mem.delete_report(idx)
                        st.rerun()

    with tab2:
        experiments = mem.data.get("experiments", [])
        if not experiments:
            st.info("暂无实验结论。")
        else:
            st.subheader(f"已沉淀 {len(experiments)} 组实验结论")
            for e in experiments[-15:]:
                st.write(f"· {e.get('conclusion', '')}（{e.get('time', '')}）")


# ---------------------------------------------------------------------
# 任务⑤：策略代码库（确定性引擎 · 公开可审计）
# ---------------------------------------------------------------------
elif task == "⑤ 策略代码库":
    st.header("策略代码库")
    st.caption("展示的代码即实际运行代码，公开可审计")

    # ================= 代码上传通道（页面直接上传 .py，无需 git） =================
    st.markdown("---")
    st.subheader("⬆️ 上传代码（页面直接上传 .py，无需 git）")
    st.caption("上传 .py → 大模型理解 → 入库；识别为因子时自动注册到因子库")
    from research_agent import MODELS, understand_code
    from code_library import CodeLibrary
    clib = CodeLibrary()
    _model_names = {"deepseek": "DeepSeek", "qwen": "通义千问", "glm": "智谱 GLM"}
    _prov = st.selectbox("理解代码用的大模型", list(_model_names.keys()),
                         format_func=lambda p: _model_names[p])
    _key_env = MODELS[_prov]["key_env"]
    _api_key = st.text_input(f"🔑 {_key_env}", type="password",
                             value=os.environ.get(_key_env, ""),
                             help="页面输入 API Key，只存内存、不写入文件")
    if _api_key:
        os.environ[_key_env] = _api_key
    up_py = st.file_uploader("上传 .py 文件（可多选）", type="py", accept_multiple_files=True)
    if up_py is not None and st.button("🤖 理解并入库"):
        if not os.environ.get(_key_env):
            st.error(f"请先在上方输入 {_key_env}")
            st.stop()
        import hashlib
        _files = list(up_py) if isinstance(up_py, list) else [up_py]
        for _f in _files:
            _content = _f.getvalue().decode('utf-8', errors='ignore')
            _md5 = hashlib.md5(_content.encode('utf-8')).hexdigest()
            if clib.find_by_md5(_md5):
                st.warning(f"⚡ {_f.name} 已在代码库中，跳过。")
                continue
            with st.spinner(f"🤖 大模型理解 {_f.name} ..."):
                _sum = understand_code(_content, filename=_f.name, provider=_prov)
            clib.register(_f.name, _content, _sum)
            st.success(f"✅ {_f.name} 已入库：{_sum.get('name', '')}")
            if _sum.get("is_quant_factor") and _sum.get("mapped_factor"):
                from factor_library import FactorLibrary
                flib = FactorLibrary()
                flib.register(_sum["mapped_factor"], {}, {}, source=f"用户上传代码：{_f.name}",
                              definition=_sum.get("purpose", ""), ftype=_sum.get("category", ""))
                st.caption(f"🔗 已识别为因子「{_sum['mapped_factor']}」，注册到因子库（待验证）。")

    # ================= 代码库列表 =================
    st.markdown("---")
    st.subheader(f"📦 已上传代码库（{len(clib.get_all())} 个）")
    _codes = clib.get_all()
    if not _codes:
        st.info("暂无上传代码。用上方「上传 .py」可在页面直接提交，无需 git。")
    else:
        for _c in reversed(_codes):
            _s = _c.get("summary", {})
            with st.expander(f"📄 {_c.get('file', '')}｜{_s.get('name', '')}｜{_s.get('category', '')}"):
                st.write(f"**作用**：{_s.get('purpose', '')}")
                st.caption(f"输入：{_s.get('inputs', '')} ｜ 输出：{_s.get('outputs', '')} ｜ "
                           f"可复现因子：{'是' if _s.get('is_quant_factor') else '否'} ｜ "
                           f"映射：{_s.get('mapped_factor') or '无'} ｜ "
                           f"依赖：{', '.join(_s.get('deps', [])) or '无'}")
                st.code(_c.get("content", ""), language="python")

    import inspect
    from research_agent import (cross_sectional as ra_cross_sectional, volatility_factor,
                                sentiment_factor, ma_factor, rsi_factor)
    from factor_library import FactorLibrary, FACTOR_META

    factor_fn = {
        "动量": ra_cross_sectional,
        "反转": ra_cross_sectional,
        "波动率": volatility_factor,
        "情绪": sentiment_factor,
        "均线": ma_factor,
        "RSI": rsi_factor,
    }

    lib = FactorLibrary()
    for name, meta in FACTOR_META.items():
        sign_label = "正向" if meta["sign"] == 1 else ("反向" if meta["sign"] == -1 else "双向")
        fac = lib.find(name)
        status = fac[0]["status"] if fac else "待验证"
        version = fac[0]["version"] if fac else 1
        with st.expander(f"{name}｜{meta['type']}｜{status}（v{version}）｜{meta['source']}"):
            st.markdown(f"**定义**：{meta['definition']}")
            st.markdown(f"**预期方向**：{sign_label}")
            fn = factor_fn.get(name)
            if fn is not None:
                st.markdown("**核心实现代码**（与实际运行完全一致）：")
                st.code(inspect.getsource(fn), language="python")

    # ---- 回测引擎（策略层）----
    st.markdown("---")
    st.subheader("回测引擎（策略层：轮动 + 缓冲带 + 波动率目标 + 黄金防守 + 换仓成本）")
    st.markdown("**指标计算**（`metrics.py`，与实际运行一致）：")
    st.code(inspect.getsource(calc_return_metrics), language="python")

    st.markdown("**策略主循环**（`rotate_backtest_v4.py` 的 `run_strategy`，含信号滞后与换仓成本）：")
    try:
        with open("./rotate_backtest_v4.py", encoding='utf-8') as _f:
            _src = _f.read()
        _start = _src.find("def run_strategy(")
        if _start >= 0:
            _next = _src.find("\ndef ", _start + 1)
            _end = _next if _next >= 0 else len(_src)
            st.code(_src[_start:_end].rstrip(), language="python")
    except Exception as _e:
        st.warning(f"未能读取策略源码：{_e}")

    st.caption("以上即「③ 智能体」复现时调用的确定性引擎，可逐行审计")
