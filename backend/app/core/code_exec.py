# backend/app/core/code_exec.py
# =====================================================================
# 代码执行与静态安全检查（纯函数，不涉及数据库）
# 从原 code_library.py 迁移而来
# =====================================================================


def run_code_strategy(code_content, entry_func="run_strategy", params=None):
    """执行代码库里的策略代码：受限命名空间预置 np/pd，调用入口函数并返回结果。"""
    import traceback
    ns = {"__name__": "code_strategy"}
    try:
        import numpy as np
        ns["np"] = np
    except ImportError:
        pass
    try:
        import pandas as pd
        ns["pd"] = pd
    except ImportError:
        pass
    try:
        exec(code_content, ns)
    except Exception as e:
        return {"error": f"代码加载失败：{e}"}
    fn = ns.get(entry_func)
    if not callable(fn):
        for k, v in ns.items():
            if callable(v) and not k.startswith("_"):
                fn, entry_func = v, k
                break
    if not callable(fn):
        return {"error": f"未找到入口函数 {entry_func}，请确认代码含可调用的顶层函数"}
    try:
        result = fn(**(params or {}))
    except TypeError:
        result = fn()
    except Exception as e:
        return {"error": f"执行失败：{e}\n{traceback.format_exc()}"}
    return result if isinstance(result, dict) else {"result": result}


def safe_code_checks(code):
    """生成代码的静态安全/无未来函数检查，返回 (是否通过, 问题列表)。"""
    import re
    issues = []
    allowed = {"numpy", "pandas", "math", "typing"}
    for m in re.finditer(r'^\s*(?:import\s+([A-Za-z_][\w.]*)|from\s+([A-Za-z_][\w.]*)\s+import)',
                         code, re.MULTILINE):
        mod = (m.group(1) or m.group(2)).split('.')[0]
        if mod not in allowed:
            issues.append(f"违规 import：{mod}")
    for kw in ["os.", "sys.", "subprocess", "requests", "open(", "eval(", "exec(",
               "__import__", "random."]:
        if kw in code:
            issues.append(f"违规调用：{kw}")
    if "shift(-" in code:
        issues.append("疑似未来函数：shift(负数)")
    for deprecated in ["stack(dropna", ".append(", "iteritems"]:
        if deprecated in code:
            issues.append(f"pandas 3.x 已移除 API：{deprecated}")
    return (len(issues) == 0), issues


def run_generated_factor(code, close_df, params=None):
    """执行大模型生成的因子代码，传入 close 数据，返回 (因子值, 错误信息)。"""
    import traceback
    ns = {"__name__": "gen_factor"}
    try:
        import numpy as np
        ns["np"] = np
    except ImportError:
        pass
    try:
        import pandas as pd
        ns["pd"] = pd
    except ImportError:
        pass
    try:
        exec(code, ns)
    except Exception as e:
        return None, f"代码加载失败：{e}"
    fn = ns.get("factor")
    if not callable(fn):
        for k, v in ns.items():
            if callable(v) and not k.startswith("_"):
                fn = v
                break
    if not callable(fn):
        return None, "未找到 factor 函数"
    try:
        result = fn(close_df, **(params or {}))
    except TypeError:
        result = fn(close_df)
    except Exception as e:
        return None, f"执行失败：{e}\n{traceback.format_exc()}"
    return result, None
