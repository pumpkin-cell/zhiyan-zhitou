# code_library.py
# =====================================================================
# 代码库：把用户在页面直接上传的 .py 代码沉淀为可审计、可追溯的知识
#   - 上传后由大模型理解（作用/类别/输入输出/是否可复现因子/依赖）
#   - 按 MD5 内容指纹去重（同一份代码不重复入库）
#   - 若识别为可复现量化因子，可进一步注册到因子库
# 持久化：./output/code_library.json
# =====================================================================
import os
import json
import hashlib
from datetime import datetime

CODE_LIBRARY_PATH = "./output/code_library.json"


class CodeLibrary:
    """轻量代码库（JSON 持久化，无外部依赖）。"""

    def __init__(self, path=CODE_LIBRARY_PATH):
        self.path = path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, encoding='utf-8') as f:
                d = json.load(f)
                d.setdefault("codes", [])
                return d
        return {"codes": []}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def register(self, filename, content, summary):
        """入库一份代码（调用前建议先用 find_by_md5 去重）。"""
        md5 = hashlib.md5(content.encode('utf-8')).hexdigest()
        rec = {
            "file": filename,
            "md5": md5,
            "content": content,
            "summary": summary,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "页面手动上传",
        }
        self.data["codes"].append(rec)
        self.save()
        return rec

    def get_all(self):
        return self.data["codes"]

    def find_by_md5(self, md5):
        return [c for c in self.data["codes"] if c.get("md5") == md5]


def run_code_strategy(code_content, entry_func="run_strategy", params=None):
    """执行代码库里的策略代码：受限命名空间预置 np/pd，调用入口函数并返回结果。
    用于「复现时自动调用代码库策略」（企业内上传的代码，人工确认参数后执行）。"""
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
        # 入口函数不接受关键字参数时，无参调用
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


if __name__ == '__main__':
    lib = CodeLibrary()
    print(f"代码库已就绪，共 {len(lib.get_all())} 份代码，路径 {lib.path}")
