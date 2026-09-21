# research_memory.py
# =====================================================================
# 命题1 模块5「自我进化」—— 研究记忆库
# 把每次实验的假设/参数/结论/指标沉淀为可复用知识，支持查询相似历史实验，
# 使智能体「从历史任务中持续改善研究流程，而非每次重新开始」。
# 用法：
#   python research_memory.py            # 运行演示：记录几组实验 + 相似查询
#   from research_memory import ResearchMemory  # 供智能体/其他脚本调用
# =====================================================================
import os
import json
from datetime import datetime

MEMORY_PATH = "./output/research_memory.json"


class ResearchMemory:
    """轻量研究记忆库（JSON 持久化，无外部依赖）。"""

    def __init__(self, path=MEMORY_PATH):
        self.path = path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, encoding='utf-8') as f:
                d = json.load(f)
                d.setdefault("experiments", [])
                d.setdefault("reports", [])
                return d
        return {"experiments": [], "reports": []}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def record_report(self, summary):
        """记录一份研报总结（研究素材，区别于实验结论）。"""
        rep = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": summary.get("title", "未知"),
            "core_view": summary.get("core_view", ""),
            "key_points": summary.get("key_points", []),
            "topics": summary.get("topics", []),
            "region": summary.get("region", "全球"),
            "sentiment": summary.get("sentiment", "中性"),
            "sentiment_score": summary.get("sentiment_score", 0.0),
            "has_quant_factor": summary.get("has_quant_factor", False),
            "source_file": summary.get("source_file", ""),
            "md5": summary.get("md5", ""),
            "publisher": summary.get("publisher", ""),
            "publish_date": summary.get("publish_date", ""),
        }
        self.data["reports"].append(rep)
        self.save()
        return rep

    def delete_report(self, index):
        """按索引删除一份研报素材。"""
        if 0 <= index < len(self.data["reports"]):
            removed = self.data["reports"].pop(index)
            self.save()
            return removed
        return None

    def find_report_by_title(self, title, prefix_len=10):
        """按标题（前 N 字符）查找已存在的研报素材，用于录入前去重。"""
        key = (title or "").strip()[:prefix_len]
        if not key:
            return []
        return [r for r in self.data["reports"]
                if (r.get("title", "") or "").strip()[:prefix_len] == key]

    def find_report_by_file(self, filename, md5=None):
        """上传前查重：优先 MD5 内容指纹（最可靠），其次文件名归一化（兜底）。
        返回命中的研报列表；同一份 PDF 无论文件名怎么改、标题怎么翻译，MD5 都不变。"""
        import re
        if md5:
            hits = [r for r in self.data["reports"] if r.get("md5") and r.get("md5") == md5]
            if hits:
                return hits
        if filename:
            def _norm(s):
                s = (s or "").lower()
                s = re.sub(r'\.pdf$', '', s)
                s = re.sub(r'[\s_\-—·、，,。.（）()【】\[\]《》]', '', s)
                return s
            n = _norm(filename)
            if n:
                out = []
                for r in self.data["reports"]:
                    src = _norm(r.get("source_file", ""))
                    if src and (n in src or src in n):
                        out.append(r)
                return out
        return []

    def record(self, hypothesis, params, result, conclusion, tags):
        """记录一次实验。hypothesis=研究假设, params=参数, result=量化结果,
        conclusion=结论, tags=关键词（用于相似查询）。"""
        exp = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hypothesis": hypothesis,
            "params": params,
            "result": result,
            "conclusion": conclusion,
            "tags": tags,
        }
        self.data["experiments"].append(exp)
        self.save()
        return exp

    def find_similar(self, keyword, top_k=3):
        """按关键词查询相似历史实验（命中假设/结论/标签）。"""
        kw = keyword.lower()
        hits = []
        for e in self.data["experiments"]:
            text = " ".join([str(e.get("hypothesis", "")), str(e.get("conclusion", "")),
                             " ".join(e.get("tags", []))]).lower()
            if kw in text:
                hits.append(e)
        return hits[:top_k]

    def warn_if_explored(self, keyword):
        """避坑提示：命中历史『失败/无效』实验时返回提示（自我进化核心）。"""
        hits = self.find_similar(keyword)
        neg_words = ["不可复现", "无效", "失效", "反向", "失败", "未跑赢", "不显著"]
        return [e for e in hits if any(k in str(e.get("conclusion", "")) for k in neg_words)]


def main():
    mem = ResearchMemory()

    # 首次运行才记录演示数据（避免重复堆积）
    if not mem.data["experiments"]:
        mem.record(
            hypothesis="20日动量因子在A股ETF截面有效",
            params={"factor": "momentum", "lookback": 20, "top_k": 3},
            result={"annual_pct": -0.15, "sharpe": -0.10},
            conclusion="不可复现：A股ETF短期呈反转，动量失效",
            tags=["动量", "反转", "截面", "ETF"],
        )
        mem.record(
            hypothesis="5日反转因子有效",
            params={"factor": "reversal", "lookback": 5},
            result={"ic": 0.0835, "icir": 0.969},
            conclusion="有效：A股短期反转主导，可入因子库",
            tags=["反转", "有效", "IC"],
        )
        mem.record(
            hypothesis="情绪因子正向预测收益",
            params={"factor": "sentiment", "source": "全市场新闻"},
            result={"ic": -0.0907},
            conclusion="反向指标：情绪过热→未来回落，定位风控/过滤型",
            tags=["情绪", "反向", "风控"],
        )

    print("=" * 72)
    print("研究记忆库（自我进化模块）演示")
    print(f"已沉淀 {len(mem.data['experiments'])} 组实验结论，路径 {mem.path}")
    print("=" * 72)

    # 相似查询演示：当新任务要复现"动量"时，先查历史结论
    for kw in ["动量", "情绪", "反转"]:
        hits = mem.find_similar(kw)
        print(f"\n[查询「{kw}」] 命中 {len(hits)} 条历史经验：")
        for h in hits:
            print(f"  · {h['conclusion']}  (参数 {h['params']})")

    # 避坑提示：新任务命中历史失败实验时，自动提醒（避免重复探索）
    print("\n" + "=" * 72)
    print("避坑提示（自我进化：避免重复探索已证伪方向）")
    print("=" * 72)
    warn = mem.warn_if_explored("动量")
    if warn:
        for w in warn:
            print(f"  ⚠️ 「{w['hypothesis']}」已被验证：{w['conclusion']}")
            print(f"      → 建议调整假设或更换标的，无需重复探索")
    else:
        print("  （暂无命中失败实验）")


if __name__ == '__main__':
    main()
