# screenshot_streamlit.py
# =====================================================================
# Streamlit 演示 · 逐屏截图脚本（产出 PPT 可直接用的演示截图）
# 依赖：pip install playwright && playwright install chromium
# 用法：python screenshot_streamlit.py
# 产出：./output/screenshots/ 下 5 张 PNG
# =====================================================================
import subprocess
import time
import os
import sys
import urllib.request

PORT = 8501
BASE = f"http://localhost:{PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "screenshots")
os.makedirs(OUT, exist_ok=True)


def wait_server(timeout=90):
    """轮询等待 Streamlit 就绪。"""
    for _ in range(timeout * 2):
        try:
            urllib.request.urlopen(BASE, timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    # 1. 后台启动 Streamlit（避免与已启动实例冲突，先探测端口）
    proc = None
    try:
        urllib.request.urlopen(BASE, timeout=2)
        print("检测到 Streamlit 已在运行，直接复用。")
    except Exception:
        proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py",
             "--server.headless", "true", "--server.port", str(PORT)],
            cwd=HERE)
        print("已后台启动 Streamlit ...")
    try:
        if not wait_server():
            print("⚠️ Streamlit 未在 90 秒内就绪，请手动运行 streamlit run app.py 后重试。")
            return

        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 900})

            # ---- 屏 1：首页（① 因子复现，初始态） ----
            page.goto(BASE)
            page.wait_for_timeout(4000)
            page.screenshot(path=os.path.join(OUT, "01_因子复现_初始.png"))

            # ---- 屏 2：① 因子复现 运行结果 ----
            page.get_by_role("button", name="🚀 运行复现").click()
            page.wait_for_timeout(6000)
            page.screenshot(path=os.path.join(OUT, "02_因子复现_结果.png"))

            # ---- 屏 3：② 因子卡片评估 结果 ----
            page.get_by_text("② 因子卡片评估").click()
            page.wait_for_timeout(1500)
            page.get_by_role("button", name="📊 生成因子卡片").click()
            page.wait_for_timeout(5000)
            page.screenshot(path=os.path.join(OUT, "03_因子卡片_结果.png"))

            # ---- 屏 4：③ 自然语言智能体 初始（一句话输入） ----
            page.get_by_text("③ 自然语言智能体").click()
            page.wait_for_timeout(1500)
            page.screenshot(path=os.path.join(OUT, "04_智能体_初始.png"))

            # ---- 屏 5：③ 运行智能体 结果（需 DEEPSEEK_API_KEY） ----
            page.get_by_role("button", name="🤖 运行智能体").click()
            page.wait_for_timeout(10000)
            page.screenshot(path=os.path.join(OUT, "05_智能体_结果.png"))

            browser.close()
        print(f"✅ 截图完成，保存在 {OUT}")
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    main()
