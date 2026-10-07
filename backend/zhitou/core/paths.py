# backend/app/core/paths.py
# =====================================================================
# 统一路径管理：所有数据/输出路径都基于项目根目录，避免相对路径依赖工作目录
# =====================================================================
import os

# __file__ = backend/app/core/paths.py
# 往上 4 层到项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_RAW = os.path.join(_PROJECT_ROOT, "data", "raw")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")
