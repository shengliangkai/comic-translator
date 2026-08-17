import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 在任何 PaddleOCR/PaddleX 导入之前设置好环境：
# - 优先使用项目内置模型缓存（离线可用，不依赖用户目录权限）
# - 跳过模型源联网检查；关闭 MKLDNN（Windows 兼容性）
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
_local_cache = os.path.join(BASE_DIR, "models", "paddlex")
if os.path.isdir(os.path.join(_local_cache, "official_models")):
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", _local_cache)

# 从项目根目录的 .env 文件加载密钥等配置（已存在的系统环境变量优先）
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)
except Exception:
    pass

FONT_DIR = os.path.join(BASE_DIR, "fonts")
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

for d in [FONT_DIR, INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

DEFAULT_FONT_REGULAR = os.path.join(FONT_DIR, "NotoSansCJK-Regular.ttc")
DEFAULT_FONT_BOLD = os.path.join(FONT_DIR, "NotoSansCJK-Bold.ttc")

DEFAULT_FONT_SIZE = 32
MIN_FONT_SIZE = 14
MAX_FONT_SIZE = 72

DEFAULT_TEXT_COLOR = (0, 0, 0)

OCR_LANG = "japan"
OCR_USE_GPU = False
OCR_ACCURACY = "standard"  # standard=中等模型(快) | high=服务器级模型(准,慢)

TRANSLATE_ENGINE = "mock"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"

INPAINT_RADIUS = 5
INPAINT_METHOD = "telea"

BBOX_PADDING = 8
