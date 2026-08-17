# 漫画翻译助手 Comic Translator

将漫画/条漫页面的日文（或英、韩文）翻译成中文：OCR 识别 -> 翻译 -> 擦除原文 -> 自动嵌字。

## 功能

- 真实 OCR：PaddleOCR（日语/英语/韩语/中文），支持竖排文本，可选「标准 / 高精度」两档模型
- 翻译引擎：Mock（离线示例）/ DeepSeek / OpenAI
- 两种入口：
  - 桌面小程序 `gui_app.py`（Tkinter，双击 `run_gui.bat` 启动）
  - 网页界面 `app.py`（Gradio，双击 `run_app.bat` 启动）
- 人工校对：识别结果可编辑（原文/译文），改完再生成成品
- 排版：自动适配气泡大小、按真实行高垂直居中、文字描边增强可读性

## 快速开始

1. 首次使用请先双击 `一键修复环境.bat`（创建虚拟环境并安装依赖，需联网）
2. 双击 `run_gui.bat` 启动桌面程序
3. 选择漫画图片 -> 点击「一键翻译」

> 说明：OCR 模型默认不在仓库内（体积较大）。首次真实识别会自动下载到项目 `models/` 目录，之后离线可用。

## 翻译引擎配置

复制 `.env.example` 为 `.env`，填入密钥：

```
DEEPSEEK_API_KEY=sk-xxx
OPENAI_API_KEY=sk-xxx
```

或在桌面程序界面的密钥输入框里填写并点击「保存密钥到 .env」。

## 命令行

```bash
python scripts/translate_comic.py <图片路径> --engine deepseek --accuracy high
```

参数：`--engine mock|deepseek|openai`、`--accuracy standard|high`、`--ocr-lang japan|en|korean|ch`、`--save-steps`、`--json 结果.json` 等。
