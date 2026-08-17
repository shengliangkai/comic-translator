#!/usr/bin/env python3
"""漫画翻译助手 - 独立桌面版

一个零依赖（仅 Tkinter + 项目自带模块）的漫画翻译桌面小程序：
选图 -> 参数设置 -> 识别并翻译 -> 人工校对 -> 生成中文版成品。
"""

import os
import sys
import threading
import traceback
import uuid


def _fix_stdio():
    """GUI 可能在没有有效控制台的环境启动（双击 .py / IDE 隐藏控制台），
    此时 print 会抛 OSError([Errno 22])。检测到无效标准流就重定向到日志文件。"""
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "temp", "gui_console.log"
    )
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is None:
            try:
                setattr(sys, name, open(log_path, "a", encoding="utf-8"))
            except Exception:
                setattr(sys, name, open(os.devnull, "w"))
            continue
        try:
            stream.write("\n")
            stream.flush()
        except Exception:
            try:
                setattr(sys, name, open(log_path, "a", encoding="utf-8"))
            except Exception:
                setattr(sys, name, open(os.devnull, "w"))


_fix_stdio()

# 部分 Python 安装的 Tcl/Tk 不在标准搜索路径，需在导入 tkinter 前定位库文件
_base = sys.base_prefix
_tcl_dir = os.path.join(_base, "tcl", "tcl8.6")
_tk_dir = os.path.join(_base, "tcl", "tk8.6")
if os.path.isdir(_tcl_dir):
    os.environ.setdefault("TCL_LIBRARY", _tcl_dir)
if os.path.isdir(_tk_dir):
    os.environ.setdefault("TK_LIBRARY", _tk_dir)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config

_VENV_PY = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")


def _env_ready():
    try:
        import cv2  # noqa: F401
        import paddleocr  # noqa: F401

        return True
    except Exception:
        return False


def _relaunch_into_venv():
    """改用项目虚拟环境重启，保证始终使用经过验证的依赖环境。"""
    try:
        os.execv(_VENV_PY, [_VENV_PY] + sys.argv)
    except Exception as e:
        print(f"[环境] 自动切换到虚拟环境失败: {e}")


if os.environ.get("COMIC_GUI_TEST") == "1":
    print("[环境测试] python:", sys.executable)
    print("[环境测试] 虚拟环境存在:", os.path.isfile(_VENV_PY))
    print("[环境测试] 依赖完整:", _env_ready())
    sys.exit(0)

# 任何方式启动（双击 .py / IDE / bat）都统一切换到项目虚拟环境
if (
    os.path.isfile(_VENV_PY)
    and os.path.abspath(sys.executable).lower() != os.path.abspath(_VENV_PY).lower()
):
    _relaunch_into_venv()

try:
    import ctypes

    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

import cv2
import numpy as np
from PIL import Image, ImageTk

from modules.remover import TextRemover
from modules.processor import ComicTranslator
from modules.typesetter import Typesetter


class ComicTranslatorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("漫画翻译助手 Comic Translator")
        self.root.geometry("1100x760")
        self.root.minsize(900, 640)

        self.image_path = None
        self.image_tmp = None
        self.processor = None
        self.steps = None
        self._photo_refs = []

        self._build_ui()
        self.log("欢迎使用漫画翻译助手")
        self.log("第一步：选择漫画图片，设置参数后点击「识别并翻译」。")
        self.log("提示：首次真实识别会自动下载 OCR 模型（约几十 MB），之后离线可用。")

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        # 顶部：选图
        top = ttk.Frame(self.root, padding=(10, 8, 10, 2))
        top.pack(fill="x")
        ttk.Button(top, text="选择漫画图片", command=self.choose_image).pack(side="left")
        self.file_label = ttk.Label(top, text="未选择图片", foreground="#666666")
        self.file_label.pack(side="left", padx=10)

        # 参数区
        params = ttk.LabelFrame(self.root, text="参数设置", padding=8)
        params.pack(fill="x", padx=10, pady=4)

        ttk.Label(params, text="OCR 语言").grid(row=0, column=0, sticky="e", padx=4)
        self.ocr_lang = ttk.Combobox(
            params, values=["japan", "en", "korean", "ch"], state="readonly", width=10
        )
        self.ocr_lang.set("japan")
        self.ocr_lang.grid(row=0, column=1, padx=4)

        ttk.Label(params, text="翻译源语言").grid(row=0, column=2, sticky="e", padx=4)
        self.source_lang = ttk.Combobox(
            params, values=["auto", "ja", "en", "ko"], state="readonly", width=8
        )
        self.source_lang.set("auto")
        self.source_lang.grid(row=0, column=3, padx=4)

        ttk.Label(params, text="翻译引擎").grid(row=0, column=4, sticky="e", padx=4)
        self.engine = ttk.Combobox(
            params, values=["mock", "deepseek", "openai"], state="readonly", width=10
        )
        self.engine.set("mock")
        self.engine.grid(row=0, column=5, padx=4)

        ttk.Label(params, text="擦除方法").grid(row=0, column=6, sticky="e", padx=4)
        self.inpaint = ttk.Combobox(
            params, values=["telea", "ns"], state="readonly", width=8
        )
        self.inpaint.set("telea")
        self.inpaint.grid(row=0, column=7, padx=4)

        self.btn_run = ttk.Button(
            params, text="① 识别并翻译", command=self.run_ocr_translate
        )
        self.btn_run.grid(row=0, column=8, padx=(14, 4))
        self.btn_all = ttk.Button(
            params, text="一键翻译（①+②）", command=self.run_all
        )
        self.btn_all.grid(row=0, column=9, padx=(4, 4))

        ttk.Label(
            params,
            text="翻译引擎：mock=离线示例 | deepseek/openai=真实翻译（密钥可在下方填写，或写入项目 .env 文件）",
            foreground="#888888",
        ).grid(row=1, column=0, columnspan=9, sticky="w", padx=4, pady=(6, 0))

        ttk.Label(params, text="DeepSeek Key").grid(
            row=2, column=0, sticky="e", padx=4, pady=(8, 0)
        )
        self.deepseek_key_var = tk.StringVar(value=config.DEEPSEEK_API_KEY)
        ttk.Entry(
            params, textvariable=self.deepseek_key_var, show="*", width=32
        ).grid(row=2, column=1, columnspan=2, sticky="we", padx=4, pady=(8, 0))

        ttk.Label(params, text="OpenAI Key").grid(
            row=2, column=3, sticky="e", padx=4, pady=(8, 0)
        )
        self.openai_key_var = tk.StringVar(value=config.OPENAI_API_KEY)
        ttk.Entry(
            params, textvariable=self.openai_key_var, show="*", width=32
        ).grid(row=2, column=4, columnspan=2, sticky="we", padx=4, pady=(8, 0))

        ttk.Button(
            params, text="保存密钥到 .env", command=self.save_keys_to_env
        ).grid(row=2, column=6, columnspan=2, padx=4, pady=(8, 0))

        ttk.Label(params, text="识别精度").grid(
            row=3, column=0, sticky="e", padx=4, pady=(8, 0)
        )
        self.accuracy = ttk.Combobox(
            params, values=["standard", "high"], state="readonly", width=10
        )
        self.accuracy.set(config.OCR_ACCURACY)
        self.accuracy.grid(row=3, column=1, sticky="w", padx=4, pady=(8, 0))
        ttk.Label(
            params,
            text="标准=中等模型（约35秒/页）| 高精度=服务器级模型（约1-2分钟/页，小字更准）",
            foreground="#888888",
        ).grid(row=3, column=2, columnspan=7, sticky="w", padx=4, pady=(8, 0))

        # 中间：标签页
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=4)

        tab_original = ttk.Frame(self.notebook)
        self.original_label = self._make_image_label(tab_original)
        self.notebook.add(tab_original, text="原图")

        tab_preview = ttk.Frame(self.notebook)
        self.preview_label = self._make_image_label(tab_preview)
        self.notebook.add(tab_preview, text="OCR 预览")

        tab_edit = ttk.Frame(self.notebook)
        self._build_edit_table(tab_edit)
        self.notebook.add(tab_edit, text="人工校对")

        tab_final = ttk.Frame(self.notebook)
        side = ttk.Frame(tab_final)
        side.pack(fill="both", expand=True)
        self.cleaned_label = self._make_image_label(side)
        self.final_label = self._make_image_label(side)
        side.columnconfigure(0, weight=1)
        side.columnconfigure(1, weight=1)
        self.notebook.add(tab_final, text="成品")

        # 底部：状态 + 按钮
        bottom = ttk.Frame(self.root, padding=(10, 4))
        bottom.pack(fill="both")
        self.status = tk.Text(bottom, height=8, state="disabled", font=("Microsoft YaHei", 9))
        self.status.pack(fill="both", expand=True)

        btns = ttk.Frame(bottom)
        btns.pack(fill="x", pady=4)
        self.btn_generate = ttk.Button(
            btns, text="② 生成中文版成品", command=self.generate, state="disabled"
        )
        self.btn_generate.pack(side="left", padx=(0, 8))
        self.btn_save = ttk.Button(
            btns, text="保存成品图", command=self.save_output, state="disabled"
        )
        self.btn_save.pack(side="left")

    def _make_image_label(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        label = ttk.Label(frame, text="（图片将显示在这里）", anchor="center")
        label.pack(fill="both", expand=True)
        return label

    def _build_edit_table(self, parent):
        hint = ttk.Label(
            parent,
            text="双击任意行可修改「原文 / 译文」，修改后点击「生成中文版成品」。",
            foreground="#888888",
        )
        hint.pack(fill="x", padx=6, pady=(6, 2))
        columns = ("idx", "conf", "bbox", "src", "tr")
        self.tree = ttk.Treeview(parent, columns=columns, show="headings", height=16)
        headings = {
            "idx": ("序号", 50),
            "conf": ("置信度", 70),
            "bbox": ("坐标", 150),
            "src": ("原文", 260),
            "tr": ("译文", 260),
        }
        for col, (text, width) in headings.items():
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=6, pady=4)
        self.tree.bind("<Double-1>", self.edit_row)

    # ---------------------------------------------------------------- 工具
    def log(self, msg: str):
        self.status.config(state="normal")
        self.status.insert(tk.END, msg + "\n")
        self.status.see(tk.END)
        self.status.config(state="disabled")

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.btn_run.config(state=state)
        if busy:
            self.btn_run.config(text="处理中…")
        else:
            self.btn_run.config(text="① 识别并翻译")

    def _cv2_to_pil(self, img) -> Image.Image:
        if img is None:
            return None
        if len(img.shape) == 2:
            return Image.fromarray(img)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _show_image(self, label: ttk.Label, pil: Image.Image):
        if pil is None:
            label.config(text="（无图片）", image="")
            return
        w, h = label.winfo_width(), label.winfo_height()
        max_w = max(260, w)
        max_h = max(260, h)
        if w <= 1 or h <= 1:
            max_w, max_h = 900, 480
        pil = pil.copy()
        pil.thumbnail((max_w, max_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        label.config(image=photo, text="")
        self._photo_refs.append(photo)
        # 只保留最近引用，避免内存无限增长
        if len(self._photo_refs) > 12:
            self._photo_refs = self._photo_refs[-6:]

    def _populate_table(self, results):
        self.tree.delete(*self.tree.get_children())
        for r in results:
            bbox = r.get("bbox", (0, 0, 0, 0))
            self.tree.insert(
                "",
                tk.END,
                values=(
                    r.get("index", 0),
                    f"{r.get('confidence', 0):.0%}",
                    f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})",
                    r.get("text", ""),
                    r.get("translation", ""),
                ),
            )

    # ---------------------------------------------------------------- 动作
    def choose_image(self):
        path = filedialog.askopenfilename(
            title="选择漫画图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.webp *.bmp"), ("所有文件", "*.*")],
        )
        if not path:
            return
        self.image_path = path
        self.file_label.config(text=path)
        self.log(f"已选择图片: {path}")
        try:
            pil = Image.open(path)
            self._show_image(self.original_label, pil)
            self.notebook.select(0)
        except Exception as e:
            self.log(f"[错误] 无法打开图片: {e}")

    def run_ocr_translate(self):
        if not self.image_path:
            messagebox.showwarning("提示", "请先选择一张漫画图片")
            return
        self.auto_generate = False
        self._set_busy(True)
        self.log("开始识别并翻译……")
        threading.Thread(target=self._worker_ocr_translate, daemon=True).start()

    def run_all(self):
        """一键完成：识别翻译 -> 生成中文版成品。"""
        if not self.image_path:
            messagebox.showwarning("提示", "请先选择一张漫画图片")
            return
        self.auto_generate = True
        self._set_busy(True)
        self.log("一键翻译：开始识别并翻译……（完成后自动生成成品）")
        threading.Thread(target=self._worker_ocr_translate, daemon=True).start()

    def _apply_api_keys(self):
        ds = self.deepseek_key_var.get().strip()
        oa = self.openai_key_var.get().strip()
        if ds:
            config.DEEPSEEK_API_KEY = ds
            self.log(f"[配置] 已应用 DeepSeek API Key（{ds[:6]}****）")
        if oa:
            config.OPENAI_API_KEY = oa
            self.log(f"[配置] 已应用 OpenAI API Key（{oa[:6]}****）")

    def save_keys_to_env(self):
        ds = self.deepseek_key_var.get().strip()
        oa = self.openai_key_var.get().strip()
        env_path = os.path.join(BASE_DIR, ".env")
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        out = []
        found_ds = found_oa = False
        for line in lines:
            if line.startswith("DEEPSEEK_API_KEY="):
                out.append(f"DEEPSEEK_API_KEY={ds}")
                found_ds = True
            elif line.startswith("OPENAI_API_KEY="):
                out.append(f"OPENAI_API_KEY={oa}")
                found_oa = True
            else:
                out.append(line)
        if not found_ds:
            out.append(f"DEEPSEEK_API_KEY={ds}")
        if not found_oa:
            out.append(f"OPENAI_API_KEY={oa}")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        self.log("[配置] 密钥已保存到 .env 文件，重启后自动加载（密钥仅保存在本机）")

    def _worker_ocr_translate(self):
        try:
            self._apply_api_keys()
            # 中文路径处理：复制到临时目录（ASCII 文件名）
            ext = os.path.splitext(self.image_path)[1].lower() or ".png"
            self.image_tmp = os.path.join(
                config.TEMP_DIR, f"gui_{uuid.uuid4().hex[:12]}{ext}"
            )
            try:
                with Image.open(self.image_path) as im:
                    im.convert("RGB").save(self.image_tmp)
            except Exception:
                # 备用：直接用 OpenCV 字节流方式读取复制（兼容特殊路径/格式）
                buf = np.fromfile(self.image_path, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    raise
                cv2.imwrite(self.image_tmp, img)

            self.log("[进度] 图片已复制，正在初始化 OCR 模型……")
            processor = ComicTranslator(
                ocr_lang=self.ocr_lang.get(),
                translate_engine=self.engine.get(),
                inpaint_method=self.inpaint.get(),
                ocr_accuracy=self.accuracy.get(),
            )
            self.log("[进度] OCR 模型就绪，开始识别并翻译……")
            steps = processor.process(
                self.image_tmp,
                source_lang=self.source_lang.get(),
                target_lang="zh",
                return_steps=True,
            )
            self.processor = processor
            self.steps = steps
            self.root.after(0, self._on_ocr_done, steps)
        except Exception as e:
            tb = traceback.format_exc()
            self.root.after(0, self._on_worker_error, e, tb)

    def _on_ocr_done(self, steps):
        self._set_busy(False)
        results = steps.get("results", [])
        self._populate_table(results)
        self._show_image(self.preview_label, self._cv2_to_pil(steps.get("preview_image")))
        self.btn_generate.config(state="normal")
        self.log(
            f"[完成] OCR 识别 + 翻译完成，共 {len(results)} 个文本区域"
            + ("（Mock 演示模式，非真实识别！）" if self.processor.ocr.is_mock else "")
        )
        if self.processor.ocr.is_mock:
            err = getattr(self.processor.ocr, "init_error", None)
            reason = f"原因：{err}" if err else "原因见控制台输出"
            self.log(f"[警告] 未检测到可用 PaddleOCR，红框与识别文字为示例数据。{reason}")
        self.log("提示：请点击「② 生成中文版成品」（或使用「一键翻译」）把中文写入图片。")
        self.notebook.select(2)
        if getattr(self, "auto_generate", False):
            self.root.after(200, self.generate)

    def _on_worker_error(self, e, tb=None):
        self._set_busy(False)
        self.log(f"[错误] {e}")
        self.log(tb or "(无堆栈信息)")
        if "Errno 22" in str(e) or "Invalid argument" in str(e):
            self.log(
                "提示：此类错误常因图片被占用（OneDrive 同步/杀毒软件临时锁定）引起，"
                "请把图片先复制到本地文件夹再打开，或稍后重试。"
            )
        messagebox.showerror("处理失败", str(e))

    def edit_row(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            return
        values = self.tree.item(item, "values")
        try:
            idx = int(values[0])
        except Exception:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title(f"校对 #{idx}")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="原文").grid(row=0, column=0, sticky="nw", padx=8, pady=8)
        src_text = tk.Text(dialog, width=40, height=4)
        src_text.insert(tk.END, values[3])
        src_text.grid(row=0, column=1, padx=8, pady=8)

        ttk.Label(dialog, text="译文").grid(row=1, column=0, sticky="nw", padx=8, pady=8)
        tr_text = tk.Text(dialog, width=40, height=4)
        tr_text.insert(tk.END, values[4])
        tr_text.grid(row=1, column=1, padx=8, pady=8)

        def apply():
            new_src = src_text.get("1.0", tk.END).strip()
            new_tr = tr_text.get("1.0", tk.END).strip()
            if self.processor:
                self.processor.update_result(idx, text=new_src, translation=new_tr)
            for r in self.steps.get("results", []):
                if r.get("index") == idx:
                    r["text"] = new_src
                    r["translation"] = new_tr
                    break
            self.tree.item(
                item,
                values=(values[0], values[1], values[2], new_src, new_tr),
            )
            self.log(f"已更新 #{idx}: {new_tr}")
            dialog.destroy()

        ttk.Button(dialog, text="保存", command=apply).grid(
            row=2, column=1, sticky="e", padx=8, pady=8
        )
        dialog.resizable(False, False)

    def generate(self):
        if self.processor is None:
            messagebox.showwarning("提示", "请先执行「识别并翻译」")
            return
        self.btn_generate.config(state="disabled")
        self.btn_generate.config(text="生成中…")
        self.log("开始擦除原文并嵌字……")
        threading.Thread(target=self._worker_generate, daemon=True).start()

    def _worker_generate(self):
        try:
            processor = self.processor
            results = processor._last_results
            if not results:
                raise RuntimeError("没有可用的识别结果")

            remover = TextRemover(method=self.inpaint.get())
            boxes = [r.bbox for r in results]
            cleaned_cv, bubble_info = remover.remove_text(
                self.image_tmp, boxes, return_colors=True
            )
            info_map = {tuple(bi["bbox"]): bi for bi in bubble_info}
            for r in results:
                info = info_map.get(tuple(r.bbox))
                if info:
                    r.bg_color_rgb = tuple(info["bg_color_rgb"])
                    r.text_color_rgb = tuple(info.get("text_color_rgb", (0, 0, 0)))
                    r.is_white_bg = bool(info.get("is_white_bg", True))

            final_cv = processor.do_typeset(cleaned_cv, results)
            self.root.after(0, self._on_generate_done, cleaned_cv, final_cv)
        except Exception as e:
            tb = traceback.format_exc()
            self.root.after(0, self._on_worker_error, e, tb)

    def _on_generate_done(self, cleaned_cv, final_cv):
        self.btn_generate.config(state="normal", text="② 生成中文版成品")
        self.cleaned_cv = cleaned_cv
        self.final_cv = final_cv
        cleaned_pil = self._cv2_to_pil(cleaned_cv)
        final_pil = self._cv2_to_pil(final_cv)
        self._show_image(self.cleaned_label, cleaned_pil)
        self._show_image(self.final_label, final_pil)
        self.btn_save.config(state="normal")
        try:
            auto_path = os.path.join(config.OUTPUT_DIR, "latest_translated.png")
            cv2.imwrite(auto_path, final_cv)
            self.log(f"[信息] 成品已自动备份: {auto_path}")
            orig = cv2.imread(self.image_tmp)
            if orig is not None and orig.shape == final_cv.shape:
                changed = float((cv2.absdiff(orig, final_cv) > 20).mean())
                if changed < 0.005:
                    self.log(
                        "[警告] 成品与原图几乎一致，擦除/嵌字可能未生效，"
                        "请到「人工校对」检查识别结果后重新生成。"
                    )
        except Exception:
            pass
        self.log("[完成] 中文版成品已生成，可切换「成品」标签查看或保存。")
        self.notebook.select(3)

    def save_output(self):
        if self.final_label.cget("image") == "":
            messagebox.showwarning("提示", "还没有成品图")
            return
        default = os.path.join(
            config.OUTPUT_DIR,
            f"comic_translated_{uuid.uuid4().hex[:8]}.png",
        )
        path = filedialog.asksaveasfilename(
            title="保存成品图",
            initialdir=config.OUTPUT_DIR,
            initialfile=os.path.basename(default),
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("JPEG 图片", "*.jpg")],
        )
        if not path:
            return
        try:
            final_cv = getattr(self, "final_cv", None)
            if final_cv is None:
                final_cv = self.steps.get("final_image")
            final_pil = self._cv2_to_pil(final_cv)
            if final_pil is None:
                raise RuntimeError("final_image 不存在")
            final_pil.save(path)
            self.log(f"已保存: {path}")
            messagebox.showinfo("完成", f"成品已保存到:\n{path}")
        except Exception as e:
            self.log(f"[错误] 保存失败: {e}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    root = tk.Tk()
    ComicTranslatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
