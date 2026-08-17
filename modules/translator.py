import os
import sys
import json
import time
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Translator:
    def __init__(self, engine: str = None):
        self.engine = engine or config.TRANSLATE_ENGINE
        self._mock_dict = {
            "こんにちは": "你好",
            "こんにちは！": "你好！",
            "こんにちは!": "你好!",
            "何をしているの？": "你在做什么？",
            "何をしているの?": "你在做什么?",
            "何をしてるの？": "你在干嘛？",
            "ありがとう": "谢谢",
            "ありがとう！": "谢谢！",
            "ありがとうございます": "非常感谢",
            "すみません": "对不起",
            "すみません！": "对不起！",
            "ごめんなさい": "抱歉",
            "はい": "是的",
            "いいえ": "不是",
            "わかった": "明白了",
            "わかりました": "我知道了",
            "おはよう": "早上好",
            "おはようございます": "早安",
            "おやすみ": "晚安",
            "おやすみなさい": "晚安",
            "ただいま": "我回来了",
            "おかえり": "欢迎回来",
            "おかえりなさい": "欢迎回来",
            "いただきます": "我开动了",
            "ごちそうさま": "多谢款待",
            "さようなら": "再见",
            "じゃあね": "拜拜",
            "またね": "回头见",
            "大好き": "最喜欢了",
            "愛してる": "我爱你",
            "可愛い": "可爱",
            "綺麗": "漂亮",
            "すごい": "厉害",
            "すごい！": "厉害！",
            "本当？": "真的吗？",
            "マジで？": "真的假的？",
            "冗談だよ": "开玩笑的",
            "どういたしまして": "不客气",
            "助けて": "救命",
            "危ない": "危险",
            "やめて": "不要",
            "やめて！": "不要！",
            "もう一度": "再来一次",
            "大丈夫": "没关系",
            "大丈夫ですか？": "你没事吧？",
            "元気ですか？": "你好吗？",
            "はじめまして": "初次见面",
            "よろしくお願いします": "请多关照",
            "お疲れ様です": "辛苦了",
            "いってらっしゃい": "一路顺风",
            "いってきます": "我走了",
        }

    def _mock_translate(self, text: str) -> str:
        if not text:
            return ""
        if text in self._mock_dict:
            return self._mock_dict[text]
        stripped = text.rstrip("。！？!?、，,")
        for k, v in self._mock_dict.items():
            k_stripped = k.rstrip("。！？!?、，,")
            if k_stripped == stripped:
                suffix = text[len(stripped) :] if len(text) > len(stripped) else ""
                return v + suffix
        try:
            ja_common = {
                "の": "的", "に": "在", "は": "是", "を": "把", "で": "在/用",
                "と": "和", "から": "从", "まで": "到", "が": "（主格）",
                "する": "做", "して": "在做", "です": "是", "ます": "（礼貌形）",
                "ない": "不", "た": "了", "だ": "是",
                "私": "我", "あなた": "你", "彼": "他", "彼女": "她",
                "君": "你", "俺": "我", "僕": "我",
                "これ": "这个", "それ": "那个", "あれ": "那个",
                "ここ": "这里", "そこ": "那里",
                "今": "现在", "今日": "今天", "明日": "明天", "昨日": "昨天",
                "人": "人", "日": "天", "年": "年", "月": "月",
                "行く": "去", "来る": "来", "見る": "看", "聞く": "听",
                "食べる": "吃", "飲む": "喝", "買う": "买", "作る": "做",
                "いい": "好", "悪い": "坏", "大きい": "大", "小さい": "小",
                "新しい": "新", "古い": "旧", "多い": "多", "少ない": "少",
                "早い": "早", "遅い": "晚", "速い": "快", "遅い": "慢",
                "高い": "高/贵", "安い": "便宜", "安い": "便宜", "低い": "低",
                "強い": "强", "弱い": "弱", "重い": "重", "軽い": "轻",
                "先生": "老师", "学生": "学生", "友達": "朋友",
                "本": "书", "手": "手", "目": "眼睛", "顔": "脸",
                "心": "心", "夢": "梦", "愛": "爱", "恋": "恋",
                "名前": "名字", "時間": "时间", "場所": "地方",
                "どうして": "为什么", "なぜ": "为什么",
                "どうやって": "怎么做", "いつ": "什么时候",
                "どこ": "哪里", "誰": "谁", "何": "什么",
                "どれ": "哪个", "どう": "怎样", "いくら": "多少钱",
                "わたし": "我", "あたし": "我",
                "すごい": "厉害", "かわいい": "可爱", "きれい": "漂亮",
                "うまい": "好吃/棒", "おいしい": "好吃",
                "たのしい": "开心", "つまらない": "无聊",
                "かっこいい": "帅气", "やばい": "不妙/超棒",
                "まさか": "难道", "どうしよう": "怎么办",
                "ほんとう": "真的", "本当": "真的", "嘘": "骗人",
                "ええ": "嗯", "うん": "嗯", "ああ": "啊",
                "なるほど": "原来如此", "そうか": "是吗",
                "待って": "等等", "ちょっと": "稍微/等一下",
                "もう": "已经", "まだ": "还没",
                "すぐ": "马上", "とても": "非常", "ちゃんと": "好好地",
                "少し": "一点", "沢山": "很多", "たくさん": "很多",
            }
            translated_chars = []
            for ch in text:
                if ch in ja_common:
                    translated_chars.append(ja_common[ch])
                elif "\u3040" <= ch <= "\u309f":
                    translated_chars.append(ch)
                elif "\u30a0" <= ch <= "\u30ff":
                    translated_chars.append(ch)
                elif "\u4e00" <= ch <= "\u9fff":
                    translated_chars.append(ch)
                else:
                    translated_chars.append(ch)
            return "".join(translated_chars)
        except Exception:
            return f"(未翻译) {text}"

    def _call_deepseek(self, text: str, source_lang: str = "auto", target_lang: str = "zh") -> str:
        if not config.DEEPSEEK_API_KEY:
            print("[翻译 警告] 未配置 DEEPSEEK_API_KEY，回退到 Mock 翻译")
            return self._mock_translate(text)

        import requests

        try:
            lang_prompt = ""
            if source_lang == "ja":
                lang_prompt = "日语"
            elif source_lang == "en":
                lang_prompt = "英语"
            elif source_lang == "ko":
                lang_prompt = "韩语"
            else:
                lang_prompt = "源语言"

            messages = [
                {
                    "role": "system",
                    "content": (
                        f"你是专业的漫画翻译助手，将{lang_prompt}翻译成自然流畅的中文。"
                        "要求：1. 符合漫画对白口语化风格；2. 保持语气和情感；3. 译文要简洁适合气泡排版；"
                        "4. 只输出翻译结果，不要加任何解释、引号或其他文字。"
                    ),
                },
                {"role": "user", "content": f"翻译：{text}"},
            ]

            resp = requests.post(
                config.DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.DEEPSEEK_MODEL,
                    "messages": messages,
                    "temperature": 0.6,
                    "max_tokens": 1024,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[翻译 错误] DeepSeek 调用失败: {e}，回退到 Mock 翻译")
            return self._mock_translate(text)

    def _call_openai(self, text: str, source_lang: str = "auto", target_lang: str = "zh") -> str:
        if not config.OPENAI_API_KEY:
            print("[翻译 警告] 未配置 OPENAI_API_KEY，回退到 Mock 翻译")
            return self._mock_translate(text)

        import requests

        try:
            lang_prompt = "源语言"
            if source_lang == "ja":
                lang_prompt = "日语"
            elif source_lang == "en":
                lang_prompt = "英语"
            elif source_lang == "ko":
                lang_prompt = "韩语"

            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a professional comic translator. Translate the following {lang_prompt} "
                        "text into natural, conversational Chinese suitable for comic speech bubbles. "
                        "Output ONLY the translation, no explanations, no quotes."
                    ),
                },
                {"role": "user", "content": text},
            ]

            resp = requests.post(
                config.OPENAI_API_URL,
                headers={
                    "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.OPENAI_MODEL,
                    "messages": messages,
                    "temperature": 0.6,
                    "max_tokens": 1024,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[翻译 错误] OpenAI 调用失败: {e}，回退到 Mock 翻译")
            return self._mock_translate(text)

    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "zh") -> str:
        if not text or not str(text).strip():
            return ""
        text = str(text).strip()

        if self.engine == "mock":
            return self._mock_translate(text)
        elif self.engine == "deepseek":
            return self._call_deepseek(text, source_lang, target_lang)
        elif self.engine == "openai":
            return self._call_openai(text, source_lang, target_lang)
        else:
            return self._mock_translate(text)

    def translate_batch(
        self, texts: List[str], source_lang: str = "auto", target_lang: str = "zh"
    ) -> List[str]:
        results = []
        for i, t in enumerate(texts):
            if i > 0 and self.engine in ("deepseek", "openai"):
                time.sleep(0.3)
            results.append(self.translate(t, source_lang, target_lang))
        return results
