import os
import pandas as pd
import re

EN_POSITIVE_WORDS = {
    "love", "happy", "joy", "great", "excellent", "good", "amazing", "wonderful", "delight", "pleasant",
    "excited", "sweet", "beautiful", "calm", "trust", "hope", "smile", "perfect", "brilliant", "fantastic"
}
EN_NEGATIVE_WORDS = {
    "hate", "sad", "angry", "awful", "terrible", "bad", "horrible", "pain", "fear", "disgust",
    "anxious", "depressed", "ugly", "annoyed", "worst", "cry", "broken", "toxic", "nervous", "frustrated"
}
ZH_POSITIVE_WORDS = {
    "开心", "快乐", "高兴", "幸福", "满意", "喜欢", "热爱", "激动", "温暖", "美好",
    "放松", "希望", "安心", "惊喜", "赞", "舒服", "甜", "治愈", "轻松", "感动"
}
ZH_NEGATIVE_WORDS = {
    "难过", "悲伤", "生气", "愤怒", "痛苦", "害怕", "焦虑", "失望", "讨厌", "崩溃",
    "烦", "糟糕", "压抑", "绝望", "恐惧", "恶心", "委屈", "心累", "窒息", "伤心"
}
CONCEPT_RELATION_PATTERNS = [
    ("is a", r"\bis\s+(an?|the)?\s*[a-z]+\b"),
    ("like", r"\blike\b"),
    ("as if", r"\bas if\b"),
    ("cause", r"\b(cause|causes|caused|because)\b"),
    ("part of", r"\bpart of\b"),
    ("used for", r"\bused for\b"),
    ("become", r"\bbecome\b"),
    ("是", r"是"),
    ("像", r"像"),
    ("仿佛", r"仿佛"),
    ("因为", r"因为"),
    ("导致", r"导致"),
    ("用于", r"用于"),
    ("变成", r"变成"),
]


def read_csv(data_dir, file_name):
    file_path = os.path.join(data_dir, file_name)
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='utf-8')


def store_data(data_obj: list, store_file: str, results_dir):
    full_path = os.path.join(results_dir, store_file)
    df = pd.DataFrame(data_obj)
    df.to_csv(full_path, index=False, encoding='utf-8')
    print(f"Data stored at: {full_path}")


def safe_parse_int(text, default=0):
    cleaned_text = re.sub(r'[^\d]', '', text)
    return int(cleaned_text) if cleaned_text.isdigit() else default


def extract_field(text, field_name, default="none"):
    if not isinstance(text, str):
        return default
    pattern = rf"{re.escape(field_name)}\s*:\s*(.*)"
    match = re.search(pattern, text)
    if not match:
        return default
    value = match.group(1).strip()
    if "\n" in value:
        value = value.split("\n", 1)[0].strip()
    return value if value else default


def build_emotion_lexicon_hint(sentence, raw_hint="none"):
    sentence_text = "" if sentence is None else str(sentence)
    lowered = sentence_text.lower()
    en_tokens = re.findall(r"[a-z]+", lowered)
    zh_tokens = re.findall(r"[\u4e00-\u9fff]+", sentence_text)
    zh_chars = []
    for token in zh_tokens:
        zh_chars.extend(list(token))

    en_pos_hits = [token for token in en_tokens if token in EN_POSITIVE_WORDS]
    en_neg_hits = [token for token in en_tokens if token in EN_NEGATIVE_WORDS]

    zh_pos_hits = [word for word in ZH_POSITIVE_WORDS if word in sentence_text]
    zh_neg_hits = [word for word in ZH_NEGATIVE_WORDS if word in sentence_text]
    if not zh_pos_hits and zh_chars:
        zh_pos_hits = [char for char in zh_chars if char in ZH_POSITIVE_WORDS]
    if not zh_neg_hits and zh_chars:
        zh_neg_hits = [char for char in zh_chars if char in ZH_NEGATIVE_WORDS]

    pos_count = len(en_pos_hits) + len(zh_pos_hits)
    neg_count = len(en_neg_hits) + len(zh_neg_hits)
    score = pos_count - neg_count
    total_hits = pos_count + neg_count
    intensity = 0.0 if total_hits == 0 else round(total_hits / max(len(en_tokens) + len(zh_chars), 1), 4)

    if score > 0:
        polarity = "positive"
    elif score < 0:
        polarity = "negative"
    else:
        polarity = "neutral"

    normalized_hint = "none" if raw_hint is None else str(raw_hint).strip()
    if not normalized_hint:
        normalized_hint = "none"
    lexicon_hits = list(dict.fromkeys(en_pos_hits + zh_pos_hits + en_neg_hits + zh_neg_hits))
    hit_text = ",".join(lexicon_hits[:8]) if lexicon_hits else "none"
    return (
        f"raw_hint={normalized_hint}; "
        f"lexicon_polarity={polarity}; "
        f"lexicon_score={score}; "
        f"lexicon_intensity={intensity}; "
        f"lexicon_hits={hit_text}"
    )


def build_conceptnet_hint(sentence):
    sentence_text = "" if sentence is None else str(sentence)
    lowered = sentence_text.lower()
    relation_hits = []
    for relation, pattern in CONCEPT_RELATION_PATTERNS:
        if re.search(pattern, lowered) or re.search(pattern, sentence_text):
            relation_hits.append(relation)
    relation_hits = list(dict.fromkeys(relation_hits))

    noun_candidates = re.findall(r"[a-z]{3,}", lowered)
    zh_candidates = re.findall(r"[\u4e00-\u9fff]{2,}", sentence_text)
    node_candidates = list(dict.fromkeys(noun_candidates[:4] + zh_candidates[:4]))

    if not relation_hits:
        relation_hits = ["none"]
    if not node_candidates:
        node_candidates = ["none"]
    return (
        f"concept_relations={','.join(relation_hits[:6])}; "
        f"concept_nodes={','.join(node_candidates[:6])}"
    )
