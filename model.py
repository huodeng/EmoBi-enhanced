import os
import time
import json
import re
import requests
from utils import extract_field, safe_parse_int
from config import (
    API_RETRIES,
    API_RETRY_BASE_SECONDS,
    API_CALL_DELAY_SECONDS,
    GROQ_API_URL,
    GROQ_API_KEY,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_ENV_PATH = os.path.join(SCRIPT_DIR, ".env.local")
_LOCAL_ENV_LOADED = False


def _load_local_env_once():
    global _LOCAL_ENV_LOADED
    if _LOCAL_ENV_LOADED:
        return
    _LOCAL_ENV_LOADED = True
    if not os.path.exists(LOCAL_ENV_PATH):
        return
    try:
        with open(LOCAL_ENV_PATH, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = str(raw_line).strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                env_key = str(key).strip()
                env_value = str(value).strip().strip('"').strip("'")
                if env_key and env_key not in os.environ:
                    os.environ[env_key] = env_value
    except Exception:
        # Keep runtime robust even if .env.local is malformed.
        return


def llm_chat(message, model_name, retries=API_RETRIES, timeout=60):
    _load_local_env_once()
    api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
    if not api_key:
        raise RuntimeError("Missing Groq API key. Set GROQ_API_KEY in environment, .env.local, or config.py.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0,
    }

    last_error = None
    for attempt in range(retries):
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
            if API_CALL_DELAY_SECONDS > 0:
                time.sleep(API_CALL_DELAY_SECONDS)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 429 and attempt < retries - 1:
                retry_after = exc.response.headers.get("Retry-After") if exc.response is not None else None
                try:
                    wait_seconds = int(float(retry_after)) if retry_after else None
                except ValueError:
                    wait_seconds = None
                if wait_seconds is None:
                    wait_seconds = API_RETRY_BASE_SECONDS * (2 ** attempt)
                print(f"Groq rate limit hit. Waiting {wait_seconds}s before retry {attempt + 2}/{retries}...")
                time.sleep(wait_seconds)
                continue
            break
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                wait_seconds = min(API_RETRY_BASE_SECONDS * (2 ** attempt), 120)
                print(f"Groq API request failed. Waiting {wait_seconds}s before retry {attempt + 2}/{retries}...")
                time.sleep(wait_seconds)
                continue
            break
    raise RuntimeError(f"Groq API request failed after {retries} attempts: {last_error}")

def analyze_emo(text, model_name, emotion_hint="none"):
    message = f"""
    Analyze the emotion valence of the following sentence and explain it.
    Also estimate how rhetorically intensified the emotion is.
    - 0.0 = completely plain factual statement, no emotional exaggeration at all
    - 0.5 = moderately emotional, could be literal or figurative
    - 1.0 = extremely exaggerated emotional expression, clearly beyond literal meaning
    Sentence: {text}
    Lexicon hint: {emotion_hint}
    Output format:
    Emotion analysis reason: <brief reason>
    Emotion intensity: <float between 0.0 and 1.0>
    """
    result_content = llm_chat(message, model_name)
    emotion_analysis_reason = extract_field(result_content, "Emotion analysis reason", default="none")
    raw_intensity = extract_field(result_content, "Emotion intensity", default="0.5")
    try:
        emotion_intensity = float(str(raw_intensity).strip())
        emotion_intensity = max(0.0, min(1.0, emotion_intensity))
    except Exception:
        emotion_intensity = 0.5
    return emotion_analysis_reason, emotion_intensity


def target_source_domains(text, emotion_analysis_reason, model_name, emotion_hint="none", concept_hint="none"):
    message = f"""
    Based on emotion valence, identify the target and source domains in the following sentence.
    If no metaphor can be identified, set all fields to 'none'.
    Sentence: {text}
    Emotion analysis reason: {emotion_analysis_reason}
    Lexicon hint: {emotion_hint}
    Concept hint: {concept_hint}
    Output format:
    Target domain: <target domain or 'none'>
    Select reason: <reason or 'none'>
    Source domain: <source domain or 'none'>
    Generate reason: <reason or 'none'>
    """
    result_content = llm_chat(message, model_name)
    target_domain = extract_field(result_content, "Target domain", default="none")
    select_reason = extract_field(result_content, "Select reason", default="none")
    source_domain = extract_field(result_content, "Source domain", default="none")
    generate_reason = extract_field(result_content, "Generate reason", default="none")
    return target_domain, select_reason, source_domain, generate_reason


def metaphor_learning(text, emotion_analysis_reason, target_domain, source_domain, model_name, concept_hint="none"):
    message = f"""
    You are judging whether a sentence contains metaphor.
    A sentence should be labeled as metaphor (1) when it uses one concept/domain to describe another concept/domain through implicit comparison or domain transfer, even if it is also colloquial, emotional, or slightly exaggerated.
    Do not require explicit markers like "like" or "as".
    If the wording is not meant to be interpreted literally and instead maps one domain onto another, prefer metaphor = 1.
    If the sentence is only strong emotion, literal emphasis, or plain exaggeration without cross-domain mapping, label metaphor = 0.
    Think carefully about idiomatic metaphorical expressions such as "cooked", "to the moon and back", "mountain out of a molehill", or "out of this world".
    Sentence: {text}
    Emotion analysis reason: {emotion_analysis_reason}
    Target domain: {target_domain}
    Source domain: {source_domain}
    Concept hint: {concept_hint}
    Decision hints:
    1. If target domain and source domain are both meaningful and different, that is strong evidence for metaphor = 1.
    2. If the sentence sounds odd literally but natural figuratively, prefer metaphor = 1.
    3. Be recall-oriented for possible metaphor, but do not mark clearly literal sentences as metaphor.
    Output format:
    Metaphor judgment: <0 or 1>
    Metaphor reason: <brief reason>
    """
    result_content = llm_chat(message, model_name)
    metaphor_judgment = safe_parse_int(extract_field(result_content, "Metaphor judgment", default="0"))
    metaphor_reason = extract_field(result_content, "Metaphor reason", default="none")
    return metaphor_judgment, metaphor_reason


def hyperbole_learning(text, metaphor_judgment, emotion_analysis_reason, target_domain,
                      source_domain, model_name, concept_hint="none", emotion_intensity=0.5):
    import os
    if os.environ.get("DISABLE_INTENSITY", "0") == "1":
        emotion_intensity = 0.5  # ablation: neutralize intensity signal

    if emotion_intensity >= 0.75:
        intensity_hint = (
            f"Emotion intensity is HIGH ({emotion_intensity:.2f}). "
            "The expression likely goes well beyond literal meaning — strongly consider hyperbole = 1."
        )
    elif emotion_intensity >= 0.45:
        intensity_hint = (
            f"Emotion intensity is MODERATE ({emotion_intensity:.2f}). "
            "The sentence may or may not be hyperbolic — weigh all cues carefully."
        )
    else:
        intensity_hint = (
            f"Emotion intensity is LOW ({emotion_intensity:.2f}). "
            "The sentence is likely a plain or factual statement — lean toward hyperbole = 0 unless strong evidence otherwise."
        )

    message = f"""
    Based on previous judgement and knowledge, re-evaluate if the sentence contains hyperbole.
    Sentence: {text}
    Metaphor judgment: {metaphor_judgment}
    Emotion analysis reason: {emotion_analysis_reason}
    Emotion intensity guidance: {intensity_hint}
    Target domain: {target_domain}
    Source domain: {source_domain}
    Concept hint: {concept_hint}
    Output format:
    Hyperbole judgment: <0 or 1>
    Hyperbole reason: <brief reason>
    """
    result_content = llm_chat(message, model_name)
    hyperbole_judgment = safe_parse_int(extract_field(result_content, "Hyperbole judgment", default="0"))
    hyperbole_reason = extract_field(result_content, "Hyperbole reason", default="none")

    return hyperbole_judgment, hyperbole_reason


def hyperbole_metaphor(text, emotion_analysis_reason, hyperbole_judgment, hyperbole_reason, metaphor_judgment,
                       metaphor_reason, target_domain, source_domain, model_name, concept_hint="none"):
    message = f"""
    You are a LLM with rich metaphor and hyperbole knowledge and strong judgment and reasoning ability. Please re-verify according to previous knowledge, including previous metaphor and exaggeration judgment and corresponding judgment reason, emotion and domain knowledge, to determine whether the current input is wrong. If your prediction is wrong, re-reason.
    Sentence: {text}
    Emotion analysis reason: {emotion_analysis_reason}
    Target domain: {target_domain}
    Source domain: {source_domain}
    Concept hint: {concept_hint}
    Previous hyperbole judgment: {hyperbole_judgment}
    Previous metaphor judgment: {metaphor_judgment}
    Previous hyperbole reason: {hyperbole_reason}
    Previous metaphor reason: {metaphor_reason}
    Output format:
    Hyperbole judgment: <0 or 1>
    Hyperbole reason: <brief reason>
    Metaphor judgment: <0 or 1>
    Metaphor reason: <brief reason>
    """
    result_content = llm_chat(message, model_name)

    hyperbole_judgment = safe_parse_int(extract_field(result_content, "Hyperbole judgment", default="0"))
    hyperbole_reason = extract_field(result_content, "Hyperbole reason", default="none")
    metaphor_judgment = safe_parse_int(extract_field(result_content, "Metaphor judgment", default="0"))
    metaphor_reason = extract_field(result_content, "Metaphor reason", default="none")

    return hyperbole_judgment, hyperbole_reason, metaphor_judgment, metaphor_reason


def _normalize_binary_label(value, default=0):
    parsed = safe_parse_int(str(value), default=default)
    return 1 if parsed >= 1 else 0


def _normalize_reason(value, default="none"):
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _normalize_confidence(value, default=0.0):
    try:
        confidence = float(value)
    except Exception:
        return default
    if confidence < 0:
        return 0.0
    if confidence > 1:
        return 1.0
    return confidence


def _parse_stage2_json(raw, hyperbole_judgment, hyperbole_reason, metaphor_judgment, metaphor_reason):
    stage2_parse_ok = 0
    confidence = 0.0
    parsed = {}
    try:
        parsed = json.loads(raw)
        stage2_parse_ok = 1
    except Exception:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                parsed = json.loads(match.group(0))
                stage2_parse_ok = 1
            except Exception:
                parsed = {}
    if stage2_parse_ok == 1:
        hyperbole_judgment = _normalize_binary_label(
            parsed.get("hyperbole_judgment", hyperbole_judgment), default=hyperbole_judgment
        )
        metaphor_judgment = _normalize_binary_label(
            parsed.get("metaphor_judgment", metaphor_judgment), default=metaphor_judgment
        )
        hyperbole_reason = _normalize_reason(parsed.get("hyperbole_reason", hyperbole_reason), default=hyperbole_reason)
        metaphor_reason = _normalize_reason(parsed.get("metaphor_reason", metaphor_reason), default=metaphor_reason)
        confidence = _normalize_confidence(parsed.get("confidence", 0.0), default=0.0)
    stage2_error = "none" if stage2_parse_ok == 1 else "json_parse_failed"
    return hyperbole_judgment, hyperbole_reason, metaphor_judgment, metaphor_reason, confidence, stage2_parse_ok, stage2_error


def stage2_json_refine(text, emotion_analysis_reason, target_domain, source_domain,
                       hyperbole_judgment, hyperbole_reason, metaphor_judgment, metaphor_reason, model_name):
    message = f"""
    Refine the final decision and output strict JSON only.
    Sentence: {text}
    Emotion analysis reason: {emotion_analysis_reason}
    Target domain: {target_domain}
    Source domain: {source_domain}
    Current hyperbole judgment: {hyperbole_judgment}
    Current hyperbole reason: {hyperbole_reason}
    Current metaphor judgment: {metaphor_judgment}
    Current metaphor reason: {metaphor_reason}
    Output JSON schema:
    {{
      "hyperbole_judgment": 0 or 1,
      "hyperbole_reason": "brief reason",
      "metaphor_judgment": 0 or 1,
      "metaphor_reason": "brief reason",
      "confidence": number from 0 to 1
    }}
    """
    raw = llm_chat(message, model_name).strip()
    return _parse_stage2_json(raw, hyperbole_judgment, hyperbole_reason, metaphor_judgment, metaphor_reason)


def fast_stage1_emo_domain(text, model_name, emotion_hint="none", concept_hint="none"):
    message = f"""
Analyze this sentence for emotion and domain mapping in one pass.

Sentence: {text}
Lexicon hint: {emotion_hint}
Concept hint: {concept_hint}

Output format (strict):
Emotion analysis reason: <brief reason>
Emotion intensity: <0.0-1.0, where 1.0=extremely exaggerated>
Target domain: <domain or 'none'>
Source domain: <domain or 'none'>
"""
    result_content = llm_chat(message, model_name)
    emotion_analysis_reason = extract_field(result_content, "Emotion analysis reason", default="none")
    raw_intensity = extract_field(result_content, "Emotion intensity", default="0.5")
    try:
        emotion_intensity = float(str(raw_intensity).strip())
        emotion_intensity = max(0.0, min(1.0, emotion_intensity))
    except Exception:
        emotion_intensity = 0.5
    target_domain = extract_field(result_content, "Target domain", default="none")
    source_domain = extract_field(result_content, "Source domain", default="none")
    if target_domain == "none" and source_domain == "none":
        select_reason = "none"
        generate_reason = "none"
    else:
        select_reason = "fast-mode combined"
        generate_reason = "fast-mode combined"
    return emotion_analysis_reason, emotion_intensity, target_domain, select_reason, source_domain, generate_reason


def fast_stage2_meta_hyper(text, emotion_analysis_reason, target_domain, source_domain,
                           emotion_intensity, model_name, concept_hint="none"):
    if emotion_intensity >= 0.75:
        intensity_tag = "HIGH"
    elif emotion_intensity >= 0.45:
        intensity_tag = "MODERATE"
    else:
        intensity_tag = "LOW"

    message = f"""
Judge both metaphor AND hyperbole for this sentence in one pass.
Use domain knowledge and emotion cues. Output strict JSON only.

Sentence: {text}
Emotion: {emotion_analysis_reason} (intensity={intensity_tag}, {emotion_intensity:.2f})
Target domain: {target_domain}
Source domain: {source_domain}
Concept hint: {concept_hint}

Rules:
- Metaphor=1 if cross-domain mapping exists (e.g. "time is money"), even if colloquial
- Hyperbole=1 if emotion is clearly exaggerated beyond literal meaning
- Metaphor=1 can coexist with Hyperbole=1
- If target=source=none, both should be 0 unless strong contrary evidence

Output JSON:
{{"metaphor_judgment": 0 or 1, "metaphor_reason": "brief", "hyperbole_judgment": 0 or 1, "hyperbole_reason": "brief", "confidence": 0.0-1.0}}
"""
    raw = llm_chat(message, model_name).strip()
    parsed = {}
    try:
        parsed = json.loads(raw)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = {}
    metaphor_judgment = safe_parse_int(str(parsed.get("metaphor_judgment", 0)))
    metaphor_reason = str(parsed.get("metaphor_reason", "none")) or "none"
    hyperbole_judgment = safe_parse_int(str(parsed.get("hyperbole_judgment", 0)))
    hyperbole_reason = str(parsed.get("hyperbole_reason", "none")) or "none"
    try:
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except Exception:
        confidence = 0.5
    return metaphor_judgment, metaphor_reason, hyperbole_judgment, hyperbole_reason, confidence


def analyze_batch(texts, model_name, emotion_hints=None, concept_hints=None):
    if not texts:
        return []
    n = len(texts)
    if emotion_hints is None:
        emotion_hints = ["none"] * n
    if concept_hints is None:
        concept_hints = ["none"] * n
    sentences_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
    hints_block = "\n".join([f"Hint for sentence {i+1}: emotion={emotion_hints[i]}, concept={concept_hints[i]}" for i in range(n)])
    message = f"""You are an expert at analyzing emotion, metaphor, and hyperbole in text.

Task: For each sentence below, provide emotion analysis, domain mapping, and judgment for both metaphor and hyperbole.

{hints_block}

Sentences:
{sentences_block}

Output format for EACH sentence - output ONLY the line below, nothing else:
Sentence {{i}}: Emotion=<reason>, Target=<domain>, Source=<domain>, Metaphor=<0 or 1>, Hyperbole=<0 or 1>

Example output:
Sentence 1: Emotion=Frustration, Target=Hunger, Source=None, Metaphor=1, Hyperbole=1
Sentence 2: Emotion=Suspicion, Target=Deception, Source=None, Metaphor=1, Hyperbole=0

You must output EXACTLY {n} lines, one for each sentence."""
    result_content = llm_chat(message, model_name)
    results = []
    for i in range(n):
        pattern = rf"Sentence\s*{i+1}\s*:\s*.*Emotion\s*=\s*([^,]+).*Target\s*=\s*([^,]+).*Source\s*=\s*([^,]+).*Metaphor\s*=\s*(\d+).*Hyperbole\s*=\s*(\d+)"
        match = re.search(pattern, result_content, re.IGNORECASE | re.DOTALL)
        if match:
            emotion = match.group(1).strip()
            target = match.group(2).strip()
            source = match.group(3).strip()
            metaphor = safe_parse_int(match.group(4), default=0)
            hyperbole = safe_parse_int(match.group(5), default=0)
            results.append({
                "emotion_reason": emotion,
                "target_domain": target,
                "source_domain": source,
                "metaphor_judgment": metaphor,
                "hyperbole_judgment": hyperbole
            })
        else:
            results.append({
                "emotion_reason": "none",
                "target_domain": "none",
                "source_domain": "none",
                "metaphor_judgment": 0,
                "hyperbole_judgment": 0
            })
    return results


def verify_batch(samples, model_name, threshold=0.85):
    low_conf_samples = [(i, s) for i, s in enumerate(samples) if s.get("confidence", 1.0) < threshold]
    if not low_conf_samples:
        return samples
    indices = [i for i, _ in low_conf_samples]
    texts = [s["text"] for s in low_conf_samples]
    n = len(texts)
    sentences_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
    contexts_block = "\n".join([
        f"{i+1}. Emotion: {low_conf_samples[i][1].get('emotion_reason','none')}, "
        f"Target: {low_conf_samples[i][1].get('target_domain','none')}, "
        f"Metaphor: {low_conf_samples[i][1].get('metaphor_judgment',0)}, "
        f"Hyperbole: {low_conf_samples[i][1].get('hyperbole_judgment',0)}"
        for i in range(n)
    ])
    message = f"""Review and verify the following predictions. If wrong, correct them.

Sentences:
{sentences_block}

Current predictions:
{contexts_block}

Output format for EACH sentence that needs verification (mandatory):
[SENT{{index}}]
Final Metaphor: <0 or 1>
Final Hyperbole: <0 or 1>
Reason: <if changed, explain briefly>
[/SENT{{index}}]

You must output results for ALL {n} sentences."""
    result_content = llm_chat(message, model_name)
    for i in range(n):
        pattern = rf"\[SENT{i+1}\]([\s\S]*?)\[/SENT{i+1}\]"
        match = re.search(pattern, result_content, re.IGNORECASE)
        if match:
            block = match.group(1)
            metaphor = safe_parse_int(extract_field(block, "Final Metaphor", default=str(samples[indices[i]].get("metaphor_judgment", 0))))
            hyperbole = safe_parse_int(extract_field(block, "Final Hyperbole", default=str(samples[indices[i]].get("hyperbole_judgment", 0))))
            samples[indices[i]]["metaphor_judgment"] = metaphor
            samples[indices[i]]["hyperbole_judgment"] = hyperbole
            samples[indices[i]]["verified"] = True
    return samples


def stage2_json_refine_contrast(text, emotion_analysis_reason, target_domain, source_domain,
                                hyperbole_judgment, hyperbole_reason, metaphor_judgment, metaphor_reason, model_name):
    message = f"""
    Refine the final decision again with different evidence focus and output strict JSON only.
    You must use a different evidence focus from the previous reason and avoid repeating wording.
    Sentence: {text}
    Emotion analysis reason: {emotion_analysis_reason}
    Target domain: {target_domain}
    Source domain: {source_domain}
    Previous hyperbole judgment: {hyperbole_judgment}
    Previous hyperbole reason: {hyperbole_reason}
    Previous metaphor judgment: {metaphor_judgment}
    Previous metaphor reason: {metaphor_reason}
    Output JSON schema:
    {{
      "hyperbole_judgment": 0 or 1,
      "hyperbole_reason": "brief reason with new evidence",
      "metaphor_judgment": 0 or 1,
      "metaphor_reason": "brief reason with new evidence",
      "confidence": number from 0 to 1
    }}
    """
    raw = llm_chat(message, model_name).strip()
    return _parse_stage2_json(raw, hyperbole_judgment, hyperbole_reason, metaphor_judgment, metaphor_reason)
