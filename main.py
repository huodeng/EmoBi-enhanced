import argparse
import os
import re
import pandas as pd
from tqdm import tqdm
from utils import read_csv, store_data, build_emotion_lexicon_hint, build_conceptnet_hint
from model import analyze_emo, target_source_domains, metaphor_learning, hyperbole_learning, hyperbole_metaphor, stage2_json_refine, stage2_json_refine_contrast, analyze_batch, verify_batch, fast_stage1_emo_domain, fast_stage2_meta_hyper
from config import MODEL_NAME, INPUT_NAME, MAX_SAMPLES, RESULTS_DIR, RESUME, CHECKPOINT_INTERVAL

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "multitask_hyperbole_metaphor_detection-main/data"))
    parser.add_argument("--input-name", default=os.getenv("INPUT_NAME", INPUT_NAME))
    parser.add_argument("--results-dir", default=os.getenv("RESULTS_DIR_OVERRIDE", os.path.join(RESULTS_DIR, "runs")))
    parser.add_argument("--run-name", default=os.getenv("RUN_NAME", ""))
    parser.add_argument("--max-samples", type=int, default=int(os.getenv("MAX_SAMPLES", MAX_SAMPLES)))
    parser.add_argument("--ablation-profile", default="full")
    parser.add_argument("--use-emotion-lexicon-feature", type=int, default=1)
    parser.add_argument("--use-conceptnet-feature", type=int, default=1)
    parser.add_argument("--stage2-conf-threshold", type=float, default=0.85)
    parser.add_argument("--stage2-max-retry", type=int, default=1)
    parser.add_argument("--stage2-retry-mode", default="contrast")
    parser.add_argument("--stage2-retry-target", default="auto")
    parser.add_argument("--stage2-rule-filter", type=int, default=1)
    parser.add_argument("--stage2-metaphor-none-domain-max-conf", type=float, default=0.55)
    parser.add_argument("--stage2-consensus-enable", type=int, default=1)
    parser.add_argument("--stage2-consensus-threshold", type=float, default=0.75)
    parser.add_argument("--stage2-consensus-mode", default="contrast")
    parser.add_argument("--stage2-aggregate-enable", type=int, default=1)
    parser.add_argument("--stage2-aggregate-min-valid-candidates", type=int, default=2)
    parser.add_argument("--stage2-bootstrap-contrast-enable", type=int, default=1)
    parser.add_argument("--stage2-ensemble-support-threshold", type=float, default=0.62)
    parser.add_argument("--stage2-low-support-refine-enable", type=int, default=1)
    parser.add_argument("--stage2-reaggregate-min-support-gain", type=float, default=0.01)
    parser.add_argument("--stage2-reaggregate-adaptive-gain-enable", type=int, default=1)
    parser.add_argument("--stage2-reaggregate-adaptive-gain-step", type=float, default=0.005)
    parser.add_argument("--stage2-reaggregate-support-gap-link-enable", type=int, default=1)
    parser.add_argument("--stage2-reaggregate-support-gap-link-factor", type=float, default=0.05)
    parser.add_argument("--stage2-reaggregate-support-gap-mid", type=float, default=0.08)
    parser.add_argument("--stage2-reaggregate-support-gap-high", type=float, default=0.16)
    parser.add_argument("--stage2-reaggregate-support-gap-mid-multiplier", type=float, default=1.5)
    parser.add_argument("--stage2-reaggregate-support-gap-high-multiplier", type=float, default=2.0)
    parser.add_argument("--stage2-reaggregate-candidate-mid", type=int, default=4)
    parser.add_argument("--stage2-reaggregate-candidate-high", type=int, default=6)
    parser.add_argument("--stage2-reaggregate-candidate-mid-multiplier", type=float, default=1.1)
    parser.add_argument("--stage2-reaggregate-candidate-high-multiplier", type=float, default=1.25)
    parser.add_argument("--stage2-retry-reason-alert-threshold", type=float, default=0.05)
    parser.add_argument("--stage3-enable", type=int, default=0)
    parser.add_argument("--stage3-trigger-mode", default="conflict_or_low_conf")
    parser.add_argument("--stage3-low-conf-threshold", type=float, default=0.60)
    parser.add_argument("--stage3-image-columns", default="Image path,image_path,Image,image,Frame path,frame_path")
    parser.add_argument("--stage3-audio-columns", default="Audio path,audio_path,Audio,audio,Wav path,wav_path")
    parser.add_argument("--stage3-video-columns", default="Video path,video_path,Video,video,Clip path,clip_path")
    parser.add_argument("--stage3-media-check-exists", type=int, default=1)
    parser.add_argument("--stage3-media-base-dir", default="")
    parser.add_argument("--stage3-media-extra-base-dirs", default="")
    parser.add_argument("--stage3-media-resolve-order", default="raw,base,data,extra")
    parser.add_argument("--stage3-media-path-from", default="")
    parser.add_argument("--stage3-media-path-to", default="")
    parser.add_argument("--stage3-media-path-map", default="")
    parser.add_argument("--resume-enable", type=int, default=1 if bool(RESUME) else 0)
    parser.add_argument("--checkpoint-interval", type=int, default=int(os.getenv("CHECKPOINT_INTERVAL", CHECKPOINT_INTERVAL)))
    parser.add_argument("--use-batch-inference", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--fast", type=int, default=0, help="Fast mode: 2 LLM calls/sentence (default 0 = 5-step pipeline)")
    return parser.parse_args()


RETRY_TARGET_REASON_EXPLICIT = "explicit"
RETRY_TARGET_REASON_AUTO_HYPOL = "auto:hypol"
RETRY_TARGET_REASON_AUTO_DEFAULT = "auto:default"
RETRY_TARGET_REASON_ALLOWED = {
    RETRY_TARGET_REASON_EXPLICIT,
    RETRY_TARGET_REASON_AUTO_HYPOL,
    RETRY_TARGET_REASON_AUTO_DEFAULT
}


def normalize_retry_target_resolution_reason(reason):
    normalized = str(reason).strip().lower()
    if normalized in RETRY_TARGET_REASON_ALLOWED:
        return normalized
    return RETRY_TARGET_REASON_AUTO_DEFAULT


def stage1_counterfactual_filter(text):
    if not isinstance(text, str) or not text.strip():
        return 0, "empty sentence"
    lowered = text.lower()
    keyword_cues = [
        "always", "never", "everyone", "nobody", "all", "none",
        "literally", "totally", "absolutely", "million", "billion",
        "like", "as if", "as ... as", "像", "仿佛", "简直", "永远", "根本", "完全", "亿"
    ]
    matched = [kw for kw in keyword_cues if kw in lowered or kw in text]
    if re.search(r"\d{3,}", text):
        matched.append("large-number")
    if "!" in text or "！" in text:
        matched.append("exclamation")
    if matched:
        return 1, f"cue matched: {matched[0]}"
    return 1, "fallback pass for high recall"


def resolve_retry_target(args):
    if args.stage2_retry_target in {"all", "metaphor_zero"}:
        return args.stage2_retry_target, RETRY_TARGET_REASON_EXPLICIT
    input_name = str(args.input_name).lower()
    if "hypo-l" in input_name or "hypol" in input_name:
        return "all", RETRY_TARGET_REASON_AUTO_HYPOL
    return "metaphor_zero", RETRY_TARGET_REASON_AUTO_DEFAULT


def apply_ablation_profile(args):
    profile = str(args.ablation_profile).strip().lower()
    allowed = {"full", "wo_feature", "plus_e", "plus_c", "wo_cascade"}
    if profile not in allowed:
        profile = "full"
    args.ablation_profile = profile
    if profile == "wo_feature":
        args.use_emotion_lexicon_feature = 0
        args.use_conceptnet_feature = 0
    elif profile == "plus_e":
        args.use_emotion_lexicon_feature = 1
        args.use_conceptnet_feature = 0
    elif profile == "plus_c":
        args.use_emotion_lexicon_feature = 0
        args.use_conceptnet_feature = 1
    else:
        args.use_emotion_lexicon_feature = int(args.use_emotion_lexicon_feature)
        args.use_conceptnet_feature = int(args.use_conceptnet_feature)
    if profile == "wo_cascade":
        args.stage2_max_retry = 0
        args.stage2_rule_filter = 0
        args.stage2_consensus_enable = 0
        args.stage2_aggregate_enable = 0
        args.stage2_bootstrap_contrast_enable = 0
        args.stage2_low_support_refine_enable = 0
    return args


def should_retry_stage2(effective_retry_target, stage2_confidence, stage2_metaphor_judgment, stage2_conf_threshold):
    if stage2_confidence >= stage2_conf_threshold:
        return False
    if effective_retry_target == "all":
        return True
    if effective_retry_target == "metaphor_zero":
        return int(stage2_metaphor_judgment) == 0
    return True


def apply_stage2_rules(args, target_domain, source_domain, stage2_hyperbole_judgment, stage2_hyperbole_reason,
                       stage2_metaphor_judgment, stage2_metaphor_reason, stage2_confidence):
    rule_applied = 0
    if int(args.stage2_rule_filter) != 1:
        return stage2_hyperbole_judgment, stage2_hyperbole_reason, stage2_metaphor_judgment, stage2_metaphor_reason, rule_applied
    target_none = str(target_domain).strip().lower() in {"", "none", "null", "na", "n/a"}
    source_none = str(source_domain).strip().lower() in {"", "none", "null", "na", "n/a"}
    if target_none and source_none and int(stage2_metaphor_judgment) == 1 and float(stage2_confidence) < args.stage2_metaphor_none_domain_max_conf:
        stage2_metaphor_judgment = 0
        stage2_metaphor_reason = "none-domain low-confidence override"
        rule_applied = 1
    return stage2_hyperbole_judgment, stage2_hyperbole_reason, stage2_metaphor_judgment, stage2_metaphor_reason, rule_applied


def stage2_consensus_refine(args, text, emotion_analysis_reason, target_domain, source_domain,
                            stage2_hyperbole_judgment, stage2_hyperbole_reason, stage2_metaphor_judgment,
                            stage2_metaphor_reason, stage2_confidence):
    if int(args.stage2_consensus_enable) != 1:
        return stage2_hyperbole_judgment, stage2_hyperbole_reason, stage2_metaphor_judgment, stage2_metaphor_reason, stage2_confidence, 0, 0
    if float(stage2_confidence) >= float(args.stage2_consensus_threshold):
        return stage2_hyperbole_judgment, stage2_hyperbole_reason, stage2_metaphor_judgment, stage2_metaphor_reason, stage2_confidence, 0, 0
    if str(args.stage2_consensus_mode).strip().lower() == "base":
        alt_h, alt_hr, alt_m, alt_mr, alt_conf, _, _ = stage2_json_refine(
            text,
            emotion_analysis_reason,
            target_domain,
            source_domain,
            stage2_hyperbole_judgment,
            stage2_hyperbole_reason,
            stage2_metaphor_judgment,
            stage2_metaphor_reason,
            MODEL_NAME
        )
    else:
        alt_h, alt_hr, alt_m, alt_mr, alt_conf, _, _ = stage2_json_refine_contrast(
            text,
            emotion_analysis_reason,
            target_domain,
            source_domain,
            stage2_hyperbole_judgment,
            stage2_hyperbole_reason,
            stage2_metaphor_judgment,
            stage2_metaphor_reason,
            MODEL_NAME
        )
    consensus_conflict = int(int(alt_h) != int(stage2_hyperbole_judgment) or int(alt_m) != int(stage2_metaphor_judgment))
    if float(alt_conf) > float(stage2_confidence):
        return alt_h, alt_hr, alt_m, alt_mr, alt_conf, 1, consensus_conflict
    return stage2_hyperbole_judgment, stage2_hyperbole_reason, stage2_metaphor_judgment, stage2_metaphor_reason, stage2_confidence, 1, consensus_conflict


def aggregate_stage2_candidates(candidates):
    if not candidates:
        return {
            "stage2_hyperbole_judgment": 0,
            "stage2_hyperbole_reason": "none",
            "stage2_metaphor_judgment": 0,
            "stage2_metaphor_reason": "none",
            "stage2_confidence": 0.0,
            "stage2_parse_ok": 0,
            "stage2_error": "json_parse_failed",
            "stage2_ensemble_agreement": 0.0,
            "stage2_ensemble_support": 0.0,
            "stage2_valid_candidate_count": 0
        }
    valid_candidates = [item for item in candidates if int(item.get("stage2_parse_ok", 0)) == 1]
    pool = valid_candidates if valid_candidates else list(candidates)

    def _clamp_conf(value):
        try:
            conf = float(value)
        except Exception:
            return 0.0
        if conf < 0:
            return 0.0
        if conf > 1:
            return 1.0
        return conf

    pair_scores = {}
    pair_best_candidate = {}
    pair_best_conf = {}
    pair_counts = {}
    total_score = 0.0
    for item in pool:
        pair = (
            int(item.get("stage2_hyperbole_judgment", 0)),
            int(item.get("stage2_metaphor_judgment", 0))
        )
        conf = _clamp_conf(item.get("stage2_confidence", 0.0))
        total_score += conf
        pair_scores[pair] = pair_scores.get(pair, 0.0) + conf
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if pair not in pair_best_conf or conf > pair_best_conf[pair]:
            pair_best_conf[pair] = conf
            pair_best_candidate[pair] = item
    chosen_pair = (0, 0)
    chosen_score = -1.0
    chosen_best_conf = -1.0
    for pair in pair_scores:
        score = pair_scores[pair]
        best_conf = pair_best_conf[pair]
        if score > chosen_score or (score == chosen_score and best_conf > chosen_best_conf):
            chosen_pair = pair
            chosen_score = score
            chosen_best_conf = best_conf
    reason_item = pair_best_candidate.get(chosen_pair, pool[0])
    final_h = int(chosen_pair[0])
    final_m = int(chosen_pair[1])
    final_hr = str(reason_item.get("stage2_hyperbole_reason", "none"))
    final_mr = str(reason_item.get("stage2_metaphor_reason", "none"))
    agreement = pair_counts.get(chosen_pair, 0) / len(pool) if pool else 0.0
    support = 0.0
    if total_score > 0:
        support = chosen_score / total_score
    matched_conf = []
    for item in pool:
        pair = (
            int(item.get("stage2_hyperbole_judgment", 0)),
            int(item.get("stage2_metaphor_judgment", 0))
        )
        if pair == chosen_pair:
            matched_conf.append(_clamp_conf(item.get("stage2_confidence", 0.0)))
    final_conf = max(matched_conf) if matched_conf else support
    parse_ok = 1 if valid_candidates else 0
    stage2_error = "none" if parse_ok == 1 else "json_parse_failed"
    return {
        "stage2_hyperbole_judgment": final_h,
        "stage2_hyperbole_reason": final_hr,
        "stage2_metaphor_judgment": final_m,
        "stage2_metaphor_reason": final_mr,
        "stage2_confidence": final_conf,
        "stage2_parse_ok": parse_ok,
        "stage2_error": stage2_error,
        "stage2_ensemble_agreement": agreement,
        "stage2_ensemble_support": support,
        "stage2_valid_candidate_count": len(valid_candidates)
    }


def run_stage2_pipeline(args, effective_retry_target, text, emotion_analysis_reason, target_domain, source_domain,
                        hyperbole_judgment, hyperbole_reason, metaphor_judgment, metaphor_reason):
    stage2_hyperbole_judgment, stage2_hyperbole_reason, stage2_metaphor_judgment, stage2_metaphor_reason, stage2_confidence, stage2_parse_ok, stage2_error = stage2_json_refine(
        text,
        emotion_analysis_reason,
        target_domain,
        source_domain,
        hyperbole_judgment,
        hyperbole_reason,
        metaphor_judgment,
        metaphor_reason,
        MODEL_NAME
    )
    stage2_candidates = [{
        "stage2_hyperbole_judgment": stage2_hyperbole_judgment,
        "stage2_hyperbole_reason": stage2_hyperbole_reason,
        "stage2_metaphor_judgment": stage2_metaphor_judgment,
        "stage2_metaphor_reason": stage2_metaphor_reason,
        "stage2_confidence": stage2_confidence,
        "stage2_parse_ok": stage2_parse_ok,
        "stage2_error": stage2_error
    }]
    stage2_bootstrap_contrast_applied = 0
    if int(args.stage2_bootstrap_contrast_enable) == 1:
        alt_h, alt_hr, alt_m, alt_mr, alt_conf, alt_parse_ok, alt_error = stage2_json_refine_contrast(
            text,
            emotion_analysis_reason,
            target_domain,
            source_domain,
            stage2_hyperbole_judgment,
            stage2_hyperbole_reason,
            stage2_metaphor_judgment,
            stage2_metaphor_reason,
            MODEL_NAME
        )
        stage2_candidates.append({
            "stage2_hyperbole_judgment": alt_h,
            "stage2_hyperbole_reason": alt_hr,
            "stage2_metaphor_judgment": alt_m,
            "stage2_metaphor_reason": alt_mr,
            "stage2_confidence": alt_conf,
            "stage2_parse_ok": alt_parse_ok,
            "stage2_error": alt_error
        })
        if int(alt_parse_ok) == 1 and float(alt_conf) > float(stage2_confidence):
            stage2_hyperbole_judgment = alt_h
            stage2_hyperbole_reason = alt_hr
            stage2_metaphor_judgment = alt_m
            stage2_metaphor_reason = alt_mr
            stage2_confidence = alt_conf
            stage2_parse_ok = alt_parse_ok
            stage2_error = alt_error
        stage2_bootstrap_contrast_applied = 1
    stage2_retried = 0
    stage2_retry_count = 0
    while stage2_retry_count < max(args.stage2_max_retry, 0) and should_retry_stage2(
        effective_retry_target, stage2_confidence, stage2_metaphor_judgment, args.stage2_conf_threshold
    ):
        stage2_retried = 1
        stage2_retry_count += 1
        if args.stage2_retry_mode == "contrast":
            stage2_hyperbole_judgment, stage2_hyperbole_reason, stage2_metaphor_judgment, stage2_metaphor_reason, stage2_confidence, stage2_parse_ok, stage2_error = stage2_json_refine_contrast(
                text,
                emotion_analysis_reason,
                target_domain,
                source_domain,
                stage2_hyperbole_judgment,
                stage2_hyperbole_reason,
                stage2_metaphor_judgment,
                stage2_metaphor_reason,
                MODEL_NAME
            )
        else:
            stage2_hyperbole_judgment, stage2_hyperbole_reason, stage2_metaphor_judgment, stage2_metaphor_reason, stage2_confidence, stage2_parse_ok, stage2_error = stage2_json_refine(
                text,
                emotion_analysis_reason,
                target_domain,
                source_domain,
                stage2_hyperbole_judgment,
                stage2_hyperbole_reason,
                stage2_metaphor_judgment,
                stage2_metaphor_reason,
                MODEL_NAME
            )
        stage2_candidates.append({
            "stage2_hyperbole_judgment": stage2_hyperbole_judgment,
            "stage2_hyperbole_reason": stage2_hyperbole_reason,
            "stage2_metaphor_judgment": stage2_metaphor_judgment,
            "stage2_metaphor_reason": stage2_metaphor_reason,
            "stage2_confidence": stage2_confidence,
            "stage2_parse_ok": stage2_parse_ok,
            "stage2_error": stage2_error
        })
    stage2_hyperbole_judgment, stage2_hyperbole_reason, stage2_metaphor_judgment, stage2_metaphor_reason, stage2_confidence, stage2_consensus_applied, stage2_consensus_conflict = stage2_consensus_refine(
        args,
        text,
        emotion_analysis_reason,
        target_domain,
        source_domain,
        stage2_hyperbole_judgment,
        stage2_hyperbole_reason,
        stage2_metaphor_judgment,
        stage2_metaphor_reason,
        stage2_confidence
    )
    if stage2_consensus_applied == 1:
        stage2_candidates.append({
            "stage2_hyperbole_judgment": stage2_hyperbole_judgment,
            "stage2_hyperbole_reason": stage2_hyperbole_reason,
            "stage2_metaphor_judgment": stage2_metaphor_judgment,
            "stage2_metaphor_reason": stage2_metaphor_reason,
            "stage2_confidence": stage2_confidence,
            "stage2_parse_ok": stage2_parse_ok,
            "stage2_error": stage2_error
        })
    stage2_ensemble_applied = 0
    stage2_ensemble_agreement = 0.0
    stage2_ensemble_support = 0.0
    stage2_ensemble_support_threshold_hit = 0
    stage2_low_support_refined = 0
    stage2_reaggregate_applied = 0
    stage2_reaggregate_support = 0.0
    stage2_reaggregate_support_gain = 0.0
    stage2_reaggregate_kept = 0
    stage2_reaggregate_required_gain = float(args.stage2_reaggregate_min_support_gain)
    stage2_reaggregate_valid_candidates = 0
    stage2_reaggregate_support_gap = 0.0
    stage2_reaggregate_linked_gain_bonus = 0.0
    stage2_reaggregate_linked_factor_used = 0.0
    stage2_reaggregate_gap_band = "none"
    stage2_reaggregate_candidate_band = "none"
    stage2_reaggregate_candidate_multiplier_used = 1.0
    stage2_reaggregate_threshold_guard_triggered = 0
    stage2_reaggregate_multiplier_guard_triggered = 0
    stage2_reaggregate_effective_gap_mid = float(args.stage2_reaggregate_support_gap_mid)
    stage2_reaggregate_effective_gap_high = float(args.stage2_reaggregate_support_gap_high)
    stage2_reaggregate_effective_candidate_mid = max(int(args.stage2_reaggregate_candidate_mid), 1)
    stage2_reaggregate_effective_candidate_high = max(int(args.stage2_reaggregate_candidate_high), stage2_reaggregate_effective_candidate_mid)
    stage2_valid_candidate_count = sum(int(item.get("stage2_parse_ok", 0)) for item in stage2_candidates)
    if int(args.stage2_aggregate_enable) == 1 and stage2_valid_candidate_count >= max(int(args.stage2_aggregate_min_valid_candidates), 1):
        aggregated_output = aggregate_stage2_candidates(stage2_candidates)
        stage2_hyperbole_judgment = aggregated_output["stage2_hyperbole_judgment"]
        stage2_hyperbole_reason = aggregated_output["stage2_hyperbole_reason"]
        stage2_metaphor_judgment = aggregated_output["stage2_metaphor_judgment"]
        stage2_metaphor_reason = aggregated_output["stage2_metaphor_reason"]
        stage2_confidence = aggregated_output["stage2_confidence"]
        stage2_parse_ok = aggregated_output["stage2_parse_ok"]
        stage2_error = aggregated_output["stage2_error"]
        stage2_ensemble_agreement = aggregated_output["stage2_ensemble_agreement"]
        stage2_ensemble_support = aggregated_output["stage2_ensemble_support"]
        stage2_ensemble_applied = 1
        if float(stage2_ensemble_support) < float(args.stage2_ensemble_support_threshold):
            stage2_ensemble_support_threshold_hit = 1
            if int(args.stage2_low_support_refine_enable) == 1:
                if args.stage2_retry_mode == "contrast":
                    low_h, low_hr, low_m, low_mr, low_conf, low_parse_ok, low_error = stage2_json_refine_contrast(
                        text,
                        emotion_analysis_reason,
                        target_domain,
                        source_domain,
                        stage2_hyperbole_judgment,
                        stage2_hyperbole_reason,
                        stage2_metaphor_judgment,
                        stage2_metaphor_reason,
                        MODEL_NAME
                    )
                else:
                    low_h, low_hr, low_m, low_mr, low_conf, low_parse_ok, low_error = stage2_json_refine(
                        text,
                        emotion_analysis_reason,
                        target_domain,
                        source_domain,
                        stage2_hyperbole_judgment,
                        stage2_hyperbole_reason,
                        stage2_metaphor_judgment,
                        stage2_metaphor_reason,
                        MODEL_NAME
                    )
                stage2_candidates.append({
                    "stage2_hyperbole_judgment": low_h,
                    "stage2_hyperbole_reason": low_hr,
                    "stage2_metaphor_judgment": low_m,
                    "stage2_metaphor_reason": low_mr,
                    "stage2_confidence": low_conf,
                    "stage2_parse_ok": low_parse_ok,
                    "stage2_error": low_error
                })
                if int(low_parse_ok) == 1 and float(low_conf) >= float(stage2_confidence):
                    stage2_hyperbole_judgment = low_h
                    stage2_hyperbole_reason = low_hr
                    stage2_metaphor_judgment = low_m
                    stage2_metaphor_reason = low_mr
                    stage2_confidence = low_conf
                    stage2_parse_ok = low_parse_ok
                    stage2_error = low_error
                stage2_low_support_refined = 1
                stage2_valid_candidate_count = sum(int(item.get("stage2_parse_ok", 0)) for item in stage2_candidates)
                if stage2_valid_candidate_count >= max(int(args.stage2_aggregate_min_valid_candidates), 1):
                    stage2_reaggregate_valid_candidates = stage2_valid_candidate_count
                    adaptive_extra = max(stage2_reaggregate_valid_candidates - int(args.stage2_aggregate_min_valid_candidates), 0)
                    stage2_reaggregate_required_gain = float(args.stage2_reaggregate_min_support_gain)
                    if int(args.stage2_reaggregate_adaptive_gain_enable) == 1:
                        stage2_reaggregate_required_gain += adaptive_extra * float(args.stage2_reaggregate_adaptive_gain_step)
                    stage2_reaggregate_support_gap = max(float(args.stage2_ensemble_support_threshold) - float(stage2_ensemble_support), 0.0)
                    stage2_reaggregate_linked_gain_bonus = 0.0
                    stage2_reaggregate_linked_factor_used = 0.0
                    stage2_reaggregate_gap_band = "none"
                    if int(args.stage2_reaggregate_support_gap_link_enable) == 1:
                        base_factor = max(float(args.stage2_reaggregate_support_gap_link_factor), 0.0)
                        if base_factor != float(args.stage2_reaggregate_support_gap_link_factor):
                            stage2_reaggregate_multiplier_guard_triggered = 1
                        factor_used = base_factor
                        mid_gap = max(float(args.stage2_reaggregate_support_gap_mid), 0.0)
                        high_gap = max(float(args.stage2_reaggregate_support_gap_high), mid_gap)
                        if mid_gap != float(args.stage2_reaggregate_support_gap_mid) or high_gap != float(args.stage2_reaggregate_support_gap_high):
                            stage2_reaggregate_threshold_guard_triggered = 1
                        stage2_reaggregate_effective_gap_mid = mid_gap
                        stage2_reaggregate_effective_gap_high = high_gap
                        gap_mid_multiplier = max(float(args.stage2_reaggregate_support_gap_mid_multiplier), 0.0)
                        gap_high_multiplier = max(float(args.stage2_reaggregate_support_gap_high_multiplier), 0.0)
                        if gap_mid_multiplier != float(args.stage2_reaggregate_support_gap_mid_multiplier) or gap_high_multiplier != float(args.stage2_reaggregate_support_gap_high_multiplier):
                            stage2_reaggregate_multiplier_guard_triggered = 1
                        if stage2_reaggregate_support_gap >= high_gap:
                            factor_used = base_factor * gap_high_multiplier
                            stage2_reaggregate_gap_band = "high"
                        elif stage2_reaggregate_support_gap >= mid_gap:
                            factor_used = base_factor * gap_mid_multiplier
                            stage2_reaggregate_gap_band = "mid"
                        else:
                            stage2_reaggregate_gap_band = "base"
                        candidate_mid = max(int(args.stage2_reaggregate_candidate_mid), 1)
                        candidate_high = max(int(args.stage2_reaggregate_candidate_high), candidate_mid)
                        if candidate_mid != int(args.stage2_reaggregate_candidate_mid) or candidate_high != int(args.stage2_reaggregate_candidate_high):
                            stage2_reaggregate_threshold_guard_triggered = 1
                        stage2_reaggregate_effective_candidate_mid = candidate_mid
                        stage2_reaggregate_effective_candidate_high = candidate_high
                        candidate_mid_multiplier = max(float(args.stage2_reaggregate_candidate_mid_multiplier), 0.0)
                        candidate_high_multiplier = max(float(args.stage2_reaggregate_candidate_high_multiplier), 0.0)
                        if candidate_mid_multiplier != float(args.stage2_reaggregate_candidate_mid_multiplier) or candidate_high_multiplier != float(args.stage2_reaggregate_candidate_high_multiplier):
                            stage2_reaggregate_multiplier_guard_triggered = 1
                        if stage2_reaggregate_valid_candidates >= candidate_high:
                            stage2_reaggregate_candidate_band = "high"
                            stage2_reaggregate_candidate_multiplier_used = candidate_high_multiplier
                        elif stage2_reaggregate_valid_candidates >= candidate_mid:
                            stage2_reaggregate_candidate_band = "mid"
                            stage2_reaggregate_candidate_multiplier_used = candidate_mid_multiplier
                        else:
                            stage2_reaggregate_candidate_band = "base"
                            stage2_reaggregate_candidate_multiplier_used = 1.0
                        factor_used = factor_used * stage2_reaggregate_candidate_multiplier_used
                        stage2_reaggregate_linked_factor_used = factor_used
                        stage2_reaggregate_linked_gain_bonus = stage2_reaggregate_support_gap * factor_used
                    stage2_reaggregate_required_gain += stage2_reaggregate_linked_gain_bonus
                    previous_support = float(stage2_ensemble_support)
                    reaggregated_output = aggregate_stage2_candidates(stage2_candidates)
                    stage2_reaggregate_applied = 1
                    stage2_reaggregate_support = reaggregated_output["stage2_ensemble_support"]
                    stage2_reaggregate_support_gain = float(stage2_reaggregate_support) - previous_support
                    if stage2_reaggregate_support_gain >= float(stage2_reaggregate_required_gain):
                        stage2_hyperbole_judgment = reaggregated_output["stage2_hyperbole_judgment"]
                        stage2_hyperbole_reason = reaggregated_output["stage2_hyperbole_reason"]
                        stage2_metaphor_judgment = reaggregated_output["stage2_metaphor_judgment"]
                        stage2_metaphor_reason = reaggregated_output["stage2_metaphor_reason"]
                        stage2_confidence = reaggregated_output["stage2_confidence"]
                        stage2_parse_ok = reaggregated_output["stage2_parse_ok"]
                        stage2_error = reaggregated_output["stage2_error"]
                        stage2_ensemble_agreement = reaggregated_output["stage2_ensemble_agreement"]
                        stage2_ensemble_support = reaggregated_output["stage2_ensemble_support"]
                        stage2_reaggregate_kept = 1
    stage2_candidate_count = len(stage2_candidates)
    stage2_valid_candidate_count = sum(int(item.get("stage2_parse_ok", 0)) for item in stage2_candidates)
    stage2_hyperbole_judgment, stage2_hyperbole_reason, stage2_metaphor_judgment, stage2_metaphor_reason, stage2_rule_applied = apply_stage2_rules(
        args,
        target_domain,
        source_domain,
        stage2_hyperbole_judgment,
        stage2_hyperbole_reason,
        stage2_metaphor_judgment,
        stage2_metaphor_reason,
        stage2_confidence
    )
    return {
        "stage2_hyperbole_judgment": stage2_hyperbole_judgment,
        "stage2_hyperbole_reason": stage2_hyperbole_reason,
        "stage2_metaphor_judgment": stage2_metaphor_judgment,
        "stage2_metaphor_reason": stage2_metaphor_reason,
        "stage2_confidence": stage2_confidence,
        "stage2_parse_ok": stage2_parse_ok,
        "stage2_error": stage2_error,
        "stage2_retried": stage2_retried,
        "stage2_retry_count": stage2_retry_count,
        "stage2_rule_applied": stage2_rule_applied,
        "stage2_consensus_applied": stage2_consensus_applied,
        "stage2_consensus_conflict": stage2_consensus_conflict,
        "stage2_candidate_count": stage2_candidate_count,
        "stage2_valid_candidate_count": stage2_valid_candidate_count,
        "stage2_ensemble_applied": stage2_ensemble_applied,
        "stage2_ensemble_agreement": stage2_ensemble_agreement,
        "stage2_ensemble_support": stage2_ensemble_support,
        "stage2_ensemble_support_threshold_hit": stage2_ensemble_support_threshold_hit,
        "stage2_low_support_refined": stage2_low_support_refined,
        "stage2_reaggregate_applied": stage2_reaggregate_applied,
        "stage2_reaggregate_support": stage2_reaggregate_support,
        "stage2_reaggregate_support_gain": stage2_reaggregate_support_gain,
        "stage2_reaggregate_kept": stage2_reaggregate_kept,
        "stage2_reaggregate_required_gain": stage2_reaggregate_required_gain,
        "stage2_reaggregate_valid_candidates": stage2_reaggregate_valid_candidates,
        "stage2_reaggregate_support_gap": stage2_reaggregate_support_gap,
        "stage2_reaggregate_linked_gain_bonus": stage2_reaggregate_linked_gain_bonus,
        "stage2_reaggregate_linked_factor_used": stage2_reaggregate_linked_factor_used,
        "stage2_reaggregate_gap_band": stage2_reaggregate_gap_band,
        "stage2_reaggregate_candidate_band": stage2_reaggregate_candidate_band,
        "stage2_reaggregate_candidate_multiplier_used": stage2_reaggregate_candidate_multiplier_used,
        "stage2_reaggregate_threshold_guard_triggered": stage2_reaggregate_threshold_guard_triggered,
        "stage2_reaggregate_multiplier_guard_triggered": stage2_reaggregate_multiplier_guard_triggered,
        "stage2_reaggregate_guard_any_triggered": int(
            stage2_reaggregate_threshold_guard_triggered == 1 or stage2_reaggregate_multiplier_guard_triggered == 1
        ),
        "stage2_reaggregate_effective_gap_mid": stage2_reaggregate_effective_gap_mid,
        "stage2_reaggregate_effective_gap_high": stage2_reaggregate_effective_gap_high,
        "stage2_reaggregate_effective_candidate_mid": stage2_reaggregate_effective_candidate_mid,
        "stage2_reaggregate_effective_candidate_high": stage2_reaggregate_effective_candidate_high,
        "stage2_bootstrap_contrast_applied": stage2_bootstrap_contrast_applied
    }


def evaluate_stage3_trigger(args, stage2_hyperbole_judgment, stage2_metaphor_judgment, stage2_confidence, stage2_parse_ok):
    if int(args.stage3_enable) != 1:
        return 0, "stage3_disabled"
    parse_failed = int(stage2_parse_ok) == 0
    low_conf = float(stage2_confidence) < float(args.stage3_low_conf_threshold)
    conflict = int(stage2_hyperbole_judgment) != int(stage2_metaphor_judgment)
    mode = str(args.stage3_trigger_mode).strip().lower()
    if mode == "parse_fail_only":
        trigger = parse_failed
    elif mode == "low_conf_only":
        trigger = low_conf
    elif mode == "conflict_only":
        trigger = conflict
    elif mode == "never":
        trigger = False
    else:
        trigger = parse_failed or low_conf or conflict
    if not trigger:
        return 0, "not_triggered"
    reasons = []
    if parse_failed:
        reasons.append("parse_failed")
    if low_conf:
        reasons.append("low_confidence")
    if conflict:
        reasons.append("task_conflict")
    return 1, "|".join(reasons) if reasons else "triggered"


def resolve_media_path(row, candidates):
    for key in candidates:
        if key in row.index and not pd.isna(row[key]):
            value = str(row[key]).strip()
            if value:
                return value
    return "none"


def parse_column_candidates(raw_text):
    values = [item.strip() for item in str(raw_text).split(",")]
    values = [item for item in values if item]
    if not values:
        return ["none"]
    return list(dict.fromkeys(values))


def enrich_candidates_with_existing_columns(candidates, all_columns, keyword_patterns):
    merged = list(candidates)
    lower_map = {str(col).lower(): col for col in all_columns}
    for lower_col, original_col in lower_map.items():
        if any(pattern in lower_col for pattern in keyword_patterns):
            merged.append(original_col)
    merged = [item for item in merged if str(item).strip() and str(item).lower() != "none"]
    if not merged:
        return ["none"]
    return list(dict.fromkeys(merged))


def resolve_media_path_with_key(row, candidates):
    for key in candidates:
        if key in row.index and not pd.isna(row[key]):
            value = str(row[key]).strip()
            if value:
                return key, value
    return "none", "none"


def parse_media_base_dirs(raw_text):
    values = [item.strip() for item in re.split(r"[;,]", str(raw_text))]
    values = [item for item in values if item]
    return list(dict.fromkeys(values))


def parse_media_resolve_order(raw_text):
    supported = {"raw", "base", "data", "extra"}
    values = [item.strip().lower() for item in str(raw_text).split(",")]
    values = [item for item in values if item in supported]
    if not values:
        values = ["raw", "base", "data", "extra"]
    return list(dict.fromkeys(values))


def resolve_media_path_existence(path_value, data_dir, media_base_dir, extra_base_dirs=None, resolve_order=None):
    normalized = str(path_value).strip()
    if normalized.lower() in {"", "none", "null", "na", "n/a"}:
        return 0, "none", "none"
    extra_dirs = list(extra_base_dirs or [])
    order = list(resolve_order or ["raw", "base", "data", "extra"])
    candidates = []
    if os.path.isabs(normalized):
        candidates.append((normalized, "raw"))
    else:
        for step in order:
            if step == "raw":
                candidates.append((normalized, "raw"))
            elif step == "base" and media_base_dir and str(media_base_dir).strip():
                candidates.append((os.path.join(str(media_base_dir).strip(), normalized), "base"))
            elif step == "data":
                candidates.append((os.path.join(str(data_dir).strip(), normalized), "data"))
            elif step == "extra":
                for base_dir in extra_dirs:
                    candidates.append((os.path.join(str(base_dir).strip(), normalized), "extra"))
    if not candidates:
        candidates = [(normalized, "raw")]
    dedup_candidates = []
    seen_paths = set()
    for candidate_path, candidate_source in candidates:
        if candidate_path in seen_paths:
            continue
        seen_paths.add(candidate_path)
        dedup_candidates.append((candidate_path, candidate_source))
    for item, source in dedup_candidates:
        if os.path.exists(item):
            return 1, item, source
    return 0, dedup_candidates[0][0], "unresolved"


def rewrite_media_path(path_value, prefix_from, prefix_to):
    normalized = str(path_value).strip()
    if normalized.lower() in {"", "none", "null", "na", "n/a"}:
        return normalized
    source_prefix = str(prefix_from).strip()
    target_prefix = str(prefix_to).strip()
    if source_prefix and target_prefix and normalized.startswith(source_prefix):
        return target_prefix + normalized[len(source_prefix):]
    return normalized


def parse_media_path_map(raw_text, fallback_from="", fallback_to=""):
    mappings = []
    raw = str(raw_text).strip()
    if raw:
        for item in raw.split(";"):
            pair = str(item).strip()
            if not pair or "=>" not in pair:
                continue
            source_prefix, target_prefix = pair.split("=>", 1)
            source_prefix = str(source_prefix).strip()
            target_prefix = str(target_prefix).strip()
            if source_prefix and target_prefix:
                mappings.append((source_prefix, target_prefix))
    source_prefix = str(fallback_from).strip()
    target_prefix = str(fallback_to).strip()
    if source_prefix and target_prefix:
        mappings.append((source_prefix, target_prefix))
    unique_mappings = []
    seen = set()
    for source_prefix, target_prefix in mappings:
        key = (source_prefix, target_prefix)
        if key in seen:
            continue
        seen.add(key)
        unique_mappings.append(key)
    return unique_mappings


def rewrite_media_path_by_mappings(path_value, mappings):
    current = str(path_value).strip()
    if current.lower() in {"", "none", "null", "na", "n/a"}:
        return current, 0
    hit = 0
    for source_prefix, target_prefix in mappings:
        rewritten = rewrite_media_path(current, source_prefix, target_prefix)
        if rewritten != current:
            current = rewritten
            hit = 1
    return current, hit


def load_existing_records(results_dir, file_name):
    file_path = os.path.join(results_dir, file_name)
    if not os.path.exists(file_path):
        return []
    try:
        return pd.read_csv(file_path).to_dict(orient="records")
    except Exception:
        return []


def safe_int(value, default=-1):
    try:
        return int(value)
    except Exception:
        return default


def save_stage_outputs(args, emo, domain, meta, hype, result, stage3_candidates):
    store_data(emo, "stage_emotion.csv", args.results_dir)
    store_data(domain, "stage_domain.csv", args.results_dir)
    store_data(meta, "stage_metaphor.csv", args.results_dir)
    store_data(hype, "stage_hyperbole.csv", args.results_dir)
    store_data(result, "stage_final.csv", args.results_dir)
    if int(args.stage3_enable) == 1:
        store_data(stage3_candidates, "stage3_candidates.csv", args.results_dir)


def sanitize_run_name(name):
    normalized = str(name).strip().replace("\\", "_").replace("/", "_")
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("._-")
    return normalized or "default_run"


def resolve_results_dir(args):
    configured = str(args.results_dir).strip()
    if not configured:
        configured = os.path.join(RESULTS_DIR, "runs")
    normalized = os.path.normpath(configured)
    run_name_raw = str(getattr(args, "run_name", "")).strip()
    base_name = os.path.basename(normalized).lower()

    # Backward compatibility: explicit custom non-runs path remains unchanged
    # unless user passes --run-name.
    if base_name == "results":
        base_dir = os.path.join(normalized, "runs")
    else:
        base_dir = normalized

    if os.path.basename(base_dir).lower() == "runs":
        if not run_name_raw:
            input_stem = os.path.splitext(os.path.basename(str(args.input_name)))[0] or "dataset"
            profile = str(args.ablation_profile).strip().lower() or "full"
            run_name_raw = f"{input_stem}__{profile}"
        run_name = sanitize_run_name(run_name_raw)
        return os.path.join(base_dir, run_name), run_name

    if run_name_raw:
        run_name = sanitize_run_name(run_name_raw)
        return os.path.join(base_dir, run_name), run_name

    return base_dir, ""


def main():
    args = parse_args()
    args = apply_ablation_profile(args)
    if not os.path.isabs(str(args.data_dir)):
        args.data_dir = os.path.normpath(os.path.join(SCRIPT_DIR, str(args.data_dir)))
    args.results_dir, resolved_run_name = resolve_results_dir(args)
    effective_retry_target, retry_target_resolution_reason = resolve_retry_target(args)
    retry_target_resolution_reason = normalize_retry_target_resolution_reason(retry_target_resolution_reason)
    stage3_media_path_mappings = parse_media_path_map(
        args.stage3_media_path_map,
        args.stage3_media_path_from,
        args.stage3_media_path_to
    )
    stage3_media_extra_base_dirs = parse_media_base_dirs(args.stage3_media_extra_base_dirs)
    stage3_media_resolve_order = parse_media_resolve_order(args.stage3_media_resolve_order)
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Results directory: {args.results_dir}")
    if resolved_run_name:
        print(f"Run name: {resolved_run_name}")
    pf = read_csv(args.data_dir, args.input_name)
    if int(args.max_samples) > 0:
        pf = pf.head(int(args.max_samples))
    if int(args.resume_enable) == 1:
        emo = load_existing_records(args.results_dir, "stage_emotion.csv")
        domain = load_existing_records(args.results_dir, "stage_domain.csv")
        meta = load_existing_records(args.results_dir, "stage_metaphor.csv")
        hype = load_existing_records(args.results_dir, "stage_hyperbole.csv")
        result = load_existing_records(args.results_dir, "stage_final.csv")
        stage3_candidates = load_existing_records(args.results_dir, "stage3_candidates.csv")
    else:
        emo, domain, meta, hype, result, stage3_candidates = [], [], [], [], [], []
    processed_ids = set()
    for item in result:
        item_id = safe_int(item.get("Id"), default=-1)
        if item_id >= 0:
            processed_ids.add(item_id)
    if processed_ids:
        print(f"Resume mode: skip existing samples {len(processed_ids)}")
    checkpoint_interval = max(int(args.checkpoint_interval), 0)
    newly_processed = 0
    stage3_image_columns = enrich_candidates_with_existing_columns(
        parse_column_candidates(args.stage3_image_columns),
        pf.columns,
        ["image", "img", "frame", "picture", "photo", "截图", "图像", "图片"]
    )
    stage3_audio_columns = enrich_candidates_with_existing_columns(
        parse_column_candidates(args.stage3_audio_columns),
        pf.columns,
        ["audio", "wav", "voice", "speech", "声音", "音频", "语音"]
    )
    stage3_video_columns = enrich_candidates_with_existing_columns(
        parse_column_candidates(args.stage3_video_columns),
        pf.columns,
        ["video", "clip", "mp4", "movie", "录像", "视频", "片段"]
    )

    if int(args.use_batch_inference) == 1:
        batch_size = int(args.batch_size)
        batch_texts = []
        batch_indices = []
        batch_emotion_hints = []
        batch_concept_hints = []
        batch_stage1_pass = []
        batch_row_data = []
        for i, row in tqdm(pf.iterrows(), total=len(pf), desc="Batch inference"):
            if i in processed_ids:
                continue
            text = row["Sentence"]
            stage1_pass, stage1_reason = stage1_counterfactual_filter(text)
            emotion_hint_raw = "none"
            if "Emotion lexicon hint" in pf.columns and not pd.isna(row["Emotion lexicon hint"]):
                emotion_hint_raw = str(row["Emotion lexicon hint"])
            emotion_hint = "none"
            if int(args.use_emotion_lexicon_feature) == 1:
                emotion_hint = build_emotion_lexicon_hint(text, emotion_hint_raw)
            concept_hint = "none"
            if int(args.use_conceptnet_feature) == 1:
                concept_hint = build_conceptnet_hint(text)
            batch_texts.append(text)
            batch_indices.append(i)
            batch_emotion_hints.append(emotion_hint)
            batch_concept_hints.append(concept_hint)
            batch_stage1_pass.append((stage1_pass, stage1_reason))
            batch_row_data.append({
                "Emotion lexicon hint raw": emotion_hint_raw,
                "True hyperbole": int(row["Hyperbole"]) if not pd.isna(row["Hyperbole"]) else 0,
                "True metaphor": int(row["Metaphor"]) if not pd.isna(row["Metaphor"]) else 0
            })
            if len(batch_texts) >= batch_size:
                batch_results = analyze_batch(batch_texts, MODEL_NAME, batch_emotion_hints, batch_concept_hints)
                for idx, (batch_idx, row_data, stage1_info) in enumerate(zip(batch_indices, batch_row_data, batch_stage1_pass)):
                    res = batch_results[idx]
                    stage1_pass, stage1_reason = stage1_info
                    emo.append({
                        "Id": batch_idx,
                        "Sentence": batch_texts[idx],
                        "Stage1 pass": stage1_pass,
                        "Stage1 reason": stage1_reason,
                        "Emotion lexicon hint raw": row_data["Emotion lexicon hint raw"],
                        "Emotion lexicon hint": batch_emotion_hints[idx],
                        "Emotion analysis reason": res["emotion_reason"]
                    })
                    domain.append({
                        "Id": batch_idx,
                        "Sentence": batch_texts[idx],
                        "Stage1 pass": stage1_pass,
                        "Stage1 reason": stage1_reason,
                        "ConceptNet hint": batch_concept_hints[idx],
                        "Target domain": res["target_domain"],
                        "Select reason": "none",
                        "Source domain": res["source_domain"],
                        "Generate reason": "none"
                    })
                    meta.append({
                        "Id": batch_idx,
                        "Sentence": batch_texts[idx],
                        "Stage1 pass": stage1_pass,
                        "Stage1 reason": stage1_reason,
                        "Metaphor judgment": res["metaphor_judgment"],
                        "Metaphor reason": "none"
                    })
                    hype.append({
                        "Id": batch_idx,
                        "Sentence": batch_texts[idx],
                        "Stage1 pass": stage1_pass,
                        "Stage1 reason": stage1_reason,
                        "Metaphor judgment": res["metaphor_judgment"],
                        "Metaphor reason": "none",
                        "Hyperbole judgment": res["hyperbole_judgment"],
                        "Hyperbole reason": "none"
                    })
                    init_hyperbole = res["hyperbole_judgment"]
                    init_metaphor = res["metaphor_judgment"]
                    hyperbole_judgment, hyperbole_reason, metaphor_judgment, metaphor_reason = hyperbole_metaphor(
                        batch_texts[idx],
                        res["emotion_reason"],
                        res["hyperbole_judgment"],
                        "none",
                        res["metaphor_judgment"],
                        "none",
                        res["target_domain"],
                        res["source_domain"],
                        MODEL_NAME,
                        batch_concept_hints[idx]
                    )
                    stage2_output = run_stage2_pipeline(
                        args,
                        effective_retry_target,
                        batch_texts[idx],
                        res["emotion_reason"],
                        res["target_domain"],
                        res["source_domain"],
                        hyperbole_judgment,
                        hyperbole_reason,
                        metaphor_judgment,
                        metaphor_reason
                    )
                    stage2_hyperbole_judgment = stage2_output["stage2_hyperbole_judgment"]
                    stage2_hyperbole_reason = stage2_output["stage2_hyperbole_reason"]
                    stage2_metaphor_judgment = stage2_output["stage2_metaphor_judgment"]
                    stage2_metaphor_reason = stage2_output["stage2_metaphor_reason"]
                    stage2_confidence = stage2_output["stage2_confidence"]
                    stage2_parse_ok = stage2_output.get("stage2_parse_ok", 0)
                    stage2_error = stage2_output.get("stage2_error", "")
                    stage2_retried = stage2_output.get("stage2_retried", 0)
                    stage2_retry_count = stage2_output.get("stage2_retry_count", 0)
                    stage2_rule_applied = stage2_output.get("stage2_rule_applied", "")
                    stage2_consensus_applied = stage2_output.get("stage2_consensus_applied", 0)
                    stage2_consensus_conflict = stage2_output.get("stage2_consensus_conflict", 0)
                    stage2_ensemble_support_threshold_hit = stage2_output.get("stage2_ensemble_support_threshold_hit", 0)
                    stage2_low_support_refined = stage2_output.get("stage2_low_support_refined", 0)
                    stage2_reaggregate_applied = stage2_output.get("stage2_reaggregate_applied", 0)
                    stage2_reaggregate_support = stage2_output.get("stage2_reaggregate_support", 0.0)
                    stage2_reaggregate_support_gain = stage2_output.get("stage2_reaggregate_support_gain", 0.0)
                    stage2_reaggregate_kept = stage2_output.get("stage2_reaggregate_kept", 0)
                    stage2_reaggregate_required_gain = stage2_output.get("stage2_reaggregate_required_gain", 0.0)
                    stage2_reaggregate_valid_candidates = stage2_output.get("stage2_reaggregate_valid_candidates", 0)
                    stage2_reaggregate_support_gap = stage2_output.get("stage2_reaggregate_support_gap", 0.0)
                    stage2_reaggregate_linked_gain_bonus = stage2_output.get("stage2_reaggregate_linked_gain_bonus", 0.0)
                    stage2_reaggregate_linked_factor_used = stage2_output.get("stage2_reaggregate_linked_factor_used", 0.0)
                    stage2_reaggregate_gap_band = stage2_output.get("stage2_reaggregate_gap_band", "none")
                    stage2_reaggregate_candidate_band = stage2_output.get("stage2_reaggregate_candidate_band", "none")
                    stage2_reaggregate_candidate_multiplier_used = stage2_output.get("stage2_reaggregate_candidate_multiplier_used", 1.0)
                    stage2_reaggregate_threshold_guard_triggered = stage2_output.get("stage2_reaggregate_threshold_guard_triggered", 0)
                    stage2_reaggregate_multiplier_guard_triggered = stage2_output.get("stage2_reaggregate_multiplier_guard_triggered", 0)
                    stage2_reaggregate_guard_any_triggered = stage2_output.get("stage2_reaggregate_guard_any_triggered", 0)
                    stage2_reaggregate_effective_gap_mid = stage2_output.get("stage2_reaggregate_effective_gap_mid", 0.0)
                    stage2_reaggregate_effective_gap_high = stage2_output.get("stage2_reaggregate_effective_gap_high", 0.0)
                    stage2_reaggregate_effective_candidate_mid = stage2_output.get("stage2_reaggregate_effective_candidate_mid", 0)
                    stage2_reaggregate_effective_candidate_high = stage2_output.get("stage2_reaggregate_effective_candidate_high", 0)
                    stage2_bootstrap_contrast_applied = stage2_output.get("stage2_bootstrap_contrast_applied", 0)
                    stage2_candidate_count = len(stage2_output.get("stage2_candidates", []))
                    stage2_valid_candidate_count = sum(int(item.get("stage2_parse_ok", 0)) for item in stage2_output.get("stage2_candidates", []))
                    stage2_ensemble_applied = stage2_output.get("stage2_ensemble_applied", 0)
                    stage2_ensemble_agreement = stage2_output.get("stage2_ensemble_agreement", 0.0)
                    stage2_ensemble_support = stage2_output.get("stage2_ensemble_support", 0.0)
                    stage3_triggered, stage3_trigger_reason = evaluate_stage3_trigger(
                        args, stage2_hyperbole_judgment, stage2_metaphor_judgment,
                        stage2_confidence, stage2_consensus_conflict
                    )
                    stage3_status = "skipped"
                    if stage3_triggered == 1:
                        stage3_status = "triggered"
                    result.append({
                        "Id": batch_idx,
                        "Sentence": batch_texts[idx],
                        "Stage1 pass": stage1_pass,
                        "Stage1 reason": stage1_reason,
                        "Stage2 retry reason": stage2_output.get("stage2_retry_reason", ""),
                        "Stage2 retry count": stage2_retry_count,
                        "Stage2 rule applied": stage2_rule_applied,
                        "Stage2 consensus applied": stage2_consensus_applied,
                        "Stage2 consensus conflict": stage2_consensus_conflict,
                        "Stage2 bootstrap contrast applied": stage2_bootstrap_contrast_applied,
                        "Stage2 candidate count": stage2_candidate_count,
                        "Stage2 valid candidate count": stage2_valid_candidate_count,
                        "Stage2 ensemble applied": stage2_ensemble_applied,
                        "Stage2 ensemble agreement": stage2_ensemble_agreement,
                        "Stage2 ensemble support": stage2_ensemble_support,
                        "Stage2 ensemble support threshold hit": stage2_ensemble_support_threshold_hit,
                        "Stage2 low support refined": stage2_low_support_refined,
                        "Stage2 reaggregate applied": stage2_reaggregate_applied,
                        "Stage2 reaggregate support": stage2_reaggregate_support,
                        "Stage2 reaggregate support gain": stage2_reaggregate_support_gain,
                        "Stage2 reaggregate kept": stage2_reaggregate_kept,
                        "Stage2 reaggregate required gain": stage2_reaggregate_required_gain,
                        "Stage2 reaggregate valid candidates": stage2_reaggregate_valid_candidates,
                        "Stage2 reaggregate support gap": stage2_reaggregate_support_gap,
                        "Stage2 reaggregate linked gain bonus": stage2_reaggregate_linked_gain_bonus,
                        "Stage2 reaggregate linked factor used": stage2_reaggregate_linked_factor_used,
                        "Stage2 reaggregate gap band": stage2_reaggregate_gap_band,
                        "Stage2 reaggregate candidate band": stage2_reaggregate_candidate_band,
                        "Stage2 reaggregate candidate multiplier used": stage2_reaggregate_candidate_multiplier_used,
                        "Stage2 reaggregate threshold guard triggered": stage2_reaggregate_threshold_guard_triggered,
                        "Stage2 reaggregate multiplier guard triggered": stage2_reaggregate_multiplier_guard_triggered,
                        "Stage2 reaggregate guard any triggered": stage2_reaggregate_guard_any_triggered,
                        "Stage2 reaggregate effective gap mid": stage2_reaggregate_effective_gap_mid,
                        "Stage2 reaggregate effective gap high": stage2_reaggregate_effective_gap_high,
                        "Stage2 reaggregate effective candidate mid": stage2_reaggregate_effective_candidate_mid,
                        "Stage2 reaggregate effective candidate high": stage2_reaggregate_effective_candidate_high,
                        "Stage3 triggered": stage3_triggered,
                        "Stage3 trigger reason": stage3_trigger_reason,
                        "Stage3 status": stage3_status,
                        "True hyperbole": row_data["True hyperbole"],
                        "True metaphor": row_data["True metaphor"],
                        "Hyperbole judgment": stage2_hyperbole_judgment,
                        "Hyperbole reason": stage2_hyperbole_reason,
                        "Metaphor judgment": stage2_metaphor_judgment,
                        "Metaphor reason": stage2_metaphor_reason,
                        "Interaction updated": int(init_hyperbole != stage2_hyperbole_judgment or init_metaphor != stage2_metaphor_judgment)
                    })
                    processed_ids.add(batch_idx)
                    newly_processed += 1
                batch_texts = []
                batch_indices = []
                batch_emotion_hints = []
                batch_concept_hints = []
                batch_stage1_pass = []
                batch_row_data = []
                if checkpoint_interval > 0 and newly_processed % checkpoint_interval == 0:
                    save_stage_outputs(args, emo, domain, meta, hype, result, stage3_candidates)
                    print(f"Checkpoint saved: +{newly_processed} new samples (batch mode)")
        if batch_texts:
            batch_results = analyze_batch(batch_texts, MODEL_NAME, batch_emotion_hints, batch_concept_hints)
            for idx, (batch_idx, row_data, stage1_info) in enumerate(zip(batch_indices, batch_row_data, batch_stage1_pass)):
                res = batch_results[idx]
                stage1_pass, stage1_reason = stage1_info
                emo.append({
                    "Id": batch_idx,
                    "Sentence": batch_texts[idx],
                    "Stage1 pass": stage1_pass,
                    "Stage1 reason": stage1_reason,
                    "Emotion lexicon hint raw": row_data["Emotion lexicon hint raw"],
                    "Emotion lexicon hint": batch_emotion_hints[idx],
                    "Emotion analysis reason": res["emotion_reason"]
                })
                domain.append({
                    "Id": batch_idx,
                    "Sentence": batch_texts[idx],
                    "Stage1 pass": stage1_pass,
                    "Stage1 reason": stage1_reason,
                    "ConceptNet hint": batch_concept_hints[idx],
                    "Target domain": res["target_domain"],
                    "Select reason": "none",
                    "Source domain": res["source_domain"],
                    "Generate reason": "none"
                })
                meta.append({
                    "Id": batch_idx,
                    "Sentence": batch_texts[idx],
                    "Stage1 pass": stage1_pass,
                    "Stage1 reason": stage1_reason,
                    "Metaphor judgment": res["metaphor_judgment"],
                    "Metaphor reason": "none"
                })
                hype.append({
                    "Id": batch_idx,
                    "Sentence": batch_texts[idx],
                    "Stage1 pass": stage1_pass,
                    "Stage1 reason": stage1_reason,
                    "Metaphor judgment": res["metaphor_judgment"],
                    "Metaphor reason": "none",
                    "Hyperbole judgment": res["hyperbole_judgment"],
                    "Hyperbole reason": "none"
                })
                init_hyperbole = res["hyperbole_judgment"]
                init_metaphor = res["metaphor_judgment"]
                hyperbole_judgment, hyperbole_reason, metaphor_judgment, metaphor_reason = hyperbole_metaphor(
                    batch_texts[idx],
                    res["emotion_reason"],
                    res["hyperbole_judgment"],
                    "none",
                    res["metaphor_judgment"],
                    "none",
                    res["target_domain"],
                    res["source_domain"],
                    MODEL_NAME,
                    batch_concept_hints[idx]
                )
                stage2_output = run_stage2_pipeline(
                    args,
                    effective_retry_target,
                    batch_texts[idx],
                    res["emotion_reason"],
                    res["target_domain"],
                    res["source_domain"],
                    hyperbole_judgment,
                    hyperbole_reason,
                    metaphor_judgment,
                    metaphor_reason
                )
                stage2_hyperbole_judgment = stage2_output["stage2_hyperbole_judgment"]
                stage2_hyperbole_reason = stage2_output["stage2_hyperbole_reason"]
                stage2_metaphor_judgment = stage2_output["stage2_metaphor_judgment"]
                stage2_metaphor_reason = stage2_output["stage2_metaphor_reason"]
                stage2_confidence = stage2_output["stage2_confidence"]
                stage2_parse_ok = stage2_output.get("stage2_parse_ok", 0)
                stage2_error = stage2_output.get("stage2_error", "")
                stage2_retried = stage2_output.get("stage2_retried", 0)
                stage2_retry_count = stage2_output.get("stage2_retry_count", 0)
                stage2_rule_applied = stage2_output.get("stage2_rule_applied", "")
                stage2_consensus_applied = stage2_output.get("stage2_consensus_applied", 0)
                stage2_consensus_conflict = stage2_output.get("stage2_consensus_conflict", 0)
                stage2_ensemble_support_threshold_hit = stage2_output.get("stage2_ensemble_support_threshold_hit", 0)
                stage2_low_support_refined = stage2_output.get("stage2_low_support_refined", 0)
                stage2_reaggregate_applied = stage2_output.get("stage2_reaggregate_applied", 0)
                stage2_reaggregate_support = stage2_output.get("stage2_reaggregate_support", 0.0)
                stage2_reaggregate_support_gain = stage2_output.get("stage2_reaggregate_support_gain", 0.0)
                stage2_reaggregate_kept = stage2_output.get("stage2_reaggregate_kept", 0)
                stage2_reaggregate_required_gain = stage2_output.get("stage2_reaggregate_required_gain", 0.0)
                stage2_reaggregate_valid_candidates = stage2_output.get("stage2_reaggregate_valid_candidates", 0)
                stage2_reaggregate_support_gap = stage2_output.get("stage2_reaggregate_support_gap", 0.0)
                stage2_reaggregate_linked_gain_bonus = stage2_output.get("stage2_reaggregate_linked_gain_bonus", 0.0)
                stage2_reaggregate_linked_factor_used = stage2_output.get("stage2_reaggregate_linked_factor_used", 0.0)
                stage2_reaggregate_gap_band = stage2_output.get("stage2_reaggregate_gap_band", "none")
                stage2_reaggregate_candidate_band = stage2_output.get("stage2_reaggregate_candidate_band", "none")
                stage2_reaggregate_candidate_multiplier_used = stage2_output.get("stage2_reaggregate_candidate_multiplier_used", 1.0)
                stage2_reaggregate_threshold_guard_triggered = stage2_output.get("stage2_reaggregate_threshold_guard_triggered", 0)
                stage2_reaggregate_multiplier_guard_triggered = stage2_output.get("stage2_reaggregate_multiplier_guard_triggered", 0)
                stage2_reaggregate_guard_any_triggered = stage2_output.get("stage2_reaggregate_guard_any_triggered", 0)
                stage2_reaggregate_effective_gap_mid = stage2_output.get("stage2_reaggregate_effective_gap_mid", 0.0)
                stage2_reaggregate_effective_gap_high = stage2_output.get("stage2_reaggregate_effective_gap_high", 0.0)
                stage2_reaggregate_effective_candidate_mid = stage2_output.get("stage2_reaggregate_effective_candidate_mid", 0)
                stage2_reaggregate_effective_candidate_high = stage2_output.get("stage2_reaggregate_effective_candidate_high", 0)
                stage2_bootstrap_contrast_applied = stage2_output.get("stage2_bootstrap_contrast_applied", 0)
                stage2_candidate_count = len(stage2_output.get("stage2_candidates", []))
                stage2_valid_candidate_count = sum(int(item.get("stage2_parse_ok", 0)) for item in stage2_output.get("stage2_candidates", []))
                stage2_ensemble_applied = stage2_output.get("stage2_ensemble_applied", 0)
                stage2_ensemble_agreement = stage2_output.get("stage2_ensemble_agreement", 0.0)
                stage2_ensemble_support = stage2_output.get("stage2_ensemble_support", 0.0)
                result.append({
                    "Id": batch_idx,
                    "Sentence": batch_texts[idx],
                    "Stage1 pass": stage1_pass,
                    "Stage1 reason": stage1_reason,
                    "Stage2 retry reason": stage2_output.get("stage2_retry_reason", ""),
                    "Stage2 retry count": stage2_retry_count,
                    "Stage2 rule applied": stage2_rule_applied,
                    "Stage2 consensus applied": stage2_consensus_applied,
                    "Stage2 consensus conflict": stage2_consensus_conflict,
                    "Stage2 bootstrap contrast applied": stage2_bootstrap_contrast_applied,
                    "Stage2 candidate count": stage2_candidate_count,
                    "Stage2 valid candidate count": stage2_valid_candidate_count,
                    "Stage2 ensemble applied": stage2_ensemble_applied,
                    "Stage2 ensemble agreement": stage2_ensemble_agreement,
                    "Stage2 ensemble support": stage2_ensemble_support,
                    "Stage2 ensemble support threshold hit": stage2_ensemble_support_threshold_hit,
                    "Stage2 low support refined": stage2_low_support_refined,
                    "Stage2 reaggregate applied": stage2_reaggregate_applied,
                    "Stage2 reaggregate support": stage2_reaggregate_support,
                    "Stage2 reaggregate support gain": stage2_reaggregate_support_gain,
                    "Stage2 reaggregate kept": stage2_reaggregate_kept,
                    "Stage2 reaggregate required gain": stage2_reaggregate_required_gain,
                    "Stage2 reaggregate valid candidates": stage2_reaggregate_valid_candidates,
                    "Stage2 reaggregate support gap": stage2_reaggregate_support_gap,
                    "Stage2 reaggregate linked gain bonus": stage2_reaggregate_linked_gain_bonus,
                    "Stage2 reaggregate linked factor used": stage2_reaggregate_linked_factor_used,
                    "Stage2 reaggregate gap band": stage2_reaggregate_gap_band,
                    "Stage2 reaggregate candidate band": stage2_reaggregate_candidate_band,
                    "Stage2 reaggregate candidate multiplier used": stage2_reaggregate_candidate_multiplier_used,
                    "Stage2 reaggregate threshold guard triggered": stage2_reaggregate_threshold_guard_triggered,
                    "Stage2 reaggregate multiplier guard triggered": stage2_reaggregate_multiplier_guard_triggered,
                    "Stage2 reaggregate guard any triggered": stage2_reaggregate_guard_any_triggered,
                    "Stage2 reaggregate effective gap mid": stage2_reaggregate_effective_gap_mid,
                    "Stage2 reaggregate effective gap high": stage2_reaggregate_effective_gap_high,
                    "Stage2 reaggregate effective candidate mid": stage2_reaggregate_effective_candidate_mid,
                    "Stage2 reaggregate effective candidate high": stage2_reaggregate_effective_candidate_high,
                    "Stage3 triggered": 0,
                    "Stage3 trigger reason": "none",
                    "Stage3 status": "skipped",
                    "True hyperbole": row_data["True hyperbole"],
                    "True metaphor": row_data["True metaphor"],
                    "Hyperbole judgment": stage2_hyperbole_judgment,
                    "Hyperbole reason": stage2_hyperbole_reason,
                    "Metaphor judgment": stage2_metaphor_judgment,
                    "Metaphor reason": stage2_metaphor_reason,
                    "Interaction updated": int(init_hyperbole != stage2_hyperbole_judgment or init_metaphor != stage2_metaphor_judgment)
                })
                processed_ids.add(batch_idx)
                newly_processed += 1
        save_stage_outputs(args, emo, domain, meta, hype, result, stage3_candidates)
        print(f"Batch inference completed: {newly_processed} samples")
    else:
        for i, row in tqdm(pf.iterrows(), total=len(pf)):
            if i in processed_ids:
                continue
            text = row["Sentence"]
            true_hyperbole = int(row["Hyperbole"]) if not pd.isna(row["Hyperbole"]) else 0
            true_metaphor = int(row["Metaphor"]) if not pd.isna(row["Metaphor"]) else 0
            stage1_pass, stage1_reason = stage1_counterfactual_filter(text)
            emotion_hint_raw = "none"
            if "Emotion lexicon hint" in pf.columns and not pd.isna(row["Emotion lexicon hint"]):
                emotion_hint_raw = str(row["Emotion lexicon hint"])
            emotion_hint = "none"
            if int(args.use_emotion_lexicon_feature) == 1:
                emotion_hint = build_emotion_lexicon_hint(text, emotion_hint_raw)
            concept_hint = "none"
            if int(args.use_conceptnet_feature) == 1:
                concept_hint = build_conceptnet_hint(text)
    
            if int(args.fast) == 1:
                emotion_analysis_reason, emotion_intensity, target_domain, select_reason, source_domain, generate_reason = fast_stage1_emo_domain(
                    text, MODEL_NAME, emotion_hint, concept_hint
                )
                emo.append({
                    "Id": i,
                    "Sentence": text,
                    "Stage1 pass": stage1_pass,
                    "Stage1 reason": stage1_reason,
                    "Emotion lexicon hint raw": emotion_hint_raw,
                    "Emotion lexicon hint": emotion_hint,
                    "Emotion analysis reason": emotion_analysis_reason,
                    "Emotion intensity": emotion_intensity
                })
                domain.append({
                    "Id": i,
                    "Sentence": text,
                    "Stage1 pass": stage1_pass,
                    "Stage1 reason": stage1_reason,
                    "ConceptNet hint": concept_hint,
                    "Target domain": target_domain,
                    "Select reason": select_reason,
                    "Source domain": source_domain,
                    "Generate reason": generate_reason
                })

                metaphor_judgment, metaphor_reason, hyperbole_judgment, hyperbole_reason, fast_confidence = fast_stage2_meta_hyper(
                    text, emotion_analysis_reason, target_domain, source_domain,
                    emotion_intensity, MODEL_NAME, concept_hint
                )
                meta.append({
                    "Id": i, "Sentence": text,
                    "Stage1 pass": stage1_pass, "Stage1 reason": stage1_reason,
                    "Metaphor judgment": metaphor_judgment, "Metaphor reason": metaphor_reason
                })
                hype.append({
                    "Id": i, "Sentence": text,
                    "Stage1 pass": stage1_pass, "Stage1 reason": stage1_reason,
                    "Metaphor judgment": metaphor_judgment, "Metaphor reason": metaphor_reason,
                    "Hyperbole judgment": hyperbole_judgment, "Hyperbole reason": hyperbole_reason
                })
                init_hyperbole = hyperbole_judgment
                init_metaphor = metaphor_judgment
                stage2_hyperbole_judgment = hyperbole_judgment
                stage2_hyperbole_reason = hyperbole_reason
                stage2_metaphor_judgment = metaphor_judgment
                stage2_metaphor_reason = metaphor_reason
                stage2_confidence = fast_confidence
                stage2_parse_ok = 1
                stage2_error = "none"
                stage2_retried = 0
                stage2_retry_count = 0
                stage2_rule_applied = 0
                stage2_consensus_applied = 0
                stage2_consensus_conflict = 0
                stage2_ensemble_support_threshold_hit = 0
                stage2_low_support_refined = 0
                stage2_reaggregate_applied = 0
                stage2_reaggregate_support = 0.0
                stage2_reaggregate_support_gain = 0.0
                stage2_reaggregate_kept = 0
                stage2_reaggregate_required_gain = 0.0
                stage2_reaggregate_valid_candidates = 0
                stage2_reaggregate_support_gap = 0.0
                stage2_reaggregate_linked_gain_bonus = 0.0
                stage2_reaggregate_linked_factor_used = 0.0
                stage2_reaggregate_gap_band = "none"
                stage2_reaggregate_candidate_band = "none"
                stage2_reaggregate_candidate_multiplier_used = 1.0
                stage2_reaggregate_threshold_guard_triggered = 0
                stage2_reaggregate_multiplier_guard_triggered = 0
                stage2_reaggregate_guard_any_triggered = 0
                stage2_reaggregate_effective_gap_mid = 0.0
                stage2_reaggregate_effective_gap_high = 0.0
                stage2_reaggregate_effective_candidate_mid = 0
                stage2_reaggregate_effective_candidate_high = 0
                stage2_bootstrap_contrast_applied = 0
                stage2_candidate_count = 1
                stage2_valid_candidate_count = 1
                stage2_ensemble_applied = 0
                stage2_ensemble_agreement = 1.0
                stage2_ensemble_support = 1.0
            else:
                emotion_analysis_reason, emotion_intensity = analyze_emo(text, MODEL_NAME, emotion_hint)
                emo.append({
                    "Id": i,
                    "Sentence": text,
                    "Stage1 pass": stage1_pass,
                    "Stage1 reason": stage1_reason,
                    "Emotion lexicon hint raw": emotion_hint_raw,
                    "Emotion lexicon hint": emotion_hint,
                    "Emotion analysis reason": emotion_analysis_reason,
                    "Emotion intensity": emotion_intensity
                })
    
            if int(args.fast) != 1:
                target_domain, select_reason, source_domain, generate_reason = target_source_domains(
                    text, emotion_analysis_reason, MODEL_NAME, emotion_hint, concept_hint
                )
                domain.append({
                    "Id": i,
                    "Sentence": text,
                    "Stage1 pass": stage1_pass,
                    "Stage1 reason": stage1_reason,
                    "ConceptNet hint": concept_hint,
                    "Target domain": target_domain,
                    "Select reason": select_reason,
                    "Source domain": source_domain,
                    "Generate reason": generate_reason
                })
    
                metaphor_judgment, metaphor_reason = metaphor_learning(
                    text, emotion_analysis_reason, target_domain, source_domain, MODEL_NAME, concept_hint
                )
                meta.append({
                    "Id": i,
                    "Sentence": text,
                    "Stage1 pass": stage1_pass,
                    "Stage1 reason": stage1_reason,
                    "Metaphor judgment": metaphor_judgment,
                    "Metaphor reason": metaphor_reason
                })
    
                hyperbole_judgment, hyperbole_reason = hyperbole_learning(
                    text, metaphor_judgment, emotion_analysis_reason, target_domain, source_domain, MODEL_NAME, concept_hint,
                    emotion_intensity=emotion_intensity
                )
                hype.append({
                    "Id": i,
                    "Sentence": text,
                    "Stage1 pass": stage1_pass,
                    "Stage1 reason": stage1_reason,
                    "Metaphor judgment": metaphor_judgment,
                    "Metaphor reason": metaphor_reason,
                    "Hyperbole judgment": hyperbole_judgment,
                    "Hyperbole reason": hyperbole_reason
                })
    
                init_hyperbole = hyperbole_judgment
                init_metaphor = metaphor_judgment
                hyperbole_judgment, hyperbole_reason, metaphor_judgment, metaphor_reason = hyperbole_metaphor(
                    text,
                    emotion_analysis_reason,
                    hyperbole_judgment,
                    hyperbole_reason,
                    metaphor_judgment,
                    metaphor_reason,
                    target_domain,
                    source_domain,
                    MODEL_NAME,
                    concept_hint
                )
                stage2_output = run_stage2_pipeline(
                    args,
                    effective_retry_target,
                    text,
                    emotion_analysis_reason,
                    target_domain,
                    source_domain,
                    hyperbole_judgment,
                    hyperbole_reason,
                    metaphor_judgment,
                    metaphor_reason
                )
                stage2_hyperbole_judgment = stage2_output["stage2_hyperbole_judgment"]
                stage2_hyperbole_reason = stage2_output["stage2_hyperbole_reason"]
                stage2_metaphor_judgment = stage2_output["stage2_metaphor_judgment"]
                stage2_metaphor_reason = stage2_output["stage2_metaphor_reason"]
                stage2_confidence = stage2_output["stage2_confidence"]
                stage2_parse_ok = stage2_output["stage2_parse_ok"]
                stage2_error = stage2_output["stage2_error"]
                stage2_retried = stage2_output["stage2_retried"]
                stage2_retry_count = stage2_output["stage2_retry_count"]
                stage2_rule_applied = stage2_output["stage2_rule_applied"]
                stage2_consensus_applied = stage2_output["stage2_consensus_applied"]
                stage2_consensus_conflict = stage2_output["stage2_consensus_conflict"]
                stage2_ensemble_support_threshold_hit = stage2_output["stage2_ensemble_support_threshold_hit"]
                stage2_low_support_refined = stage2_output["stage2_low_support_refined"]
                stage2_reaggregate_applied = stage2_output["stage2_reaggregate_applied"]
                stage2_reaggregate_support = stage2_output["stage2_reaggregate_support"]
                stage2_reaggregate_support_gain = stage2_output["stage2_reaggregate_support_gain"]
                stage2_reaggregate_kept = stage2_output["stage2_reaggregate_kept"]
                stage2_reaggregate_required_gain = stage2_output["stage2_reaggregate_required_gain"]
                stage2_reaggregate_valid_candidates = stage2_output["stage2_reaggregate_valid_candidates"]
                stage2_reaggregate_support_gap = stage2_output["stage2_reaggregate_support_gap"]
                stage2_reaggregate_linked_gain_bonus = stage2_output["stage2_reaggregate_linked_gain_bonus"]
                stage2_reaggregate_linked_factor_used = stage2_output["stage2_reaggregate_linked_factor_used"]
                stage2_reaggregate_gap_band = stage2_output["stage2_reaggregate_gap_band"]
                stage2_reaggregate_candidate_band = stage2_output["stage2_reaggregate_candidate_band"]
                stage2_reaggregate_candidate_multiplier_used = stage2_output["stage2_reaggregate_candidate_multiplier_used"]
                stage2_reaggregate_threshold_guard_triggered = stage2_output["stage2_reaggregate_threshold_guard_triggered"]
                stage2_reaggregate_multiplier_guard_triggered = stage2_output["stage2_reaggregate_multiplier_guard_triggered"]
                stage2_reaggregate_guard_any_triggered = stage2_output["stage2_reaggregate_guard_any_triggered"]
                stage2_reaggregate_effective_gap_mid = stage2_output["stage2_reaggregate_effective_gap_mid"]
                stage2_reaggregate_effective_gap_high = stage2_output["stage2_reaggregate_effective_gap_high"]
                stage2_reaggregate_effective_candidate_mid = stage2_output["stage2_reaggregate_effective_candidate_mid"]
                stage2_reaggregate_effective_candidate_high = stage2_output["stage2_reaggregate_effective_candidate_high"]
                stage2_bootstrap_contrast_applied = stage2_output["stage2_bootstrap_contrast_applied"]
                stage2_candidate_count = stage2_output["stage2_candidate_count"]
                stage2_valid_candidate_count = stage2_output["stage2_valid_candidate_count"]
                stage2_ensemble_applied = stage2_output["stage2_ensemble_applied"]
                stage2_ensemble_agreement = stage2_output["stage2_ensemble_agreement"]
                stage2_ensemble_support = stage2_output["stage2_ensemble_support"]
            stage3_triggered, stage3_trigger_reason = evaluate_stage3_trigger(
                args,
                stage2_hyperbole_judgment,
                stage2_metaphor_judgment,
                stage2_confidence,
                stage2_parse_ok
            )
            stage3_status = "skipped_no_multimodal" if stage3_triggered == 1 else "not_triggered"
            if stage3_triggered == 1:
                image_col_used, image_path = resolve_media_path_with_key(row, stage3_image_columns)
                audio_col_used, audio_path = resolve_media_path_with_key(row, stage3_audio_columns)
                video_col_used, video_path = resolve_media_path_with_key(row, stage3_video_columns)
                image_path, image_rewrite_hit = rewrite_media_path_by_mappings(image_path, stage3_media_path_mappings)
                audio_path, audio_rewrite_hit = rewrite_media_path_by_mappings(audio_path, stage3_media_path_mappings)
                video_path, video_rewrite_hit = rewrite_media_path_by_mappings(video_path, stage3_media_path_mappings)
                any_rewrite_hit = int(image_rewrite_hit == 1 or audio_rewrite_hit == 1 or video_rewrite_hit == 1)
                if int(args.stage3_media_check_exists) == 1:
                    image_exists, image_resolved, image_resolve_source = resolve_media_path_existence(
                        image_path, args.data_dir, args.stage3_media_base_dir, stage3_media_extra_base_dirs, stage3_media_resolve_order
                    )
                    audio_exists, audio_resolved, audio_resolve_source = resolve_media_path_existence(
                        audio_path, args.data_dir, args.stage3_media_base_dir, stage3_media_extra_base_dirs, stage3_media_resolve_order
                    )
                    video_exists, video_resolved, video_resolve_source = resolve_media_path_existence(
                        video_path, args.data_dir, args.stage3_media_base_dir, stage3_media_extra_base_dirs, stage3_media_resolve_order
                    )
                    multimodal_ready = int(image_exists == 1 or audio_exists == 1 or video_exists == 1)
                else:
                    image_exists = int(image_path != "none")
                    audio_exists = int(audio_path != "none")
                    video_exists = int(video_path != "none")
                    image_resolved = image_path
                    audio_resolved = audio_path
                    video_resolved = video_path
                    image_resolve_source = "skip_exists_check"
                    audio_resolve_source = "skip_exists_check"
                    video_resolve_source = "skip_exists_check"
                    multimodal_ready = int(image_exists == 1 or audio_exists == 1 or video_exists == 1)
                ready_modalities = []
                if image_exists == 1:
                    ready_modalities.append("image")
                if audio_exists == 1:
                    ready_modalities.append("audio")
                if video_exists == 1:
                    ready_modalities.append("video")
                ready_modalities_text = "|".join(ready_modalities) if ready_modalities else "none"
                missing_modalities = []
                if image_exists == 0:
                    missing_modalities.append("image")
                if audio_exists == 0:
                    missing_modalities.append("audio")
                if video_exists == 0:
                    missing_modalities.append("video")
                missing_modalities_text = "|".join(missing_modalities) if missing_modalities else "none"
                if multimodal_ready == 1:
                    stage3_status = "ready_for_multimodal"
                else:
                    stage3_status = "pending_media_alignment"
                stage3_candidates.append({
                    "Id": i,
                    "Sentence": text,
                    "Stage3 input id": f"{args.input_name}:{i}",
                    "Stage3 trigger reason": stage3_trigger_reason,
                    "Stage2 confidence": stage2_confidence,
                    "Stage2 parse ok": stage2_parse_ok,
                    "Hyperbole judgment": stage2_hyperbole_judgment,
                    "Metaphor judgment": stage2_metaphor_judgment,
                    "Target domain": target_domain,
                    "Source domain": source_domain,
                    "Image column used": image_col_used,
                    "Image path": image_path,
                    "Image rewrite hit": image_rewrite_hit,
                    "Image exists": image_exists,
                    "Image resolved path": image_resolved,
                    "Image resolve source": image_resolve_source,
                    "Audio column used": audio_col_used,
                    "Audio path": audio_path,
                    "Audio rewrite hit": audio_rewrite_hit,
                    "Audio exists": audio_exists,
                    "Audio resolved path": audio_resolved,
                    "Audio resolve source": audio_resolve_source,
                    "Video column used": video_col_used,
                    "Video path": video_path,
                    "Video rewrite hit": video_rewrite_hit,
                    "Video exists": video_exists,
                    "Video resolved path": video_resolved,
                    "Video resolve source": video_resolve_source,
                    "Any rewrite hit": any_rewrite_hit,
                    "Ready modalities": ready_modalities_text,
                    "Missing modalities": missing_modalities_text,
                    "Multimodal ready": multimodal_ready
                })
            result.append({
                "Id": i,
                "Sentence": text,
                "Ablation profile": args.ablation_profile,
                "Use emotion lexicon feature": int(args.use_emotion_lexicon_feature),
                "Use conceptnet feature": int(args.use_conceptnet_feature),
                "Stage1 pass": stage1_pass,
                "Stage1 reason": stage1_reason,
                "Stage2 parse ok": stage2_parse_ok,
                "Stage2 error": stage2_error,
                "Stage2 confidence": stage2_confidence,
                "Stage2 retried": stage2_retried,
                "Stage2 retry count": stage2_retry_count,
                "Stage2 retry mode": args.stage2_retry_mode,
                "Stage2 retry target raw": args.stage2_retry_target,
                "Stage2 retry target": effective_retry_target,
                "Stage2 retry target resolution reason": retry_target_resolution_reason,
                "Stage2 rule applied": stage2_rule_applied,
                "Stage2 consensus applied": stage2_consensus_applied,
                "Stage2 consensus conflict": stage2_consensus_conflict,
                "Stage2 bootstrap contrast applied": stage2_bootstrap_contrast_applied,
                "Stage2 candidate count": stage2_candidate_count,
                "Stage2 valid candidate count": stage2_valid_candidate_count,
                "Stage2 ensemble applied": stage2_ensemble_applied,
                "Stage2 ensemble agreement": stage2_ensemble_agreement,
                "Stage2 ensemble support": stage2_ensemble_support,
                "Stage2 ensemble support threshold hit": stage2_ensemble_support_threshold_hit,
                "Stage2 low support refined": stage2_low_support_refined,
                "Stage2 reaggregate applied": stage2_reaggregate_applied,
                "Stage2 reaggregate support": stage2_reaggregate_support,
                "Stage2 reaggregate support gain": stage2_reaggregate_support_gain,
                "Stage2 reaggregate kept": stage2_reaggregate_kept,
                "Stage2 reaggregate required gain": stage2_reaggregate_required_gain,
                "Stage2 reaggregate valid candidates": stage2_reaggregate_valid_candidates,
                "Stage2 reaggregate support gap": stage2_reaggregate_support_gap,
                "Stage2 reaggregate linked gain bonus": stage2_reaggregate_linked_gain_bonus,
                "Stage2 reaggregate linked factor used": stage2_reaggregate_linked_factor_used,
                "Stage2 reaggregate gap band": stage2_reaggregate_gap_band,
                "Stage2 reaggregate candidate band": stage2_reaggregate_candidate_band,
                "Stage2 reaggregate candidate multiplier used": stage2_reaggregate_candidate_multiplier_used,
                "Stage2 reaggregate threshold guard triggered": stage2_reaggregate_threshold_guard_triggered,
                "Stage2 reaggregate multiplier guard triggered": stage2_reaggregate_multiplier_guard_triggered,
                "Stage2 reaggregate guard any triggered": stage2_reaggregate_guard_any_triggered,
                "Stage2 reaggregate effective gap mid": stage2_reaggregate_effective_gap_mid,
                "Stage2 reaggregate effective gap high": stage2_reaggregate_effective_gap_high,
                "Stage2 reaggregate effective candidate mid": stage2_reaggregate_effective_candidate_mid,
                "Stage2 reaggregate effective candidate high": stage2_reaggregate_effective_candidate_high,
                "Stage3 triggered": stage3_triggered,
                "Stage3 trigger reason": stage3_trigger_reason,
                "Stage3 status": stage3_status,
                "True hyperbole": true_hyperbole,
                "True metaphor": true_metaphor,
                "Hyperbole judgment": stage2_hyperbole_judgment,
                "Hyperbole reason": stage2_hyperbole_reason,
                "Metaphor judgment": stage2_metaphor_judgment,
                "Metaphor reason": stage2_metaphor_reason,
                "Interaction updated": int(init_hyperbole != hyperbole_judgment or init_metaphor != metaphor_judgment)
            })
            newly_processed += 1
            if checkpoint_interval > 0 and newly_processed % checkpoint_interval == 0:
                save_stage_outputs(args, emo, domain, meta, hype, result, stage3_candidates)
                print(f"Checkpoint saved: +{newly_processed} new samples")

    save_stage_outputs(args, emo, domain, meta, hype, result, stage3_candidates)
    guard_any_count = sum(int(item.get("Stage2 reaggregate guard any triggered", 0)) for item in result)
    guard_threshold_count = sum(int(item.get("Stage2 reaggregate threshold guard triggered", 0)) for item in result)
    guard_multiplier_count = sum(int(item.get("Stage2 reaggregate multiplier guard triggered", 0)) for item in result)
    print(f"Stage2 guard-any: {guard_any_count}/{len(result)}")
    print(f"Stage2 guard-threshold: {guard_threshold_count}/{len(result)}")
    print(f"Stage2 guard-multiplier: {guard_multiplier_count}/{len(result)}")
    gap_band_counter = {}
    candidate_band_counter = {}
    retry_reason_counter = {
        RETRY_TARGET_REASON_EXPLICIT: 0,
        RETRY_TARGET_REASON_AUTO_HYPOL: 0,
        RETRY_TARGET_REASON_AUTO_DEFAULT: 0
    }
    for item in result:
        gap_band = str(item.get("Stage2 reaggregate gap band", "none"))
        gap_band_counter[gap_band] = gap_band_counter.get(gap_band, 0) + 1
        candidate_band = str(item.get("Stage2 reaggregate candidate band", "none"))
        candidate_band_counter[candidate_band] = candidate_band_counter.get(candidate_band, 0) + 1
        retry_reason = normalize_retry_target_resolution_reason(
            item.get("Stage2 retry target resolution reason", retry_target_resolution_reason)
        )
        retry_reason_counter[retry_reason] = retry_reason_counter.get(retry_reason, 0) + 1
    for band in sorted(gap_band_counter.keys()):
        print(f"Stage2 gap-band[{band}]: {gap_band_counter[band]}")
    for band in sorted(candidate_band_counter.keys()):
        print(f"Stage2 candidate-band[{band}]: {candidate_band_counter[band]}")
    print(f"Stage2 retry-target(raw/effective): {args.stage2_retry_target} -> {effective_retry_target}")
    print(f"Stage2 retry-target-resolution: {retry_target_resolution_reason}")
    print(f"Stage2 retry-reason[explicit]: {retry_reason_counter.get(RETRY_TARGET_REASON_EXPLICIT, 0)}")
    print(f"Stage2 retry-reason[auto:hypol]: {retry_reason_counter.get(RETRY_TARGET_REASON_AUTO_HYPOL, 0)}")
    print(f"Stage2 retry-reason[auto:default]: {retry_reason_counter.get(RETRY_TARGET_REASON_AUTO_DEFAULT, 0)}")
    total_count = len(result)
    retry_reason_unique_count = sum(int(retry_reason_counter.get(key, 0) > 0) for key in retry_reason_counter.keys())
    retry_reason_dominant = max(
        [RETRY_TARGET_REASON_EXPLICIT, RETRY_TARGET_REASON_AUTO_HYPOL, RETRY_TARGET_REASON_AUTO_DEFAULT],
        key=lambda key: retry_reason_counter.get(key, 0)
    )
    retry_reason_dominant_count = int(retry_reason_counter.get(retry_reason_dominant, 0))
    retry_reason_non_dominant_count = int(max(total_count - retry_reason_dominant_count, 0))
    retry_reason_mismatch_ratio = float(retry_reason_non_dominant_count) / float(max(total_count, 1))
    retry_reason_alert_threshold = max(float(args.stage2_retry_reason_alert_threshold), 0.0)
    retry_reason_alert_flag = int(retry_reason_mismatch_ratio > retry_reason_alert_threshold)
    retry_reason_alert_excess = max(retry_reason_mismatch_ratio - retry_reason_alert_threshold, 0.0)
    retry_reason_alert_level = "high" if retry_reason_alert_excess >= 0.10 else ("medium" if retry_reason_alert_excess >= 0.03 else ("low" if retry_reason_alert_flag == 1 else "none"))
    retry_reason_alert_score = 3 if retry_reason_alert_level == "high" else (2 if retry_reason_alert_level == "medium" else (1 if retry_reason_alert_level == "low" else 0))
    retry_reason_needs_review = int(retry_reason_alert_score >= 2)
    print(f"Stage2 retry-reason-unique-count: {retry_reason_unique_count}")
    print(f"Stage2 retry-reason-dominant: {retry_reason_dominant}")
    print(f"Stage2 retry-reason-non-dominant: {retry_reason_non_dominant_count}/{len(result)}")
    print(f"Stage2 retry-reason-alert-threshold: {retry_reason_alert_threshold}")
    print(f"Stage2 retry-reason-alert-flag: {retry_reason_alert_flag}")
    print(f"Stage2 retry-reason-alert-level: {retry_reason_alert_level}")
    print(f"Stage2 retry-reason-alert-score: {retry_reason_alert_score}")
    print(f"Stage2 retry-reason-needs-review: {retry_reason_needs_review}")
    denom = max(total_count, 1)
    stage2_guard_summary = [{
        "Total samples": total_count,
        "Ablation profile": str(args.ablation_profile),
        "Use emotion lexicon feature": int(args.use_emotion_lexicon_feature),
        "Use conceptnet feature": int(args.use_conceptnet_feature),
        "Run name": str(resolved_run_name),
        "Input name": str(args.input_name),
        "Stage2 retry target raw": str(args.stage2_retry_target),
        "Effective retry target": str(effective_retry_target),
        "Retry target resolution reason": str(retry_target_resolution_reason),
        "Stage2 retry mode": str(args.stage2_retry_mode),
        "Stage2 consensus enable": int(args.stage2_consensus_enable),
        "Stage2 consensus threshold": float(args.stage2_consensus_threshold),
        "Stage2 consensus mode": str(args.stage2_consensus_mode),
        "Stage2 rule filter": int(args.stage2_rule_filter),
        "Stage2 metaphor none-domain max conf": float(args.stage2_metaphor_none_domain_max_conf),
        "Results dir": str(args.results_dir),
        "Stage2 aggregate enable": int(args.stage2_aggregate_enable),
        "Stage2 aggregate min valid candidates": int(args.stage2_aggregate_min_valid_candidates),
        "Stage2 bootstrap contrast enable": int(args.stage2_bootstrap_contrast_enable),
        "Stage2 ensemble support threshold": float(args.stage2_ensemble_support_threshold),
        "Stage2 low support refine enable": int(args.stage2_low_support_refine_enable),
        "Stage2 reaggregate min support gain": float(args.stage2_reaggregate_min_support_gain),
        "Stage2 reaggregate adaptive gain enable": int(args.stage2_reaggregate_adaptive_gain_enable),
        "Stage2 reaggregate adaptive gain step": float(args.stage2_reaggregate_adaptive_gain_step),
        "Stage2 reaggregate support gap link enable": int(args.stage2_reaggregate_support_gap_link_enable),
        "Stage2 reaggregate support gap link factor": float(args.stage2_reaggregate_support_gap_link_factor),
        "Stage2 reaggregate support gap mid": float(args.stage2_reaggregate_support_gap_mid),
        "Stage2 reaggregate support gap high": float(args.stage2_reaggregate_support_gap_high),
        "Stage2 reaggregate support gap mid multiplier": float(args.stage2_reaggregate_support_gap_mid_multiplier),
        "Stage2 reaggregate support gap high multiplier": float(args.stage2_reaggregate_support_gap_high_multiplier),
        "Stage2 reaggregate candidate mid": int(args.stage2_reaggregate_candidate_mid),
        "Stage2 reaggregate candidate high": int(args.stage2_reaggregate_candidate_high),
        "Stage2 reaggregate candidate mid multiplier": float(args.stage2_reaggregate_candidate_mid_multiplier),
        "Stage2 reaggregate candidate high multiplier": float(args.stage2_reaggregate_candidate_high_multiplier),
        "Guard any count": guard_any_count,
        "Guard threshold count": guard_threshold_count,
        "Guard multiplier count": guard_multiplier_count,
        "Guard any ratio": float(guard_any_count) / float(denom),
        "Guard threshold ratio": float(guard_threshold_count) / float(denom),
        "Guard multiplier ratio": float(guard_multiplier_count) / float(denom),
        "Gap band base count": int(gap_band_counter.get("base", 0)),
        "Gap band mid count": int(gap_band_counter.get("mid", 0)),
        "Gap band high count": int(gap_band_counter.get("high", 0)),
        "Gap band none count": int(gap_band_counter.get("none", 0)),
        "Candidate band base count": int(candidate_band_counter.get("base", 0)),
        "Candidate band mid count": int(candidate_band_counter.get("mid", 0)),
        "Candidate band high count": int(candidate_band_counter.get("high", 0)),
        "Candidate band none count": int(candidate_band_counter.get("none", 0)),
        "Retry reason explicit count": int(retry_reason_counter.get(RETRY_TARGET_REASON_EXPLICIT, 0)),
        "Retry reason auto:hypol count": int(retry_reason_counter.get(RETRY_TARGET_REASON_AUTO_HYPOL, 0)),
        "Retry reason auto:default count": int(retry_reason_counter.get(RETRY_TARGET_REASON_AUTO_DEFAULT, 0)),
        "Retry reason explicit ratio": float(retry_reason_counter.get(RETRY_TARGET_REASON_EXPLICIT, 0)) / float(denom),
        "Retry reason auto:hypol ratio": float(retry_reason_counter.get(RETRY_TARGET_REASON_AUTO_HYPOL, 0)) / float(denom),
        "Retry reason auto:default ratio": float(retry_reason_counter.get(RETRY_TARGET_REASON_AUTO_DEFAULT, 0)) / float(denom),
        "Retry reason unique count": int(retry_reason_unique_count),
        "Retry reason dominant": str(retry_reason_dominant),
        "Retry reason dominant count": int(retry_reason_dominant_count),
        "Retry reason non-dominant count": int(retry_reason_non_dominant_count),
        "Retry reason mismatch ratio": float(retry_reason_mismatch_ratio),
        "Retry reason mismatch flag": int(retry_reason_unique_count > 1),
        "Retry reason alert threshold": float(retry_reason_alert_threshold),
        "Retry reason alert flag": int(retry_reason_alert_flag),
        "Retry reason alert excess": float(retry_reason_alert_excess),
        "Retry reason alert level": str(retry_reason_alert_level),
        "Retry reason alert score": int(retry_reason_alert_score),
        "Retry reason needs review": int(retry_reason_needs_review)
    }]
    store_data(stage2_guard_summary, "stage2_guard_summary.csv", args.results_dir)
    if int(args.stage3_enable) == 1:
        print(f"Stage3 candidates: {len(stage3_candidates)}/{len(result)}")
        ready_count = sum(int(item.get("Multimodal ready", 0)) for item in stage3_candidates)
        print(f"Stage3 media-ready: {ready_count}/{len(stage3_candidates)}")
        image_ready_count = sum(int(item.get("Image exists", 0)) for item in stage3_candidates)
        audio_ready_count = sum(int(item.get("Audio exists", 0)) for item in stage3_candidates)
        video_ready_count = sum(int(item.get("Video exists", 0)) for item in stage3_candidates)
        print(f"Stage3 image-ready: {image_ready_count}/{len(stage3_candidates)}")
        print(f"Stage3 audio-ready: {audio_ready_count}/{len(stage3_candidates)}")
        print(f"Stage3 video-ready: {video_ready_count}/{len(stage3_candidates)}")
        all_missing_count = sum(int(item.get("Missing modalities", "none") == "image|audio|video") for item in stage3_candidates)
        print(f"Stage3 all-missing: {all_missing_count}/{len(stage3_candidates)}")
        rewrite_hit_count = sum(int(item.get("Any rewrite hit", 0)) for item in stage3_candidates)
        print(f"Stage3 rewrite-hit: {rewrite_hit_count}/{len(stage3_candidates)}")
        resolve_source_counter = {}
        for item in stage3_candidates:
            for key in ["Image resolve source", "Audio resolve source", "Video resolve source"]:
                source = str(item.get(key, "none"))
                resolve_source_counter[source] = resolve_source_counter.get(source, 0) + 1
        for source in sorted(resolve_source_counter.keys()):
            print(f"Stage3 resolve-source[{source}]: {resolve_source_counter[source]}")


if __name__ == "__main__":
    main()
