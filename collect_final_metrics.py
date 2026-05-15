import os
import argparse
import pandas as pd


def find_column(df, candidates):
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for item in candidates:
        key = str(item).strip().lower()
        if key in lowered:
            return lowered[key]
    return None


def safe_int_series(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def binary_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def infer_dataset_variant(root_dir, file_path):
    rel = os.path.relpath(os.path.dirname(file_path), root_dir).replace("\\", "/")
    parts = [p for p in rel.split("/") if p]
    dataset = parts[-2] if len(parts) >= 2 else "unknown"
    variant = parts[-1] if len(parts) >= 1 else "unknown"
    if dataset == "unknown":
        token = variant.lower()
        if "hypo-l" in token or "hypol" in token:
            dataset = "hypo-l"
        elif "hypo" in token:
            dataset = "hypo"
        elif "trofi" in token:
            dataset = "trofi"
        elif "lcc" in token:
            dataset = "lcc"
    return dataset, variant


def collect(root_dir):
    rows = []
    for current_root, _, files in os.walk(root_dir):
        if "stage_final.csv" not in files:
            continue
        stage_final_path = os.path.join(current_root, "stage_final.csv")
        guard_path = os.path.join(current_root, "stage2_guard_summary.csv")
        try:
            df = pd.read_csv(stage_final_path)
        except Exception:
            continue
        true_h_col = find_column(df, ["True hyperbole", "true_hyperbole", "hyper_label", "Hyperbole label"])
        pred_h_col = find_column(df, ["Hyperbole judgment", "hyperbole_judgment", "Hyperbole prediction"])
        true_m_col = find_column(df, ["True metaphor", "true_metaphor", "meta_label", "Metaphor label"])
        pred_m_col = find_column(df, ["Metaphor judgment", "metaphor_judgment", "Metaphor prediction"])
        if not all([true_h_col, pred_h_col, true_m_col, pred_m_col]):
            continue
        y_h_true = safe_int_series(df[true_h_col])
        y_h_pred = safe_int_series(df[pred_h_col])
        y_m_true = safe_int_series(df[true_m_col])
        y_m_pred = safe_int_series(df[pred_m_col])
        p_h, r_h, f1_h = binary_metrics(y_h_true, y_h_pred)
        p_m, r_m, f1_m = binary_metrics(y_m_true, y_m_pred)
        dataset, variant = infer_dataset_variant(root_dir, stage_final_path)
        rows.append({
            "dataset": dataset,
            "variant": variant,
            "samples": int(len(df)),
            "hyper_precision": p_h,
            "hyper_recall": r_h,
            "hyper_f1": f1_h,
            "meta_precision": p_m,
            "meta_recall": r_m,
            "meta_f1": f1_m,
            "has_stage2_guard_summary": int(os.path.exists(guard_path)),
            "stage_final_path": stage_final_path,
            "stage2_guard_summary_path": guard_path if os.path.exists(guard_path) else "missing"
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="results/runs")
    parser.add_argument("--out-dir", default="results/tables")
    args = parser.parse_args()
    rows = collect(args.root_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "main_metrics.csv")
    if rows:
        df = pd.DataFrame(rows).sort_values(by=["dataset", "variant"]).reset_index(drop=True)
        df.to_csv(out_path, index=False)
        missing_guard = int((df["has_stage2_guard_summary"] == 0).sum())
        print(f"rows={len(df)}")
        print(f"missing_stage2_guard_summary={missing_guard}")
        print(f"saved={out_path}")
    else:
        pd.DataFrame(columns=[
            "dataset", "variant", "samples",
            "hyper_precision", "hyper_recall", "hyper_f1",
            "meta_precision", "meta_recall", "meta_f1",
            "has_stage2_guard_summary", "stage_final_path", "stage2_guard_summary_path"
        ]).to_csv(out_path, index=False)
        print("rows=0")
        print(f"saved={out_path}")


if __name__ == "__main__":
    main()
