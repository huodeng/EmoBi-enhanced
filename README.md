# EmoBi — Enhanced Metaphor & Hyperbole Detection

This repository contains the enhanced implementation of the EmoBi framework for our graduation thesis, extending the original paper:

> *"Enhancing Hyperbole and Metaphor Detection with Their Bidirectional Dynamic Interaction and Emotion Knowledge"* (Zheng et al., arXiv 2506.15504, 2025)

## Key Contributions

| Contribution | Description |
|---|---|
| **Multi-LLM Backend** | Supports Groq (Llama 8B/70B) and DeepSeek V3 via OpenAI-compatible API |
| **5-Step Reasoning Pipeline** | Emotion → Domain Mapping → Metaphor → Hyperbole → Bidirectional Verification |
| **Fast Mode (2-step)** | Compressed pipeline for speed-constrained scenarios |
| **5-Group Ablation** | `wo_feature` / `plus_e` / `plus_c` / `wo_cascade` / `full` |
| **4-Dataset Coverage** | HYPO, HYPO-L, LCC, TroFi |
| **Stage-2 Cascade** | JSON refinement, consensus voting, reaggregation, guard monitoring |
| **Unified Metrics** | `collect_final_metrics.py` auto-generates paper-ready tables |

## Main Results

### Model Scale Comparison (HYPO-L, 5-step pipeline, n=30)

| Model | Profile | Hyper F1 | Meta F1 | Meta Recall |
|---|---|---|---|---|
| Llama-3.1-8B | full | 0.909 | 0.750 | 0.938 |
| Llama-3.3-70B | full | 0.667* | 0.800* | 1.000* |
| **DeepSeek V3** | full | 0.571 | **0.769** | 0.938 |

### 5-Step vs Fast Mode (HYPO, 8B, n=30)

| Architecture | LLM Calls/Sample | Meta F1 | Meta Recall |
|---|---|---|---|
| Fast Mode (2-step) | 2 | 0.533 | 0.457 |
| **5-Step Pipeline** | 5–6 | **0.792** | **0.826** |

> *Partial run (n=10). Full 70B 5-step limited by Groq rate-limit.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirement.txt
```

### 2. Set API key
Create `.env.local` in the project root:
```
GROQ_API_KEY=your_api_key_here
```

### 3. Run
```bash
# Default: DeepSeek V3, 5-step pipeline, HYPO-L, n=30
python main.py

# Custom run
python main.py --input-name hypo.csv --ablation-profile full --max-samples 50

# Fast mode (2-step)
python main.py --fast 1

# Batch ablation (5 groups)
.\run_ablation_final.ps1
```

### 4. Collect metrics
```bash
python collect_final_metrics.py
# → results/tables/main_metrics.csv
```

## Supported LLM Providers

| Provider | Config | Model |
|---|---|---|
| DeepSeek | `GROQ_API_URL=https://api.deepseek.com/v1/chat/completions` | `deepseek-chat` |
| Groq 8B | `GROQ_API_URL=https://api.groq.com/openai/v1/chat/completions` | `llama-3.1-8b-instant` |
| Groq 70B | same URL | `llama-3.3-70b-versatile` |

Edit `config.py` to switch providers.

## Project Structure

```
├── main.py                    # Main pipeline (5-step default)
├── model.py                   # LLM calls + prompt functions
├── config.py                  # Model / API / experiment config
├── utils.py                   # CSV I/O, hint builders, parsers
├── collect_final_metrics.py   # Metrics aggregator
├── plot_ablation.py           # Ablation chart generator
├── run_ablation_final.ps1     # Batch ablation script
├── requirement.txt            # Python dependencies
├── .gitignore
├── README.md
├── docs/                      # Design notes, plans, slides
│   ├── notes/
│   ├── plans/4.1/
│   ├── references/
│   └── slides/
├── results/
│   ├── runs/                  # Per-experiment output
│   └── tables/                # main_metrics.csv, ablation_chart.png
└── multitask_hyperbole_metaphor_detection-main/
    └── data/                  # HYPO / HYPO-L / LCC / TroFi CSV files
```

## Ablation Profiles

| `--ablation-profile` | Emotion Lexicon | ConceptNet | Cascade |
|---|---|---|---|
| `wo_feature` | ✗ | ✗ | ✓ |
| `plus_e` | ✓ | ✗ | ✓ |
| `plus_c` | ✗ | ✓ | ✓ |
| `wo_cascade` | ✓ | ✓ | ✗ |
| `full` | ✓ | ✓ | ✓ |

## Citation

If you use this code, please cite both the original paper and this implementation:

```bibtex
@misc{zheng2025enhancinghyperbolemetaphordetection,
  title={Enhancing Hyperbole and Metaphor Detection with Their 
         Bidirectional Dynamic Interaction and Emotion Knowledge},
  author={Li Zheng and Sihang Wang and Hao Fei and Zuquan Peng and 
          Fei Li and Jianming Fu and Chong Teng and Donghong Ji},
  year={2025},
  eprint={2506.15504},
  archivePrefix={arXiv}
}
```
