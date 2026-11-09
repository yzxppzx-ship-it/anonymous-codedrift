import json
import random
import math
import numpy as np
import torch
import os
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import GaussianRandomProjection
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

random.seed(42); np.random.seed(42); torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# Common Utilities
# ==========================================
def embed_all_hf(model_path, name, comments, codes, y_true):
    print(f"Loading {name} from {model_path} ...")
    
    # Robust path checking to prevent confusing Hugging Face errors
    if model_path.startswith('.') and not os.path.exists(model_path):
        raise FileNotFoundError(f"Local model path '{model_path}' not found. Please make sure the fine-tuned model folder is copied to the root directory.")
    
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModel.from_pretrained(model_path).to(device).eval()
    def emb(text):
        inp = tok(text, return_tensors='pt', truncation=True, max_length=256, padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad(): out = mdl(**inp)
        return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    scores = np.array([cos(emb(cd), emb(c)) for c, cd in zip(comments, codes)])
    return scores

def embed_all_st(model_path, name, comments, codes, y_true):
    print(f"Loading {name} from {model_path} ...")
    try:
        model = SentenceTransformer(model_path, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load SentenceTransformer {model_path}. Error: {e}")
    def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    scores = np.array([cos(model.encode(cd), model.encode(c)) for c, cd in zip(comments, codes)])
    return scores

def paired_bootstrap(s1, s2, y_true, n=1000):
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[idx])) < 2: continue
        diffs.append(roc_auc_score(y_true[idx], s1[idx]) - roc_auc_score(y_true[idx], s2[idx]))
    diffs = np.array(diffs)
    return np.mean(diffs), np.percentile(diffs, [2.5, 97.5]), 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))

def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count; n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    delongcov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, delongcov

def delong_test(y_true, pred1, pred2):
    order = np.argsort(-np.array(y_true))
    y_true = np.array(y_true)[order]
    pred1 = np.array(pred1)[order]; pred2 = np.array(pred2)[order]
    predictions_sorted_transposed = np.vstack((pred1, pred2))
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, int(np.sum(y_true)))
    l = np.array([[1, -1]])
    var_diff = np.dot(np.dot(l, delongcov), l.T)[0, 0]
    z = (aucs[0] - aucs[1]) ** 2 / var_diff
    return aucs[0], aucs[1], 2 * (1 - math.erf(np.sqrt(z) / np.sqrt(2)))

# ==========================================
# Load Data
# ==========================================
with open('../data/labels_380.json', 'r', encoding='utf-8') as f:
    ann_380 = json.load(f)
with open('../data/temporal_benchmark_candidates.json', 'rb') as f:
    cand_380 = json.loads(f.read().decode('utf-8'))

bench_380 = []
for k, v in ann_380.items():
    idx = int(k)
    if idx < len(cand_380):
        c = cand_380[idx].copy(); c['label'] = v; bench_380.append(c)

y_380 = np.array([1 if c['label'] == 'NO_DRIFT' else 0 for c in bench_380])
comments_380 = [c['historical_comment'] for c in bench_380]
codes_380 = [c['current_code'] for c in bench_380]

# ==========================================
# PART 1: Reproduce Table 2 (n=200 Balanced)
# ==========================================
print("\n--- Reproducing Table 2 (n=200 Balanced Benchmark) ---")
if os.path.exists('../data/balanced_200.json'):
    with open('../data/balanced_200.json', encoding='utf-8') as f:
        bench_200 = json.load(f)
else:
    print(">>> balanced_200.json not found. Creating it from 380 pool (100 Drift + 100 No-Drift)...")
    drifts = [c for c in bench_380 if c['label'] == 'DRIFT']
    no_drifts = [c for c in bench_380 if c['label'] == 'NO_DRIFT']
    random.seed(42)
    bench_200 = drifts[:100] + no_drifts[:100]
    with open('../data/balanced_200.json', 'w', encoding='utf-8') as f:
        json.dump(bench_200, f, indent=2, ensure_ascii=False)
    print(">>> balanced_200.json created successfully!")

y_200 = np.array([1 if c['label'] == 'NO_DRIFT' else 0 for c in bench_200])
comments_200 = [c['historical_comment'] for c in bench_200]
codes_200 = [c['current_code'] for c in bench_200]

s_jac_200 = np.array([len(set(c.lower().split()) & set(cd.lower().split())) / len(set(c.lower().split()) | set(cd.lower().split())) if set(c.lower().split()) and set(cd.lower().split()) else 0.0 for c, cd in zip(comments_200, codes_200)])
s_zs_200 = embed_all_hf('microsoft/codebert-base', 'Zero-Shot CodeBERT', comments_200, codes_200, y_200)
s_cb_200 = embed_all_hf('../finetuned_model', 'Ours-CodeBERT', comments_200, codes_200, y_200)
s_gcb_200 = embed_all_hf('../graphcodebert_finetuned', 'Ours-GraphCodeBERT', comments_200, codes_200, y_200)
s_nightowl_200 = embed_all_st('Shuu12121/NightOwl-CodeEmbedding-35M', 'NightOwl-35M', comments_200, codes_200, y_200)
s_sbert_200 = embed_all_st('all-MiniLM-L6-v2', 'SentenceBERT', comments_200, codes_200, y_200)

print("\n=== Fixed theta=0.5 (Table 2) ===")
print(f"Jaccard AUC: {roc_auc_score(y_200, s_jac_200):.4f}")
print(f"Zero-Shot CodeBERT AUC: {roc_auc_score(y_200, s_zs_200):.4f}")
print(f"Ours-CodeBERT AUC: {roc_auc_score(y_200, s_cb_200):.4f}")
print(f"Ours-GraphCodeBERT AUC: {roc_auc_score(y_200, s_gcb_200):.4f}")
print(f"NightOwl-35M AUC: {roc_auc_score(y_200, s_nightowl_200):.4f}")
print(f"SentenceBERT AUC: {roc_auc_score(y_200, s_sbert_200):.4f}")

# ==========================================
# PART 2: Reproduce Table 3 (n=380 Pool)
# ==========================================
print("\n\n--- Reproducing Table 3 (n=380 Pool Statistical Tests) ---")

try:
    s_gcb_380 = np.load('../results/s_gcb_380_scores.npy')
    s_nightowl_380 = np.load('../results/s_nightowl_380_scores.npy')
    s_sbert_380 = np.load('../results/s_sbert_380_scores.npy')
    s_unft_gcb_380 = np.load('../results/s_unft_gcb_380_scores.npy')
    print(">>> Successfully loaded pre-computed n=380 prediction scores from `results/`.")
except FileNotFoundError:
    print(">>> Pre-computed scores not found. Generating them now (Requires internet)...")
    s_gcb_380 = embed_all_hf('../graphcodebert_finetuned', 'Ours-GraphCodeBERT (380)', comments_380, codes_380, y_380)
    s_unft_gcb_380 = embed_all_hf('microsoft/graphcodebert-base', 'Un-fine-tuned GCB (380)', comments_380, codes_380, y_380)
    s_nightowl_380 = embed_all_st('Shuu12121/NightOwl-CodeEmbedding-35M', 'NightOwl-35M (380)', comments_380, codes_380, y_380)
    s_sbert_380 = embed_all_st('all-MiniLM-L6-v2', 'SentenceBERT (380)', comments_380, codes_380, y_380)
    
    os.makedirs('../results', exist_ok=True)
    np.save('../results/s_gcb_380_scores.npy', s_gcb_380)
    np.save('../results/s_unft_gcb_380_scores.npy', s_unft_gcb_380)
    np.save('../results/s_nightowl_380_scores.npy', s_nightowl_380)
    np.save('../results/s_sbert_380_scores.npy', s_sbert_380)
    print(">>> Scores saved to results/ for exact reproducibility.")

print("\n--- Ours-GCB vs NightOwl-35M (n=380) ---")
diff, ci, p_boot = paired_bootstrap(s_gcb_380, s_nightowl_380, y_380)
auc1, auc2, p_delong = delong_test(y_380, s_gcb_380, s_nightowl_380)
print(f"diff={diff:.4f}, 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}], Boot_p={p_boot:.4f}, DeLong_p={p_delong:.4f}")

print("\n--- Ours-GCB vs Un-fine-tuned GCB (n=380) ---")
diff, ci, p_boot = paired_bootstrap(s_gcb_380, s_unft_gcb_380, y_380)
auc1, auc2, p_delong = delong_test(y_380, s_gcb_380, s_unft_gcb_380)
print(f"diff={diff:.4f}, 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}], Boot_p={p_boot:.4f}, DeLong_p={p_delong:.4f}")

print("\n--- Ours-GCB vs SentenceBERT (n=380) ---")
diff, ci, p_boot = paired_bootstrap(s_gcb_380, s_sbert_380, y_380)
auc1, auc2, p_delong = delong_test(y_380, s_gcb_380, s_sbert_380)
print(f"diff={diff:.4f}, 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}], Boot_p={p_boot:.4f}, DeLong_p={p_delong:.4f}")