import json, torch, numpy as np, os
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('../data/labels_380.json', 'r', encoding='utf-8') as f:
    ann = json.load(f)

with open('../data/temporal_benchmark_candidates.json', 'rb') as f:
    cand = json.loads(f.read().decode('utf-8'))

bench = []
for k, v in ann.items():
    idx = int(k)
    if idx < len(cand):
        c = cand[idx].copy(); c['label'] = v; bench.append(c)

y = np.array([1 if c['label'] == 'NO_DRIFT' else 0 for c in bench])

print("Extracting features...")
model_path = '../graphcodebert_finetuned'
if not os.path.exists(model_path):
    print(f"Model path {model_path} not found. Using base model.")
    tok = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    mdl = AutoModel.from_pretrained('microsoft/codebert-base').to(device).eval()
else:
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModel.from_pretrained(model_path).to(device).eval()

def tokenize(text):
    return tok(text, truncation=True, max_length=256, padding='max_length', return_tensors='pt')

def emb(text):
    inp = tokenize(text)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad(): out = mdl(**inp)
    return out.last_hidden_state[:, 0, :].squeeze().float().cpu().numpy()

X_code, X_comment, X_concat = [], [], []
for c in tqdm(bench, desc="Extracting"):
    v1 = emb(c['current_code']); v2 = emb(c['historical_comment'])
    X_code.append(v1); X_comment.append(v2)
    X_concat.append(np.concatenate([v1, v2, np.abs(v1 - v2)]))

X_code = np.array(X_code); X_comment = np.array(X_comment); X_concat = np.array(X_concat)

np.save('../results/gcb_y.npy', y)
np.save('../results/gcb_X_code.npy', X_code)
np.save('../results/gcb_X_comment.npy', X_comment)
print("Saved feature .npy files to results/.")

print("\nRunning Label Budget Curve...")
budgets = [10, 25, 50, 100, 200]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_lr = {b: [] for b in budgets}

cos_scores = np.array([np.dot(X_code[i], X_comment[i]) / (np.linalg.norm(X_code[i]) * np.linalg.norm(X_comment[i]) + 1e-8) for i in range(len(y))])
print(f"[Budget 0] Zero-shot AUC: {roc_auc_score(y, cos_scores):.4f}")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_concat, y)):
    X_train_all, y_train_all = X_concat[train_idx], y[train_idx]
    X_test, y_test = X_concat[test_idx], y[test_idx]
    
    for b in budgets:
        actual_b = min(b, len(X_train_all))
        sample_aucs = []
        for _ in range(5):
            idx_pos = np.where(y_train_all == 1)[0]; idx_neg = np.where(y_train_all == 0)[0]
            n_pos = max(1, int(actual_b * len(idx_pos) / len(y_train_all)))
            n_neg = max(1, actual_b - n_pos)
            sampled_idx = np.concatenate([np.random.choice(idx_pos, min(n_pos, len(idx_pos)), replace=False),
                                          np.random.choice(idx_neg, min(n_neg, len(idx_neg)), replace=False)])
            X_sample, y_sample = X_train_all[sampled_idx], y_train_all[sampled_idx]
            
            n_components = min(32, len(sampled_idx) - 1)
            if n_components < 1: n_components = 1
            
            clf = make_pipeline(StandardScaler(), PCA(n_components=n_components), LogisticRegression(max_iter=2000, C=1.0))
            clf.fit(X_sample, y_sample)
            sample_aucs.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        results_lr[b].append(np.mean(sample_aucs))

print("\n=== Upgraded LR Probe Results (Figure 4) ===")
for b in budgets:
    print(f"Budget={b:<4} | AUC = {np.mean(results_lr[b]):.4f} +/- {np.std(results_lr[b]):.4f}")