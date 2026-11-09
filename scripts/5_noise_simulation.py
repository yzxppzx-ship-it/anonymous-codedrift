import numpy as np
import os
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

print("Loading feature data from results/...")
y = np.load('../results/gcb_y.npy')
X_code = np.load('../results/gcb_X_code.npy')
X_comment = np.load('../results/gcb_X_comment.npy')
X_concat = np.concatenate([X_code, X_comment, np.abs(X_code - X_comment)], axis=1)

cos_scores = np.array([np.dot(X_code[i], X_comment[i]) / (np.linalg.norm(X_code[i]) * np.linalg.norm(X_comment[i]) + 1e-8) for i in range(len(y))])

def eval_probe(y_target, X_data, budget=100):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probe_aucs = []
    for train_idx, test_idx in skf.split(X_data, y_target):
        X_tr, y_tr = X_data[train_idx], y_target[train_idx]
        X_te, y_te = X_data[test_idx], y_target[test_idx]
        fold_preds = []
        for _ in range(5):
            pos = np.where(y_tr == 1)[0]; neg = np.where(y_tr == 0)[0]
            n_pos = max(1, int(budget * len(pos) / len(y_tr))); n_neg = max(1, budget - n_pos)
            n_pos = min(n_pos, len(pos)); n_neg = min(n_neg, len(neg))
            idx = np.concatenate([np.random.choice(pos, n_pos, replace=False), np.random.choice(neg, n_neg, replace=False)])
            
            n_components = min(32, len(idx) - 1)
            if n_components < 1: n_components = 1
            
            clf = make_pipeline(StandardScaler(), PCA(n_components=n_components), LogisticRegression(max_iter=2000))
            clf.fit(X_tr[idx], y_tr[idx])
            fold_preds.append(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
        probe_aucs.append(np.mean(fold_preds))
    return np.mean(probe_aucs)

print("=== Label Noise Simulation Experiment (Table 4) ===")
base_z = roc_auc_score(y, cos_scores)
base_p = eval_probe(y, X_concat, 100)
print(f"0% noise | Zero-shot AUC: {base_z:.4f} | 100-label probe: {base_p:.4f} | boundary holds: {base_z > base_p}")

rng = np.random.default_rng(42)
for noise in [0.05, 0.10, 0.20, 0.30]:
    y_noisy = y.copy()
    n_flip = int(len(y) * noise)
    flip_idx = rng.choice(len(y), n_flip, replace=False)
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
    
    z_auc = roc_auc_score(y_noisy, cos_scores)
    p_auc = eval_probe(y_noisy, X_concat, 100)
    print(f"{int(noise*100):2d}% noise | Zero-shot AUC: {z_auc:.4f} | 100-label probe: {p_auc:.4f} | boundary holds: {z_auc > p_auc}")