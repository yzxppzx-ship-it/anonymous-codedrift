import json, torch, random, numpy as np, torch.nn.functional as F, os, gc
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch does not support CUDA!")
device = torch.device('cuda')
print(f"Successfully enabled local GPU: {torch.cuda.get_device_name(0)}")

SAVE_DIR = '../results'
os.makedirs(SAVE_DIR, exist_ok=True)

def tokenize(text, tok):
    return tok(text, truncation=True, max_length=256, padding='max_length', return_tensors='pt')

print("Loading data...")
with open('../data/csn_java_100k.json', 'rb') as f:
    all_pairs = json.loads(f.read().decode('utf-8'))

random.seed(42)
random.shuffle(all_pairs)
pairs_100k = all_pairs[:100000]

with open('../data/benchmark_sample_50.json', 'rb') as f:
    bench = json.loads(f.read().decode('utf-8'))

y = np.array([1 if c['label'] == 'NO_DRIFT' else 0 for c in bench])

class PairDataset(Dataset):
    def __init__(self, pairs, tok): 
        self.pairs = pairs
        self.tok = tok
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        p = self.pairs[idx]
        c = tokenize(p['code'], self.tok)
        m = tokenize(p['comment'], self.tok)
        return {'code_ids': c['input_ids'].squeeze(), 'code_mask': c['attention_mask'].squeeze(),
                'comment_ids': m['input_ids'].squeeze(), 'comment_mask': m['attention_mask'].squeeze()}

def train_infonce(seed, model_name, save_dir):
    print(f"\n{'='*40}\nProcessing {model_name} with Seed {seed}\n{'='*40}")
    result_file = os.path.join(SAVE_DIR, f'{save_dir.replace("../", "")}_seed_{seed}_results.json')
    
    if os.path.exists(result_file):
        print(f"Seed {seed} already exists; skipping directly!")
        with open(result_file, 'r') as f:
            return json.load(f)['auc']

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    opt = torch.optim.AdamW(mdl.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    BATCH_SIZE = 8
    ACCUM_STEPS = 2
    loader = DataLoader(PairDataset(pairs_100k, tok), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    mdl.train()
    for epoch in range(2):
        opt.zero_grad()
        for step, batch in enumerate(tqdm(loader, desc=f"Seed {seed} Epoch {epoch+1}/2")):
            ci, cm = batch['code_ids'].to(device), batch['code_mask'].to(device)
            mi, mm = batch['comment_ids'].to(device), batch['comment_mask'].to(device)
            
            with torch.amp.autocast('cuda'):
                cv = mdl(input_ids=ci, attention_mask=cm).last_hidden_state[:, 0, :]
                mv = mdl(input_ids=mi, attention_mask=mm).last_hidden_state[:, 0, :]
                cv = F.normalize(cv, dim=1); mv = F.normalize(mv, dim=1)
                
                sim = torch.matmul(cv, mv.T) * 20 
                labels = torch.arange(sim.size(0)).to(device)
                loss = F.cross_entropy(sim, labels) / ACCUM_STEPS
            
            scaler.scale(loss).backward()
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()

    mdl.eval()
    def emb(text):
        inp = tokenize(text, tok)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out = mdl(**inp)
        return out.last_hidden_state[:, 0, :].squeeze().float().cpu().numpy()

    print(f"Evaluating...")
    scores = []
    for c in tqdm(bench, desc="Evaluating"):
        v1 = emb(c['current_code'])
        v2 = emb(c['historical_comment'])
        score = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
        scores.append(score)
        
    auc = roc_auc_score(y, scores)
    
    with open(result_file, 'w') as f:
        json.dump({'seed': seed, 'auc': auc, 'scores': scores, 'labels': y.tolist()}, f)
    print(f"Seed {seed} finished successfully! AUC = {auc:.4f}")
    
    # Save the fine-tuned model for seed 42
    if seed == 42:
        mdl.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")
    
    del mdl, opt, scaler, loader
    gc.collect(); torch.cuda.empty_cache()
    return auc

if __name__ == "__main__":
    seed_list = [42, 123, 2024]
    
    # 1. Train and save CodeBERT
    print("\n>>> Training CodeBERT...")
    for s in seed_list:
        train_infonce(s, "microsoft/codebert-base", "../finetuned_model")
        
    # 2. Train and save GraphCodeBERT (Required by Scripts 3 & 4)
    print("\n>>> Training GraphCodeBERT...")
    for s in seed_list:
        train_infonce(s, "microsoft/graphcodebert-base", "../graphcodebert_finetuned")
        
    print("\nAll seeds and both models finished!")