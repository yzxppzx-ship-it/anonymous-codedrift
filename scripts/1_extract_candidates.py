import os
import re
import json
import subprocess
import sys
from git import Repo
from tqdm import tqdm

TARGET_DIR = '../data/new_projects'
REPOS = {
    'gson': 'https://github.com/google/gson.git',
    'jsoup': 'https://github.com/jhy/jsoup.git'
}

CODE_PATTERN = re.compile(r'(/\*\*[\s\S]*?\*/)\s*(.*?)(?=\n\s*/\*\*|\Z)', re.DOTALL)

def clone_repos():
    os.makedirs(TARGET_DIR, exist_ok=True)
    for name, url in REPOS.items():
        dest = os.path.join(TARGET_DIR, name)
        if not os.path.exists(dest):
            print(f"Cloning {name} ...")
            try:
                subprocess.run(['git', 'clone', url, dest], check=True)
                print(f"{name} cloned successfully!")
            except subprocess.CalledProcessError:
                print(f"Failed to clone {name}; please check the network.")
                sys.exit(1)
        else:
            print(f"{name} already exists; skipping clone.")

def extract_candidates(repo_name, max_commits=3000):
    repo_path = os.path.join(TARGET_DIR, repo_name)
    print(f"\nExtracting history from {repo_name} (checking up to {max_commits} commits)...")
    
    repo = Repo(repo_path)
    candidates = []
    
    # Limit to max_commits to prevent infinite loops
    commits = list(repo.iter_commits(max_count=max_commits))
    
    for commit in tqdm(commits, desc=f"Scanning {repo_name}"):
        if not commit.parents:
            continue
            
        parent = commit.parents[0]
        
        diffs = parent.diff(commit)
        for diff in diffs:
            if not diff.b_path or not diff.b_path.endswith('.java'):
                continue
                
            file_path = diff.b_path
            
            try:
                old_content = (parent.tree / file_path).data_stream.read().decode('utf-8', errors='ignore')
                new_content = (commit.tree / file_path).data_stream.read().decode('utf-8', errors='ignore')
            except Exception:
                continue
                
            old_blocks = CODE_PATTERN.findall(old_content)
            new_blocks = CODE_PATTERN.findall(new_content)
            
            old_dict = {doc.strip(): code for doc, code in old_blocks if len(doc.strip()) > 10}
            
            for doc, new_code in new_blocks:
                doc = doc.strip()
                if doc in old_dict:
                    old_code = old_dict[doc].strip()
                    new_code = new_code.strip()
                    
                    if old_code != new_code and len(new_code) > 20:
                        if '{' not in new_code: continue
                        
                        candidates.append({
                            'project': repo_name,
                            'commit': commit.hexsha[:8],
                            'historical_comment': doc,
                            'current_code': new_code,
                            'label': 'TBD'
                        })
                        
    return candidates

if __name__ == '__main__':
    clone_repos()
    
    all_candidates = []
    for name in REPOS.keys():
        # Increased max_commits to 3000 to ensure enough candidates
        cands = extract_candidates(name, max_commits=3000) 
        all_candidates.extend(cands)
    
    seen = set()
    unique_candidates = []
    for c in all_candidates:
        key = (c['historical_comment'], c['current_code'])
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    output_file = '../data/temporal_benchmark_candidates.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_candidates, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*50)
    print(f"Extraction complete! Found {len(unique_candidates)} temporal-mismatch candidates.")
    print(f"Data saved to: {output_file}")