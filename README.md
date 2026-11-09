Anonymous CodeDrift: Label-Free Contrastive Alignment for Temporal Code-Comment Drift Detection

This repository contains the source code, data samples, and experimental results for our submission to The Computer Journal.

Paper Title: An Exploratory Empirical Study of Label-Free Contrastive Alignment for Temporal Code-Comment Drift Detection

**Model Weights:** The fine-tuned model weights are excluded from the main branch due to GitHub's file size limit. You can download them directly from the **[Releases](https://github.com/yzxppzx-ship-it/anonymous-codedrift/releases/tag/v1.0.0)** page.

**Reviewer's Tip (Instant Reproduction):** To save your time and computational resources, we have pre-computed all the heavy `.npy` feature files and prediction scores. **You do NOT need a GPU or network connection to verify our statistical tests.** You can directly run `3_eval_zeroshot.py` and `5_noise_simulation.py` in a few seconds to see the exact numbers reported in the paper.

Repository Structure

```text
Anonymous_CodeDrift/
├── README.md                     <-- This file
├── requirements.txt              <-- Python dependencies
│
├── data/
│   ├── annotation_guideline.pdf  <-- The guideline used for manual labeling
│   ├── benchmark_format.md       <-- Format description of the JSON data
│   ├── benchmark_sample_50.json  <-- A 50-pair sample of the temporal benchmark
│   ├── balanced_200.json         <-- The balanced 200-pair benchmark for Table 2
│   ├── labels_380.json           <-- Manual labels for the full replication pool
│   ├── temporal_benchmark_candidates.json <-- Full mined candidate pool
│   └── README.md                 <-- Data directory description
│
├── finetuned_model/              <-- Fine-tuned CodeBERT weights (Available in Releases)
├── graphcodebert_finetuned/      <-- Fine-tuned GraphCodeBERT weights (Available in Releases)
│
├── results/
│   ├── README.md                 <-- Description of the raw result files
│   ├── table2_auc_scores.json    <-- Data for Table 2
│   ├── table3_significance_tests.json <-- Data for Table 3
│   ├── label_budget_means_stds.json   <-- Data for Figure 4 (Budget curves)
│   ├── S1_pilots.json            <-- Data for Supplementary Material S1
│   │
│   ├── gcb_y.npy                 <-- Feature files for Figure 4 and Table 4
│   ├── gcb_X_code.npy
│   ├── gcb_X_comment.npy
│   │
│   ├── s_gcb_380_scores.npy      <-- Pre-computed prediction scores for exact statistical tests
│   ├── s_nightowl_380_scores.npy
│   ├── s_sbert_380_scores.npy
│   └── s_unft_gcb_380_scores.npy
│
└── scripts/
    ├── 1_extract_candidates.py   <-- Script to mine candidates from Git history
    ├── 2_finetune_infonce.py     <-- Script for InfoNCE fine-tuning
    ├── 3_eval_zeroshot.py        <-- Script for Zero-shot and statistical tests
    ├── 4_label_budget_curve.py   <-- Script for Label Efficiency analysis
    └── 5_noise_simulation.py     <-- Script for Label Noise simulation
```
Environment Setup

We recommend using Python 3.8+.
To install the required packages, run:
`pip install -r requirements.txt`

Reproduction Guide for Reviewers

Important: Please make sure to change into the scripts directory before running the Python codes, as the file paths are hardcoded as ../data/ and ../results/.

3.1 Fast Reproduction (No GPU Required)
Since all feature files and prediction scores (.npy) are already provided in the results/ folder, you can verify our claims instantly:

Reproduce Table 2 and Table 3 (Statistical Tests):
python 3_eval_zeroshot.py
(This will instantly load the .npy scores and output the exact DeLong/Bootstrap p-values)

Reproduce Figure 4 (Label Budget Curves):
python 4_label_budget_curve.py
(This will use the pre-computed features to plot the budget curve)

Reproduce Table 4 (Label Noise Simulation):
python 5_noise_simulation.py
(This will run the noise simulation instantly)

3.2 Full Retraining from Scratch (Optional)
If you wish to retrain the models from scratch and regenerate all files:

Place your sampled csn_java_100k.json in the data/ folder.

Run python 1_extract_candidates.py.

Run python 2_finetune_infonce.py.

Run python 4_label_budget_curve.py to extract features, then python 3_eval_zeroshot.py to generate the prediction scores.

Data Availability

We provide a strictly balanced 50-pair sample in data/benchmark_sample_50.json for format verification. The full 380-pair annotated benchmark will be publicly released upon paper acceptance to protect the integrity of the test set.
----------
This repository has been anonymized for peer review.
