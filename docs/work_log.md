# Work Log

This document records the project setup process step by step so that anyone can understand how the repository was built and verified.

Log rule:

- Every new entry must include the time when it was added.
- Existing entries were reviewed and timestamped on `2026-05-27 18:23:09 +09:00`.

## 1. Project Goal

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

The goal of this repository is to build a small but working GPT-style language model for the **Artificial Intelligence and Financial Engineering** course.

The project was designed around three requirements:

- preserve the six course notebooks as learning evidence
- organize the notebook ideas into reusable Python source files
- provide scripts that can train and sample from a TinyGPT model

## 2. Reference Material Reviewed

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

The project structure was planned using:

- six course notebooks from the local `Downloads` folder
- Andrej Karpathy's `nn-zero-to-hero`
- Andrej Karpathy's `makemore`

The notebooks showed a gradual progression:

1. Bigram language model on `names.txt`
2. MLP character model on `names.txt`
3. MLP model on Tiny Shakespeare
4. GPT-style next-token dataset and sequence loss
5. Single-head masked self-attention
6. TinyGPT with multi-head attention and Transformer blocks

## 3. Repository Structure Created

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

The initial repository scaffold was created under:

```text
C:\Users\workgroup\Desktop\GeePeeTee
```

Main folders:

```text
data/                 raw, processed, and finance text data
docs/                 architecture notes, training notes, assignment draft, work log
notebooks/            copied course notebooks
scripts/              runnable command-line scripts
src/working_gpt/      reusable Python package
tests/                small pytest checks
checkpoints/          local model checkpoints, ignored by git
outputs/              generated samples and logs, ignored by git
```

## 4. Core Files Added

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

The first draft added:

- `README.md`: project description and quick start commands
- `requirements.txt`: Python dependencies
- `pyproject.toml`: project metadata and pytest path configuration
- `.gitignore`: ignores virtual environments, caches, checkpoints, and generated outputs
- `src/working_gpt/tokenizer.py`: character-level tokenizer
- `src/working_gpt/data.py`: next-token dataset and dataloader helper
- `src/working_gpt/models/attention.py`: masked attention head and multi-head attention
- `src/working_gpt/models/blocks.py`: feedforward and Transformer block
- `src/working_gpt/models/gpt.py`: final TinyGPT model
- `src/working_gpt/train.py`: training loop and checkpoint saving
- `src/working_gpt/generate.py`: checkpoint loading and text generation
- `scripts/prepare_data.py`: text corpus to train/validation token tensors
- `scripts/train_gpt.py`: train TinyGPT from prepared data
- `scripts/sample.py`: generate text from a saved checkpoint

## 5. Course Notebooks Preserved

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

The six provided notebooks were copied into `notebooks/` with clearer names:

```text
notebooks/01_bigram_names.ipynb
notebooks/02_mlp_names.ipynb
notebooks/03_mlp_shakespeare.ipynb
notebooks/04_sequence_lm.ipynb
notebooks/05_single_head_attention.ipynb
notebooks/06_tiny_gpt.ipynb
```

This keeps the original class work visible while the reusable implementation lives in `src/`.

## 6. Initial Git Commit

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

The local repository was initialized with git and the first commit was created:

```text
86b8a4a Initial working GPT project scaffold
```

This commit contains the first full repository draft.

## 7. Python Environment Setup

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

The default system Python was:

```text
Python 3.14.4
```

Because PyTorch support for very new Python versions can be limited, a Python 3.9 virtual environment was created:

```bash
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Installed and verified:

```text
torch 2.8.0+cpu
pytest 8.4.2
```

## 8. Dependency Update

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

`pytest` was added to `requirements.txt` so that tests can be installed consistently:

```text
pytest>=8.0
```

## 9. Python 3.9 Compatibility Fix

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

The first test run found that Python 3.9 could not evaluate newer type-hint syntax such as:

```python
str | Path
torch.Tensor | None
```

The fix was to add:

```python
from __future__ import annotations
```

to files that used modern union type hints.

Affected files:

- `src/working_gpt/tokenizer.py`
- `src/working_gpt/data.py`
- `src/working_gpt/models/gpt.py`
- `src/working_gpt/generate.py`
- `src/working_gpt/train.py`

`pyproject.toml` was also updated to state:

```text
requires-python = ">=3.9"
```

## 10. Tests Run

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

The test suite was run with:

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
3 passed
```

The tests currently check:

- tokenizer encode/decode roundtrip
- next-token dataset input/target alignment
- TinyGPT forward pass shape and loss creation

## 11. Data Preparation Verified

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

The finance sample corpus was prepared with:

```bash
.\.venv\Scripts\python.exe scripts\prepare_data.py --input data\finance\sample_corpus.txt
```

Result:

```text
tokens: 247, vocab: 32, train: 222, val: 25
```

This created local processed data files under `data/processed/`. These generated files are ignored by git.

## 12. Tiny Training Run Verified

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

A very small training run was executed to confirm that the model can train end to end:

```bash
.\.venv\Scripts\python.exe scripts\train_gpt.py --block-size 8 --emb-dim 16 --num-heads 4 --num-layers 1 --batch-size 4 --max-steps 2 --eval-interval 1
```

Observed result:

```text
step 0: train loss 3.6173, val loss 3.5754
step 1: train loss 3.5725, val loss 3.5740
```

This confirmed that:

- data loading works
- TinyGPT forward pass works
- loss computation works
- optimization step works
- checkpoint saving works

## 13. Text Generation Verified

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

Generation was tested with:

```bash
.\.venv\Scripts\python.exe scripts\sample.py --prompt risk --max-new-tokens 20
```

Example output:

```text
riskfsvkrolohFge gemktxu
```

The output is not meaningful yet because the model was trained for only two steps on a tiny corpus. The important result is that sampling from a checkpoint works.

## 14. Second Git Commit

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

The dependency and compatibility fixes were committed:

```text
01fcbc2 Install test dependency and support Python 3.9
```

## 15. GitHub Upload

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

A GitHub repository was created under the account `anamiyaromo`.

The repository name was:

```text
-working-gpt
```

The local repository remote was set to:

```text
https://github.com/anamiyaromo/-working-gpt.git
```

The local `main` branch was pushed successfully:

```bash
git push -u origin main
```

Repository URL:

```text
https://github.com/anamiyaromo/-working-gpt
```

## 16. Current Status

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

As of this log, the repository has:

- a working TinyGPT implementation
- copied class notebooks
- training and sampling scripts
- a Python 3.9 virtual environment locally
- passing tests
- a successful short training run
- a successful checkpoint sampling run
- code pushed to GitHub

## 17. Suggested Next Steps

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

Recommended next improvements:

1. Rename the GitHub repository from `-working-gpt` to `working-gpt` if the leading hyphen was accidental.
2. Add a larger text corpus, such as Tiny Shakespeare or a finance-domain corpus.
3. Train for more steps and save example outputs under `outputs/samples/`.
4. Expand `docs/assignment_report.md` into the final submission report.
5. Add command examples for Windows PowerShell and Google Colab if the class expects reproducible demos.

## 18. GPT Execution Check

Recorded/updated at: `2026-05-27 18:23:09 +09:00`

Date and time:

```text
2026-05-27 18:18:44 +09:00
```

Purpose:

Check whether the current TinyGPT project runs normally from tests to generation.

### Step 1: Run Tests

Command:

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
3 passed in 2.30s
```

Conclusion:

The tokenizer, dataset, and TinyGPT forward-pass tests passed.

### Step 2: Prepare Data

Command:

```bash
.\.venv\Scripts\python.exe scripts\prepare_data.py --input data\finance\sample_corpus.txt
```

Result:

```text
tokens: 247, vocab: 32, train: 222, val: 25
```

Conclusion:

The finance sample corpus was successfully tokenized and split into training and validation data.

### Step 3: Train TinyGPT

Command:

```bash
.\.venv\Scripts\python.exe scripts\train_gpt.py --block-size 8 --emb-dim 16 --num-heads 4 --num-layers 1 --batch-size 4 --max-steps 20 --eval-interval 5
```

Result:

```text
step 0: train loss 3.6173, val loss 3.5754
step 5: train loss 3.5936, val loss 3.5669
step 10: train loss 3.5626, val loss 3.5600
step 15: train loss 3.5401, val loss 3.5535
step 19: train loss 3.5692, val loss 3.5478
```

Conclusion:

The model trained successfully and saved a checkpoint to `checkpoints/gpt_latest.pt`.

### Step 4: Generate Text

Command:

```bash
.\.venv\Scripts\python.exe scripts\sample.py --prompt risk --max-new-tokens 80 --temperature 0.9
```

Result:

```text
riskc
bb.v
.qtghn,
..
e,p,.,xlm .x.xuefgeqc,anf.
dhxx qiq
mr.xFffFoTn,k,sto.x.gTl.op
```

Conclusion:

Text generation completed successfully from the saved checkpoint. The generated text is not meaningful yet because the model was trained for only 20 steps on a very small sample corpus, but the full GPT pipeline is working.

## 19. Large Corpus Target Revised

Recorded/updated at: `2026-05-27 18:41:32 +09:00`

The requested large-corpus target was revised from about 100 million character-level tokens to about 1 million character-level tokens.

Reason:

- 1M tokens is much more practical for the current local PC.
- The C drive had about 23 GB free, which is enough for controlled experiments but tight for repeatedly storing full wiki dumps and intermediate files.
- A 1M-token run is large enough to produce more meaningful text than the tiny smoke-test corpus while still being feasible for class-project iteration.

Changes made:

- `scripts/build_wiki_corpus.py` default target changed to `1_000_000`.
- Default output changed to `data/raw/wiki_ko_namu_1m.txt`.
- `docs/large_corpus_plan.md` updated from the 100M plan to the 1M plan.

## 20. 1M Wikipedia Corpus Training Run

Recorded/updated at: `2026-05-27 18:45:30 +09:00`

The first 1M-token training run was executed.

Important note:

The current run used Korean Wikipedia text only. The corpus builder supports Namuwiki through the `--namuwiki-text` argument, but no local Namuwiki plain-text file was available during this run.

### Step 1: Stop Oversized Download

The previous 100M-target attempt had started downloading the full Korean Wikipedia dump. After the target was revised to 1M, the large download was stopped.

Partial file left locally:

```text
data/external/kowiki-latest-pages-articles.xml.bz2
```

This file is treated as external generated data and is not intended to be committed to GitHub.

### Step 2: Build 1M Corpus

Command:

```bash
.\.venv\Scripts\python.exe scripts\build_wiki_corpus.py --target-chars 1000000 --output data\raw\wiki_ko_namu_1m.txt --kowiki-dump data\external\kowiki-latest-pages-articles.xml.bz2
```

Result:

```text
processing source: kowiki
source done: kowiki, total chars: 1,000,000
corpus saved: data\raw\wiki_ko_namu_1m.txt
total chars: 1,000,000
```

### Step 3: Prepare Token Data

Command:

```bash
.\.venv\Scripts\python.exe scripts\prepare_data.py --input data\raw\wiki_ko_namu_1m.txt
```

Result:

```text
tokens: 1000308, vocab: 2911, train: 900277, val: 100031
```

### Step 4: Train TinyGPT

Command:

```bash
.\.venv\Scripts\python.exe scripts\train_gpt.py --block-size 64 --emb-dim 128 --num-heads 4 --num-layers 4 --batch-size 16 --max-steps 500 --eval-interval 100
```

Result:

```text
step 0: train loss 8.0413, val loss 8.0175
step 100: train loss 4.9953, val loss 5.0131
step 200: train loss 4.6287, val loss 4.7007
step 300: train loss 4.2093, val loss 4.3646
step 400: train loss 4.0829, val loss 4.1910
step 499: train loss 3.9658, val loss 4.0730
```

Conclusion:

Loss decreased clearly, so the model learned structure from the 1M-token corpus.

### Step 5: Generate Text

Command:

```bash
.\.venv\Scripts\python.exe scripts\sample.py --prompt "한국" --max-new-tokens 300 --temperature 0.8
```

Result:

```text
한국의 전이하는 선다음긴 카고에 수 되그 공언령이는 학은 사으로 과 했다. 이 그 국 단의 있다른 물며 u 한다.
 후가역시 목의 가 지 이 중화킬의 접 성 있다. 등문학 軒른 RA 유자에 관역이 랙 이 원이靈는, 관 같가다른 곱셈러되었 거는 투던 양게 서 그 한다 가 있다가
 정년로 배 19001일
 않시사에 외의 기 몫 대 필atac \m) \r 참
 \tetepa \= \mm iterac \fto {\mri년럽다. \f{\t{n\ce = \bse\m } \d{\inh& \fri 노& \m1t \sgi기 \cglerhrrac{} x
```

Conclusion:

The generated text is now more sentence-like than the tiny smoke-test output, but it is still noisy. Better Korean output will require more training steps, cleaner text extraction, and ideally a stronger tokenizer such as BPE.

### Step 6: Output Encoding Fix

During the first generation attempt, Windows `cp949` console encoding could not print some generated Unicode characters.

Fix:

- `scripts/sample.py` now reconfigures stdout to UTF-8 when supported.

### Step 7: Git Ignore Update

Large/generated corpus files were added to `.gitignore`:

```text
data/external/*
data/raw/wiki_*.txt
```

This prevents downloaded dumps and generated wiki corpora from being accidentally committed.
