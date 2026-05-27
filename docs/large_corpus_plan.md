# Large Corpus Plan

Recorded/updated at: `2026-05-27 18:29:44 +09:00`

The goal is to train the project GPT on about 1 million character-level tokens from Korean wiki-style corpora.

## Sources

### Korean Wikipedia

Use the official Wikimedia dump:

```text
https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
```

The `latest` dump page showed `kowiki-latest-pages-articles.xml.bz2` dated `01-May-2026`, size about `1.28 GB` compressed.

### Namuwiki

Use a local plain-text Namuwiki corpus if available.

Recommended public reference:

```text
https://github.com/lovit/namuwikitext
```

This corpus reports:

- train: 31,235,096 lines, 500,104 docs, 4.6 GB
- dev: 153,605 lines, 2,525 docs, 23 MB
- test: 160,233 lines, 2,527 docs, 24 MB

Namuwiki-derived datasets are licensed as `CC BY-NC-SA 2.0 KR`, so they should be used for non-commercial class/research work with attribution and share-alike awareness.

## Disk Constraint

Current local disk check showed about `23 GB` free on the C drive. This is enough for a controlled 100M-character corpus, but not comfortable for repeatedly storing every compressed dump, full extracted dump, checkpoints, and intermediate files.

Therefore the project should build a capped corpus file instead of fully expanding every source.

## Command

Build a 1M-character corpus:

```bash
.\.venv\Scripts\python.exe scripts\build_wiki_corpus.py --target-chars 1000000 --output data\raw\wiki_ko_namu_1m.txt
```

If a local Namuwiki text file is available:

```bash
.\.venv\Scripts\python.exe scripts\build_wiki_corpus.py --namuwiki-text path\to\namuwiki.txt --target-chars 1000000 --output data\raw\wiki_ko_namu_1m.txt
```

Quick dry run:

```bash
.\.venv\Scripts\python.exe scripts\build_wiki_corpus.py --dry-run-chars 100000 --output data\raw\wiki_dry_run.txt
```

Prepare token data:

```bash
.\.venv\Scripts\python.exe scripts\prepare_data.py --input data\raw\wiki_ko_namu_1m.txt
```

Train:

```bash
.\.venv\Scripts\python.exe scripts\train_gpt.py --block-size 128 --emb-dim 128 --num-heads 4 --num-layers 4 --batch-size 16 --max-steps 3000 --eval-interval 300
```

## Note

The current model uses a character-level tokenizer. In this project, "1M data" means about 1 million character tokens. A future BPE tokenizer would make this closer to GPT-2-style tokenization.

## Change Log

Recorded/updated at: `2026-05-27 18:41:32 +09:00`

The original target of 100M character-level tokens was reduced to 1M character-level tokens to make the run practical on the current local machine.

Recorded/updated at: `2026-05-27 18:45:30 +09:00`

The first 1M-token run used Korean Wikipedia text only. Namuwiki is still supported by the script through `--namuwiki-text`, but a local Namuwiki plain-text file was not available during this run.
