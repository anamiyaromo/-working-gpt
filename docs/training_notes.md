# Training Notes

## Baseline

Start with the smallest working model:

```bash
python scripts/prepare_data.py --input data/finance/sample_corpus.txt
python scripts/train_gpt.py --block-size 32 --emb-dim 64 --num-heads 4 --num-layers 2 --max-steps 200
python scripts/sample.py --prompt "Risk"
```

## Next Experiments

- Train on `tinyshakespeare.txt` to compare with the class notebooks.
- Add a larger finance-domain corpus.
- Increase `block_size`, `emb_dim`, and `num_layers` after the pipeline works.
- Record validation loss and sample outputs in `outputs/samples/`.

