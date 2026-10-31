# LogLM

<p>
  <img src="https://img.shields.io/badge/Python-3.7%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.8%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97_Transformers-4.x-FFD21E?logo=huggingface&logoColor=black" alt="Transformers">
</p>

**Do decoders really need full cross-attention?**

LogLM is an exploration of a simple hypothesis: an encoder-decoder model does *not* need the
decoder to attend to all *n* encoder outputs. A bottleneck of only **logarithmically many**
`[CLS]` vectors is enough for the decoder to reconstruct and reason over the input on the
tasks tested here.

## The idea

Standard cross-attention costs O(n) memory per decoding step and hands the decoder the entire
encoder sequence. LogLM instead prefixes the input with *k* `[CLS]`-style tokens, where

```
k ≈ log₂(n),   solved from  x + log(x) = max_len   (Newton's iteration, see LogTruncation)
```

and the decoder **only ever sees those k vectors**. During training the bottleneck size is
randomly varied (keep a random prefix of the CLS tokens with probability 0.2), so the model
must work with any bottleneck from 1 to *k* — an information-bottleneck regularizer that
forces the CLS states to actually compress the sequence.

```
input:      [CLS] [CLS] ... [CLS]  t1 t2 ... tn      (k ≈ log2(n) CLS tokens)
                 ↓ BERT encoder (12 layers)
bottleneck:      c1   c2  ...  ck                    (k vectors; rest replaced by [MASK] emb)
                 ↓ shallow decoder (2 layers)
output:     reconstructed t1 ... tn  (MLM loss)
```

## How it works

1. **Encode** — a BERT encoder reads `[CLS]×k + tokens`. Token ids 1–11 are reserved as the
   variable-length CLS block (`cls_type_ids`).
2. **Bottleneck** — CLS hidden states are projected (`cls_proj`, tanh); with
   `mask_cls_prob` a random prefix length is kept, the rest of the sequence is replaced by
   `[MASK]` embeddings (+ tiny Gaussian noise on the CLS states during training).
3. **Decode** — a 2-layer non-causal decoder (`NonCausalDecoder`, BERT encoder stack +
   MLM head) reconstructs every masked token **from the bottleneck alone** (`dec_mlm_loss`).
4. **Auxiliary objectives** — standard encoder MLM (`mlm_loss`) and sentence-order
   prediction on mean-pooled CLS states (`sop_loss`).

For downstream use, `LogLMForSequenceClassification` pools over the CLS states
(or all states with `pooling_all=True`).

## Repository layout

| Path | What it is |
|---|---|
| `model/model.py` | `LogLMForPreTraining` (MLM + SOP + bottleneck reconstruction), `LogLMForSequenceClassification`, both decoders |
| `model/tokenizer.py` | `LogTruncation` — Newton's iteration for the log-length bottleneck |
| `dataset.py` | `IterableDataset`; line format `labels\|masked_pos\|token_type_lengths\|cls_length\|SO_label` |
| `preprocess_data/` | Normalization, WordPiece tokenizer training, preprocessing pipeline |
| `trainer.py` / `main.py` | Training loop and entry point |
| `dist_launch_*.sbatch`, `local_launch*.sh`, `dist_run.sh` | SLURM / torchrun launch scripts (multi-node, multi-GPU) |

## Training

Single machine, multi-GPU:

```bash
./local_launch.sh        # edit MASTER_HOST / MASTER_PORT / nproc_per_node inside first
```

SLURM (multi-node, multi-GPU):

```bash
sbatch ./dist_launch_base.sbatch   # or dist_launch_tiny.sbatch
```

Manual two-node setup: keep code and data identical on both machines, pick a free port with
`python tools/find_unused_port.py`, set `MASTER_HOST`, `MASTER_PORT`, `SLURM_NNODES=2`,
`SLURM_NODEID=0/1` in `local_launch.sh` on each node, and run it on both.

Key knobs: `log_base` (default 2), `mask_cls_prob` (default 0.2), `cls_noise_std` (default 0.002),
`num_non_casual_decoder_layers` / `num_casual_decoder_layers` (default 2/2).

## Status

Research prototype (2022, SJTU). The implemented and exercised path is the non-causal
bottleneck reconstruction + SOP, plus classification fine-tuning. The causal decoder branch
exists in code but is stubbed out (`TODO` in `model.py`) — treat it as not ready.

## Citation

```bibtex
@misc{dai2022loglm,
  title={LogLM: BERT with a Logarithmic-Length [CLS] Bottleneck},
  author={Dai, Lingfeng},
  year={2022},
  howpublished={\url{https://github.com/Randool/LogLM}}
}
```

## Author

Randool (Lingfeng Dai) — randool@sjtu.edu.cn
