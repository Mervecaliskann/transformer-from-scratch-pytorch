# 🤖 Transformer Architecture from Scratch — PyTorch

Complete Transformer encoder-decoder implemented from scratch using pure PyTorch.
No high-level abstractions — just math and `torch.nn`.

## What's Implemented

| Component | Details |
|---|---|
| Positional Encoding | Sin/cos, shape [batch, seq, d_model] |
| Multi-Head Attention | Q/K/V projections, scaled dot-product |
| Encoder | N × (Self-Attention + FFN + LayerNorm) |
| Decoder | N × (Masked Attention + Cross-Attention + FFN) |
| Causal Mask | Upper triangular — no future token leakage |
| Full Transformer | Encoder-Decoder + output projection |
| LR Scheduler | Warmup + inverse sqrt decay (original paper) |
| Mixed Precision | FP16 + GradScaler |
| Beam Search | Top-k hypothesis tracking |
| SST-2 Fine-tuning | Sentiment classification, 67K examples, T4 GPU |

## Results

| Component | Output Shape | Status |
|---|---|---|
| Positional Encoding | [2, 10, 512] | ✅ |
| Multi-Head Attention | [2, 10, 512] | ✅ |
| Encoder | [2, 10, 512] | ✅ |
| Decoder Layer | [2, 8, 512] | ✅ |
| Full Transformer | [2, 8, 5000] | ✅ |
| Beam Search | 51 tokens generated | ✅ |
| SST-2 (1 epoch demo) | Loss: 0.59 \| Acc: 0.68 | ✅ |
| LR Peak Step | 3999 (warmup=4000) | ✅ |
| AMP Training Step | Loss: 0.44 \| Scale: 65536 | ✅ |

## Causal Mask (8×8)
```
[0, 1, 1, 1, 1, 1, 1, 1]
[0, 0, 1, 1, 1, 1, 1, 1]
[0, 0, 0, 1, 1, 1, 1, 1]
[0, 0, 0, 0, 1, 1, 1, 1]
...
```
0 = visible, 1 = masked (future tokens hidden)

## Stack

PyTorch · HuggingFace Datasets · GLUE SST-2 · BERT Tokenizer

## Reference

Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Files

| File | Description |
|---|---|
| `Transformer_Mimarisi_Sifirdan.ipynb` | Full notebook — TR/EN comments |
