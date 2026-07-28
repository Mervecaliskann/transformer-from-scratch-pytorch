# Transformer Architecture from Scratch (PyTorch)

I built the Transformer architecture step by step in PyTorch to understand what's actually going on inside the models I use every day. Most of the pieces are implemented by hand; for the core attention I used PyTorch's `nn.MultiheadAttention` and wrapped it with the residual and layer-norm structure around it.

Based on *Attention Is All You Need* (Vaswani et al., 2017).

## What's in here

| Part | Notes |
|---|---|
| Positional Encoding | Sin/cos, implemented from scratch |
| Multi-Head Attention | Uses PyTorch's `nn.MultiheadAttention`, wrapped with residual + LayerNorm |
| Encoder | N × (attention + feed-forward + LayerNorm) |
| Decoder | N × (masked self-attention + cross-attention + feed-forward) |
| Causal Mask | Upper-triangular mask so the decoder can't see future tokens |
| Full Transformer | Encoder–decoder with an output projection to vocab |
| LR Scheduler | Warmup + inverse-sqrt decay, as in the paper |
| Mixed Precision | FP16 training with GradScaler |
| Beam Search | Tracks top-k hypotheses during decoding |
| SST-2 training | Sentiment classification on ~67K examples (T4 GPU) |

## A quick sanity check

Shapes line up through the whole stack:

| Component | Output shape |
|---|---|
| Positional Encoding | [2, 10, 512] |
| Multi-Head Attention | [2, 10, 512] |
| Encoder | [2, 10, 512] |
| Decoder Layer | [2, 8, 512] |
| Full Transformer | [2, 8, 5000] |
| Beam Search | 51 tokens generated |
| SST-2 (1 epoch demo) | loss 0.59, acc 0.68 |

The causal mask hides future tokens so each position can only attend to earlier ones:

```
[0, 1, 1, 1, 1, 1, 1, 1]
[0, 0, 1, 1, 1, 1, 1, 1]
[0, 0, 0, 1, 1, 1, 1, 1]
...
0 = visible, 1 = hidden
```

## Why I built it

I wanted to stop treating attention as a black box. Coding the encoder/decoder flow, the masking, the scheduler, and beam search by hand made the paper click in a way that just calling a library never did.

## Stack

PyTorch · HuggingFace Datasets · GLUE SST-2 · BERT tokenizer

## Files

`Transformer_Mimarisi_Sifirdan.ipynb` — the full notebook, with Turkish/English comments.
