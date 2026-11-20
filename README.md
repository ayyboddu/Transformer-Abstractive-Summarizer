# Abstractive Sequence Summarization with Custom Transformer Architecture

## Overview
Implemented a 4-layer encoder-decoder **Transformer** architecture (~10.47M parameters) from scratch in TensorFlow/Keras to perform long-document abstractive summarization, paired with custom WordPiece subword tokenization, multi-head attention visualization, and ROUGE evaluation.

## Architecture & Technical Implementation
- **Custom Transformer Components:** Built sinusoidal positional encodings, masked causal self-attention, global encoder self-attention, encoder-decoder cross-attention, and residual feed-forward layers (`d_model=128`, `num_heads=8`, `dff=512`).
- **Tokenization & Data Pipeline:** Trained an 8,000-token subword vocabulary (`BertTokenizer`) and constructed an optimized `tf.data` input pipeline (`shuffle`, `batch`, `prefetch`, teacher-forcing input/target alignment, and dynamic padding masks).
- **Training & Optimization:** Implemented custom masked sparse categorical cross-entropy loss and a warmup inverse-square-root Adam learning rate schedule (`warmup_steps=4000`) with model checkpointing.
- **Evaluation & Diagnostics:** Evaluated generation quality using **ROUGE-1** (0.52 precision) and **ROUGE-L** (0.26 precision) metrics and inspected multi-head cross-attention weight heatmaps to verify entity alignment between source transcripts and generated summaries.

## Tech Stack
`Python` · `TensorFlow / Keras` · `TensorFlow Text` · `NumPy` · `Pandas` · `Matplotlib` · `ROUGE-Score`
