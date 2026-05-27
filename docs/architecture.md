# Architecture

The final model is a small GPT-style decoder-only Transformer.

Input text is tokenized at the character level. For each position, the model adds a token embedding and a learned positional embedding. The sequence then passes through repeated Transformer blocks:

- masked multi-head self-attention
- residual connection
- layer normalization
- feedforward network
- residual connection
- layer normalization

The causal mask prevents each token from attending to future tokens, so the model learns the autoregressive objective:

```text
predict token[t + 1] from tokens[0:t]
```

This follows the same educational path as the course notebooks: bigram, MLP, sequence model, masked attention, and finally TinyGPT.

