# microGPT

The most atomic way to train and run inference for a GPT in pure, dependency-free Python.

This is a minimal implementation of a GPT (Generative Pre-trained Transformer) model based on Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) project. The entire algorithm is contained in a single Python file with zero external dependencies.

## Features

- **Pure Python**: No NumPy, PyTorch, or TensorFlow required
- **Self-contained**: Includes automatic differentiation (autograd) implementation
- **Minimal**: Complete GPT training and inference in < 300 lines of code
- **Educational**: Clear implementation of transformer architecture components

## How It Works

The implementation includes:

1. **Tokenizer**: Character-level tokenization
2. **Autograd**: Custom automatic differentiation engine (`Value` class)
3. **Transformer Architecture**:
   - Multi-head attention with causal masking
   - RMSNorm instead of LayerNorm
   - ReLU activation instead of GeLU
   - Residual connections
4. **Optimizer**: Adam with learning rate decay
5. **Training**: Trains on names dataset (downloads automatically)

## Usage

Simply run the script:

```bash
python microgpt.py
```

The script will:
1. Download the names dataset (if not present)
2. Train a small GPT model on the names
3. Generate new names after training

## Model Configuration

Default hyperparameters (easily modifiable in the code):

- `n_layer`: 1 (transformer layer)
- `n_embd`: 16 (embedding dimension)
- `n_head`: 4 (attention heads)
- `block_size`: 16 (context length)
- `num_steps`: 1000 (training steps)

## Example Output

```
num docs: 32033
vocab size: 27
num params: 10611
step 1000 / 1000 | loss 1.2345
--- inference (new, hallucinated names) ---
sample  1: katherine
sample  2: elizabeth
sample  3: christopher
...
```

## Educational Value

This implementation is designed for learning purposes. It demonstrates:

- How attention mechanisms work
- How backpropagation flows through transformer layers
- How to implement autograd from scratch
- The complete training loop for language models

## Credits

Based on Andrej Karpathy's work. Original gist by [xiyoulaoyuanjia](https://gist.github.com/xiyoulaoyuanjia/6b26aee71043cb7ebe75c78044ece611).

## License

MIT