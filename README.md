# microGPT

在纯 Python 且零外部依赖的前提下，以最小实现方式完成 GPT 的训练与推理。

本项目基于 Andrej Karpathy 的 [makemore](https://github.com/karpathy/makemore) 思路，实现了一个极简 GPT（Generative Pre-trained Transformer）。全部核心算法集中在一个 Python 文件中，便于阅读与学习。

## 特性

- **纯 Python**：不依赖 NumPy、PyTorch 或 TensorFlow
- **自包含**：内置自动微分（autograd）实现
- **极简实现**：不足 300 行代码即可完成 GPT 训练与推理
- **教学友好**：Transformer 关键模块实现清晰

## 工作原理

实现包含以下部分：

1. **分词器**：字符级 tokenization
2. **自动微分**：自定义自动求导引擎（`Value` 类）
3. **Transformer 结构**：
   - 带因果掩码的多头注意力
   - 使用 RMSNorm 替代 LayerNorm
   - 使用 ReLU 替代 GeLU
   - 残差连接
4. **优化器**：带学习率衰减的 Adam
5. **训练流程**：基于姓名数据集训练（首次会自动下载）

## 使用方法

直接运行脚本：

```bash
python microgpt.py
```

脚本会自动完成：
1. 下载姓名数据集（若本地不存在）
2. 训练一个小型 GPT 模型
3. 训练结束后生成新姓名

## 模型配置

默认超参数（可在代码中直接修改）：

- `n_layer`: 1（Transformer 层数）
- `n_embd`: 16（嵌入维度）
- `n_head`: 4（注意力头数）
- `block_size`: 16（上下文长度）
- `num_steps`: 1000（训练步数）

## 示例输出

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

## 学习价值

本实现用于教学与理解，重点展示：

- 注意力机制是如何工作的
- 反向传播如何穿过 Transformer 各层
- 自动微分如何从零实现
- 语言模型完整训练循环的最小实现

## 致谢

项目思路源于 Andrej Karpathy 的工作。原始 gist 由 [xiyoulaoyuanjia](https://gist.github.com/xiyoulaoyuanjia/6b26aee71043cb7ebe75c78044ece611) 提供。

## 许可证

MIT