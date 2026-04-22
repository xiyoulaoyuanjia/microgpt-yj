"""
microGPT（教学版，纯 Python 实现）
=================================

这个文件展示了一个“最小可运行”的 GPT 训练与推理全过程，特点是：
1) 不依赖 NumPy / PyTorch 等深度学习框架；
2) 自动微分（反向传播）自己实现；
3) Transformer 结构核心要素齐全（Embedding、注意力、MLP、残差、归一化）；
4) 代码短小，适合从底层理解语言模型。

建议阅读顺序（小白友好）：
1. 数据与分词（docs, vocab, BOS）
2. 自动微分 Value 类（最关键）
3. 模型参数初始化（state_dict）
4. 前向计算 gpt()
5. 训练循环（loss.backward + Adam）
6. 推理采样（temperature + random.choices）
"""

import os       # 文件是否存在检查
import math     # 数学函数：log/exp
import random   # 随机数：初始化参数、打乱数据、采样输出
random.seed(42) # 固定随机种子，保证每次运行结果更可复现（便于学习和调试）

# -----------------------------
# 1) 准备数据 docs: list[str]
# -----------------------------
# 这里使用姓名数据集，每一行是一个名字。
# 如果本地没有 input.txt，会自动从网络下载。
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# -----------------------------------------
# 2) 字符级分词器（Tokenizer）与词表（vocab）
# -----------------------------------------
# 思路：把所有出现过的字符收集出来，每个字符分配一个整数 id。
# 例如：'a' -> 0, 'b' -> 1, ...（具体顺序由 sorted 决定）
uchars = sorted(set(''.join(docs))) # 数据集中出现过的全部唯一字符
BOS = len(uchars) # 特殊起始/终止标记（Beginning of Sequence）的 token id
vocab_size = len(uchars) + 1 # 词表总大小 = 普通字符 + 1 个 BOS
print(f"vocab size: {vocab_size}")

# ----------------------------------------------------
# 3) 自动微分引擎（Autograd）核心：Value 标量计算图
# ----------------------------------------------------
# 每个 Value 节点保存：
# - data: 前向计算得到的标量值
# - grad: 损失函数对该节点的梯度 dL/dx
# - _children: 这个节点由哪些子节点计算而来
# - _local_grads: 当前节点对每个子节点的局部导数
# 反向传播时用链式法则：child.grad += local_grad * v.grad
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # 减少对象内存开销

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 前向：这个节点的数值
        self.grad = 0                   # 反向：损失对该节点的梯度
        self._children = children       # 计算图中的“输入节点”
        self._local_grads = local_grads # 当前节点对每个输入节点的局部导数

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        # 第一步：拓扑排序（确保一个节点的梯度传播发生在其“后继”节点之后）
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # 损失对自身的梯度恒为 1
        self.grad = 1
        # 第二步：按拓扑逆序做链式法则累加梯度
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ----------------------------------------------------
# 4) 模型超参数 + 参数初始化（state_dict）
# ----------------------------------------------------
# 注意：这是“极小模型”，参数量很小，目的不是追求效果，而是便于理解流程。
n_layer = 1     # Transformer 层数（深度）
n_embd = 16     # 嵌入维度（宽度）
block_size = 16 # 最大上下文长度（最长看 16 个位置）
n_head = 4      # 注意力头数
head_dim = n_embd // n_head # 每个头的维度（要求可整除）
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
print(f"num params: {len(params)}")

# ----------------------------------------------------
# 5) 定义前向计算：token -> logits
# ----------------------------------------------------
# 这里的 logits 是“下一个 token 各类别的打分”（还未归一化成概率）。
# 结构近似 GPT-2，但做了简化：
# - LayerNorm 改为 RMSNorm
# - 不使用 bias
# - GeLU 改为 ReLU
def linear(x, w):
    # 线性层 y = W x（这里按行实现）
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    # softmax 做了“减最大值”以提升数值稳定性，避免 exp 溢出
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    # RMSNorm: x / sqrt(mean(x^2) + eps)
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    # token embedding + position embedding
    tok_emb = state_dict['wte'][token_id] # 当前 token 的向量表示
    pos_emb = state_dict['wpe'][pos_id] # 当前位置的向量表示
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # 融合词义与位置信息
    x = rmsnorm(x) # 输入归一化

    for li in range(n_layer):
        # 1) 多头注意力块（Attention Block）
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            # 取出第 h 个头对应的切片
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            # 当前时刻 pos_id 只能看“历史与当前”位置：由 keys/values 逐步 append 自然形成因果性
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            # 加权求和得到该头输出
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        # 多头拼接后过输出投影，再加残差
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) 前馈网络块（MLP Block）
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# -----------------------------------------
# 6) Adam 优化器状态
# -----------------------------------------
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # 一阶动量（梯度均值）
v = [0.0] * len(params) # 二阶动量（梯度平方均值）

# -----------------------------------------
# 7) 训练循环
# -----------------------------------------
num_steps = 1000 # 训练总步数
for step in range(num_steps):

    # 取一个样本名字，并在首尾加 BOS：
    # [BOS] + 字符序列 + [BOS]
    # 这样模型就学会“从开始到结束”的生成过程
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # 前向：逐位置预测下一个字符，累积每个位置的负对数似然损失
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # 序列平均损失

    # 反向传播：自动微分把梯度传到所有参数
    loss.backward()

    # Adam 参数更新（带线性学习率衰减）
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# -----------------------------------------
# 8) 推理（生成）
# -----------------------------------------
# temperature 越低越保守，越高越发散（更“有创意”但也更容易胡说）
temperature = 0.5 # 取值通常在 (0, 1]，教学场景可适当调大到 1.0 观察变化
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        # 按概率随机采样下一个 token（不是贪心取最大值）
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            # 采到 BOS 代表序列结束
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")