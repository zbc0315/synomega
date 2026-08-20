# 单步正向反应预测

**做什么**：给一组反应物（一个或多个分子，多分子用 `.` 连成一个 SMILES），预测它们最可能生成的
**产物**并按可信度排序。它是逆合成单步模型的镜像——复用同一套 64,366 个反应模板，把命中的逆合成
模板反转后用 RDKit 正向施加到反应物上。模型原理与评测见
[研究报告 · 单步正向反应预测](../research/forward.md)；安装见[安装与总览](index.md)。

**输入 / 输出**：输入 = 反应物 SMILES（如 `CC(=O)O.NCc1ccccc1` 表示乙酸 + 苄胺）；输出 = 一个排序
列表，每个候选带产物 SMILES、分数（正向预测概率，越大越可信）、命中的模板号。

## 命令行

```bash
synomega forward "CC(=O)O.NCc1ccccc1" --top-k 5
```

真实示例输出（乙酸 + 苄胺，酰胺偶联占绝对主导）：

```
 1. CC(=O)NCc1ccccc1	score=0.8434	template=0
 2. CC(=O)OCc1ccccc1	score=0.0176	template=3418
 3. CCNCc1ccccc1	score=0.0158	template=3376
 4. NCC1CCCCC1	score=0.0027	template=918
 5. CCO	score=0.0025	template=26
```

top1 `CC(=O)NCc1ccccc1`（N-苄基乙酰胺）就是羧酸 + 胺缩合的酰胺，分数 0.84 远高于其余候选。

## Python

```python
from synomega.forward import ForwardTemplateGNN

fwd = ForwardTemplateGNN.default()             # 首次调用自动下载正向模型（约 149 MiB）
for pred in fwd.predict("CC(=O)O.NCc1ccccc1", top_k=5):
    print(pred.score, pred.product, pred.template_id)
    print(pred.meta["n_templates"])            # 有多少个模板能产出该产物（支持度）
```

`predict` 返回 `ForwardPrediction` 列表，字段：`product`（canonical SMILES）、`score`（0–1 正向
概率）、`template_id`、`meta["n_templates"]`。批量预测用 `fwd.predict_batch([...])`（GPU 上更快）。

## 参数

| 参数（CLI / Python） | 默认 | 含义 |
|---|---|---|
| `--top-k` / `top_k` | 10 | 返回多少个排序产物 |
| `--topk-templates` / `topk_templates` | 10 | 每次取模型 softmax 的前几个模板去正向施加；调大可能挖出更多罕见产物，但更慢 |
| `--model` / `from_pretrained(run_dir)` | 自动下载 | 用自己的正向模型 run 目录 |
| `--device` / `device` | 自动 | `cpu` 或 `cuda:0` |

## 读输出 / 注意

- 分数是模板的 softmax 概率：一个产物继承"能产出它的所有模板里的最大概率"；概率相同再按"有多少
  个模板支持它"（`meta["n_templates"]`）排序。
- 产物已做 sanitize、剔除自由基/卡宾伪影、取最大有机片段、canonical 化。
- 精度上限受模板法限制（验证集产物 top-1 ≈ 0.64）——把结果当**候选**看，不是保证。
