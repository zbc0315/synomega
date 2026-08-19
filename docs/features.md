# 主要功能

SynOmega 对外提供五项能力：**单步逆合成、单步正向预测、多步路径规划、可合成性评分、反应合理性评分**。本页给出安装、命令行与 Python API 的快速上手；每项能力的模型/算法细节见[研究报告](research/index.md)。

## 安装

```bash
pip install synomega           # 核心：rdkit + numpy（纯模板后端可直接用）
pip install "synomega[gnn]"    # 追加 D-MPNN 神经单步后端（torch）
```

神经后端是**可选 extra**：不装 torch 也能用纯模板规则后端跑通全流程；需要神经模板分类精度时再装 `[gnn]`。默认模型权重与 ZINC 可购砌块集在**首次调用时按需下载**到本地缓存（`~/.cache/synomega`，可用环境变量 `SYNOMEGA_CACHE` 覆盖），不随 wheel 分发。

## 能力清单

| 能力 | 命令行 | Python 入口 |
|---|---|---|
| 单步逆合成 | —（经 `plan`/`score` 调用） | `TemplateGNN.default().predict(smiles, top_k)` |
| 单步正向预测 | `synomega forward` | `ForwardTemplateGNN.default().predict(reactants, top_k)` |
| 多步路径规划 | `synomega plan` | `synomega.load_default_planner().plan(target)` |
| 可合成性评分 | `synomega score` | `synomega.load_default_scorer().score(smiles)` |
| 反应合理性 | —（可挂进 planner） | `PlausibilityScorer.default().score_reactions([...])` |
| 砌块建库 | `synomega build-stock` | `InMemoryStock.from_file(...)` |

## 命令行

```bash
# 单步正向预测：反应物（多分子用 . 连接）-> 排序产物
synomega forward "CC(=O)O.NC" --top-k 5

# 多步路径规划：给一个目标找路径（默认 Retro*，可 --simplify 用简化约束模型）
synomega plan --target "O=C(Nc1ccccc1)c1ccccc1" --max-steps 5 --simplify

# 可合成性评分：对一个 SMILES 列表批量打分（每行一个）
# 默认用简化约束模型（评分推荐），加 --original 换成原始模型
synomega score --targets targets.smi --out scores.jsonl

# 首次使用可显式预下载默认权重与砌块集
synomega download

# 把一个采购目录（SMILES 列）转成 InChIKey 砌块库，供 --stock 使用
synomega build-stock --catalogue emolecules.smi --out stock_keys.txt
```

`plan` / `score` 的常用旋钮：`--algorithm {retrostar,mcts,bfs}`、`--expansion-width`（每节点取 top-k 候选）、`--max-steps`（路径深度上限）、`--time-limit`、`--max-expansions`、`--device`、`--cache`（SQLite 扩展缓存路径）。

## Python API

开箱即用：两个便捷入口在首次调用时自动下载默认模型与砌块集到 `~/.cache/synomega`。

```python
import synomega

# 多步规划：给目标找路径（默认原始模型 + retrostar）
planner = synomega.load_default_planner()
result = planner.plan("CC(=O)Nc1ccccc1O")
print(result.best_route.describe())

# 可合成性评分：默认用简化约束模型 + k=10（评分推荐配置，见研究报告）
scorer = synomega.load_default_scorer()
report = scorer.score("CC(=O)Nc1ccccc1O")
print(report.as_dict())        # solved / bb_coverage / min_steps / score ...
```

需要直接用单步后端时（例如只取一步逆合成候选）：

```python
from synomega.singlestep import TemplateGNN

# 产物 -> 排序的反应物候选
for p in TemplateGNN.default().predict("CC(=O)Nc1ccccc1O", top_k=5):
    print(p.score, p.smiles)   # p.reactants 是已排序的 canonical tuple
```

底层组装（自定义模型 / 砌块 / 算法）时用 `Planner(model, stock, algorithm=...)` +
`SynthesizabilityScorer(planner)`，`load_default_*` 即是它们的默认封装。

正向预测（独立于逆合成接口，规划器不会误用它）：

```python
from synomega.forward import ForwardTemplateGNN

fwd = ForwardTemplateGNN.default()
for pred in fwd.predict("CC(=O)O.NC", top_k=5):
    print(pred.score, pred.product)   # meta["n_templates"] = 能产出该产物的模板数
```

## 设计取舍（一眼看懂）

- **窄接口**：单步后端只实现 `predict(smiles, top_k) -> [Prediction]`；搜索、评分、合理性过滤都建立在这个接口之上，可无痛替换后端。
- **InChIKey 作分子身份**：AND-OR 图去重、库存判定、缓存命中三处口径一致，图因此是 DAG 而非树。
- **可选依赖分层**：核心只依赖 rdkit+numpy；神经后端、SQLite 超大砌块库都是按需启用。
- **诚实的默认值**：单步反应合理性过滤经评测为**净负收益**，因此在软件里**默认关闭**、保留可显式开启（见[反应合理性评分](research/plausibility.md)）。
