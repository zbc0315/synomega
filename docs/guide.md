# 功能介绍

SynOmega 对外提供六项能力，本页按功能给出**使用方法**（命令行与 Python API）；每项的模型 /
算法原理见对应的[研究报告](research/index.md)章节，两边一一对应：

| 功能 | 使用方法（本页） | 原理（研究报告） |
|---|---|---|
| 单步正向反应预测 | [↓](#forward) | [research/forward](research/forward.md) |
| 多组分演化预测 | [↓](#evolution) | [research/evolution](research/evolution.md) |
| 单步逆向反应预测 | [↓](#retro) | [research/retro](research/retro.md) |
| 多步反应路径规划 | [↓](#planning) | [research/planning](research/planning.md) |
| 反应合理性评分 | [↓](#plausibility) | [research/plausibility](research/plausibility.md) |
| 可合成性评分 SynScore | [↓](#synscore) | [research/synscore](research/synscore.md) |

## 安装

```bash
pip install synomega           # 核心：rdkit + numpy（纯模板后端可直接用）
pip install "synomega[gnn]"    # 追加 D-MPNN 神经单步后端（torch），神经能力推荐
```

神经后端是**可选 extra**：不装 torch 也能用纯模板规则后端；需要神经模板分类精度（正向 / 逆向 /
演化 / 合理性都基于它）时再装 `[gnn]`。默认模型权重与 ZINC 可购砌块集在**首次调用时按需下载**到
本地缓存（`~/.cache/synomega`，可用 `SYNOMEGA_CACHE` 覆盖；`SYNOMEGA_MIRROR=ustc|github` 选源），
不随 wheel 分发。可显式预取：`synomega download`。要求 Python ≥ 3.10。

## 单步正向反应预测 {#forward}

给一组反应物，预测并排序最可能的**产物**。

```bash
synomega forward "CC(=O)O.NCc1ccccc1" --top-k 5     # 多分子用 . 连接
```

```python
from synomega.forward import ForwardTemplateGNN

fwd = ForwardTemplateGNN.default()                   # 首次调用自动下载正向模型
for pred in fwd.predict("CC(=O)O.NCc1ccccc1", top_k=5):
    print(pred.score, pred.product, pred.template_id)
```

输出为排序产物，每个带 `product`（SMILES）、`score`（正向预测概率）、`template_id`。原理见
[单步正向反应预测](research/forward.md)。

## 多组分演化预测 {#evolution}

从一组起始反应物出发，反复两两反应、把产物加回分子池，长出一张正向**合成网络**。每个分子带
**总合成分数**（`min(两父总分) × 单步概率`，起始物=1.0）与**合成树深度**。

```bash
synomega evolve --reactants "CC(=O)c1ccccc1.C=O.CNC" \
                --max-depth 3 --score-threshold 0.01 --out network.json
```

```python
from synomega.forward import ForwardTemplateGNN, MultiComponentEvolution

evo = MultiComponentEvolution(ForwardTemplateGNN.default(),
                              max_depth=3, score_threshold=0.01)
result = evo.evolve(["CC(=O)c1ccccc1", "C=O", "CNC"])   # 三组分 Mannich 起始物
print(result.describe())
for m in result.top(10, min_depth=1):
    print(m.total_score, f"d{m.depth}", m.smiles)
result.close()
```

常用参数：`--mode {memory,disk,auto}`（大量起始物用 `disk` 落 SQLite，需 `--work-dir`）、
`--forward-top-k`（每对取几个产物）、`--frontier-width`（每代配对数封顶，控扇出）、
`--no-self-pair`（禁 A+A 自反应）。原理见[多组分演化预测](research/evolution.md)。

## 单步逆向反应预测 {#retro}

给一个产物，预测并排序可能的**反应物**（拆分）。无独立命令行子命令——它是 `plan` / `score` 的
底层引擎；要单独取一步逆合成候选，用 Python：

```python
from synomega.singlestep import TemplateGNN

for p in TemplateGNN.default().predict("CC(=O)Nc1ccccc1O", top_k=5):
    print(p.score, p.reactants)          # p.reactants 是排序好的 canonical SMILES 元组
```

`TemplateGNN.simplify()` 是**简化约束变体**（只输出把目标拆成两个及以上前体的"简化型"拆分），
是可合成性评分的推荐后端。原理见[单步逆向反应预测](research/retro.md)。

## 多步反应路径规划 {#planning}

给一个目标分子，在 AND-OR 图上搜索到可购砌块的完整合成路径。

```bash
synomega plan --target "CC(=O)Nc1ccccc1O" --max-steps 5 --simplify
```

```python
import synomega

planner = synomega.load_default_planner()            # 默认原始模型 + retrostar
result = planner.plan("CC(=O)Nc1ccccc1O")
print(result.solved)
print(result.best_route.describe())
```

常用旋钮：`--algorithm {retrostar,mcts,bfs}`、`--expansion-width`（每节点取 top-k 候选）、
`--max-steps`（深度上限）、`--time-limit`、`--max-expansions`、`--exclude-target`（把目标本身
当作不可购，避免 0 步 trivially solved）、`--simplify`（用简化约束单步模型，搜索更省）。原理见
[多步反应路径规划](research/planning.md)。

## 反应合理性评分 {#plausibility}

对"这组反应物到底能不能给出该产物"打一个 0–1 的合理性分，用来**过滤**明显错误的单步拆分。
经评测其对单步 top-k 召回为净负收益，故**默认关闭**，需要时显式开启。

```python
import synomega

# 挂进 planner，自动筛掉每一步不合理的候选（只删不重排）
planner = synomega.load_default_planner(plausibility=True, plausibility_threshold=0.4)

# 或直接给一批候选反应打分
from synomega.plausibility import PlausibilityScorer
scorer = PlausibilityScorer.default()
scores = scorer.score_reactions(["CC(=O)O.NC>>CC(=O)NC"])   # 每条返回 0–1
```

默认关闭的依据见[反应合理性评分](research/plausibility.md)。

## 可合成性评分 SynScore {#synscore}

对一个目标分子给出连续的**可合成性分数** SynScore = \(1/(U+1)^U\)（`U` = 最优路径中不可购起始物
数；全部可购→1，缺得越多分越低），用于对一批分子排序。

```bash
# 默认用简化约束模型 @ 扩展宽度 k=10（评分推荐配置），--original 换原始模型
synomega score --targets targets.smi --out scores.jsonl
```

```python
import synomega

scorer = synomega.load_default_scorer()              # 默认 simplify=True, k=10
report = scorer.score("CC(=O)Nc1ccccc1O")
print(report.as_dict())                              # score / solved / min_steps ...

batch = scorer.score_batch(open("targets.smi").read().split())
print(batch.solve_rate)
```

分数定义与操作点见[可合成性评分 SynScore](research/synscore.md)。
