# 可合成性评分 SynScore

**做什么**：给一个目标分子打一个连续的**可合成性分数** SynScore，用来对一批分子按"好不好合成"
排序。它在内部跑一次[多步规划](planning.md)，再把结果折成分数。分数定义与操作点见
[研究报告 · 可合成性评分 SynScore](../research/synscore.md)；安装见[安装与总览](index.md)。

$$\mathrm{SynScore} = \frac{1}{(U+1)^{U}}$$

`U` = 最优路线里**不可购起始物**的个数：全可购(U=0)→1.0，U=1→0.5，U=2→0.11，U=3→0.016，完全找
不到路线→0。缺得越多、掉得越狠，所以它能把"完全解出 / 差几个砌块 / 差很多"清晰分开。

## 命令行

```bash
# 默认用简化约束模型 @ 扩展宽度 k=10（评分推荐配置），--original 换原始模型
synomega score --targets targets.smi --out scores.jsonl
```

## Python

```python
import synomega

scorer = synomega.load_default_scorer()              # 默认 simplify=True, k=10
r = scorer.score("CC(=O)Nc1ccccc1O", max_steps=5)
print(r.as_dict())
# {'smiles': ..., 'solved': True, 'score': 1.0, 'bb_coverage': 1.0,
#  'min_steps': 2, 'min_route_depth': 2, 'num_leaves': 2,
#  'num_purchasable_leaves': 2, 'expansions': ..., 'terminated_by': 'solved', ...}

# 批量：每行一个 SMILES
report = scorer.score_batch(open("targets.smi").read().split())
print(report.solve_rate, report.mean_bb_coverage)
print(report.describe())
df = report.to_dataframe()                            # 每分子一行（需要 pandas）
```

单分子返回 `MoleculeReport`（`.score` / `.solved` / `.bb_coverage` / `.min_steps` /
`.num_unpurchasable_leaves`（=U）/ `.as_dict()`）；批量返回 `BatchReport`（`.solve_rate` /
`.mean_bb_coverage` / `.describe()` / `.to_dataframe()` / `.to_json()`）。

## 读分数

| 情况 | U | SynScore |
|---|---|---|
| 全部起始物可购（solved） | 0 | 1.0 |
| 差 1 个砌块 | 1 | 0.5 |
| 差 2 个 | 2 | 0.11 |
| 完全无路线 | — | 0 |

`solved` 是二元的"能不能在深度内全叶可购"，用来和文献 `solve_rate` 对比；`SynScore` 是连续、
近失敏感、可排序的版本，用来对一批分子排序。

## 参数

| 参数（CLI / Python） | 默认 | 含义 |
|---|---|---|
| `--targets` | 必填 | 每行一个 SMILES |
| `--original` / `simplify=` | 默认简化模型 | 换回原始单步模型打分 |
| `--max-steps` / `max_steps=` | 5 | 路线深度上限 |
| `--exclude-target` | 关 | 把目标本身当不可购（避免可买目标 0 步满分） |
