# 多组分演化预测

**做什么**：给一组起始反应物，反复从"分子池"里挑两个分子反应、把产物加回池，长出一张正向**合成
网络**，用来看"这些原料多轮反应能演化出哪些分子、经哪条路线、可信度多高"，适合探索多组分 / 一锅法
体系。算法与端到端验证见[研究报告 · 多组分演化预测](../research/evolution.md)；安装见
[安装与总览](index.md)。

**每个分子带两个量**：**总合成分数** total（`min(两父总分) × 单步正向概率`，起始物 = 1.0，"最弱
环节"式乘积）；**合成树深度** depth（`max(两父深度) + 1`，是树高不是步数）。

## 命令行

```bash
synomega evolve --reactants "CC(=O)c1ccccc1.C=O.CNC" \
                --max-depth 3 --score-threshold 0.01 --out network.json
```

真实示例输出（三组分 Mannich：苯乙酮 + 甲醛 + 二甲胺）：

```
molecules: 95497  pairs-run: 30628  reaction-edges: 145713  rounds: 3  stop: exhausted
top 15 products by total score:
  0.9021  d1  C=CC(=O)c1ccccc1       (step=0.9021)   ← 烯酮中间体（羟醛缩合脱水）
  0.7611  d2  CN(C)CCC(=O)c1ccccc1   (step=0.8438)   ← 经典 Mannich 碱（aza-Michael 加成）
  0.4990  d1  CN(C)C ...
```

网络在 d1 长出烯酮中间体、d2 命中 Mannich 碱，与教科书机理一致。`--out network.json` 存整张网络
（含全部反应边），供后续分析。

## Python

```python
from synomega.forward import ForwardTemplateGNN, MultiComponentEvolution

evo = MultiComponentEvolution(ForwardTemplateGNN.default(),
                              max_depth=3, score_threshold=0.01)
result = evo.evolve(["CC(=O)c1ccccc1", "C=O", "CNC"])   # 三组分 Mannich 起始物
print(result.describe())                    # 汇总 + top 产物
for m in result.top(10, min_depth=1):       # min_depth=1 只看真正的产物（排除起始物）
    print(m.total_score, f"d{m.depth}", m.smiles)
for edge in result.reactions():             # 遍历全部反应边
    print(edge.reaction_smiles, edge.step_score)
result.close()                              # disk 模式务必 close（释放 SQLite 句柄）
```

`result` 常用方法：`describe()`、`top(n, min_depth=, min_score=)`、`reactions()`、
`best_route(smiles)`（回溯某分子的最优路线）、`to_json()`。用 `with evo.evolve(...) as result:` 可
自动 close。

## 参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--max-depth` | 必填 | 合成树深度上限（闸"能否继续作反应物"，非步数） |
| `--score-threshold` | 必填 | 低于此总分的分子不可再作反应物；越大剪枝越狠、越快 |
| `--forward-top-k` | 5 | 每对反应取几个产物 |
| `--mode {memory,disk,auto}` | memory | 大量起始物用 `disk`（落 SQLite，需 `--work-dir`）；`auto` 按起始物数自动切 |
| `--frontier-width` | 无上限 | 每代只配对分数最高的前 N 个分子，封住 O(n²) 扇出 |
| `--no-self-pair` | 默认允许 A+A | 禁止分子与自身反应 |

## 注意

- 计算量随起始物数呈 O(n²)，大规模务必配 `--frontier-width` + `--mode disk`。
- 分数是路线**可信度**的相对排序，不是产率 / 热力学可行性，也不替代对条件、选择性的判断。
- 分数回传播为保留一条更高分路线，可能把某分子记录深度抬到 > `max_depth`（分数优先，属预期）。
