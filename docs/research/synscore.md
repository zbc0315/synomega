# 可合成性评分 SynScore

**任务**：给一个目标分子一个 \([0,1]\) 的**可合成性分数**——它有多容易从可购原料合成出来。
与只看分子结构的启发式打分（SAscore/SCScore/RAscore）不同，SynScore 直接**跑一次多步路径
搜索**，用"最优路径里还差几个不可购起始物"来定分，因此可解释、可排序、带路径依据。

## 1. 定义

设最优路径中**不可购起始物**的数量为 \(U\)，则

\[
\mathrm{SynScore} = \frac{1}{(U+1)^{U}}
\]

| \(U\) | 0 | 1 | 2 | 3 | 4 | ≥5 | 无路径 |
|---|---|---|---|---|---|---|---|
| SynScore | 1.0 | 0.5 | ≈0.111 | ≈0.016 | ≈0.0016 | ≤1e-3 | 0 |

- \(U=0\)（全部叶子可购、深度 ≤5）称为 **solved**，得满分 1。
- 分数随 \(U\) **超线性衰减**——比"可购叶子比例"这种线性覆盖度分离得更陡，越接近完全可解
  的分子越被拉开差距。
- **最优路径**的选择：有解时取**反应步数最少**的解；无解时退而取 \(U\) 最小（再看覆盖度、
  步数）的部分路径，以携带"近失"信号。

配套的连续指标 **bb_coverage**（最优路径中可购叶子占比）在批量分析中给出比二值 solved 更
平滑的分布。

## 2. 算法与伪代码

SynScore 由 `SynthesizabilityScorer` 驱动一次规划、再从路径聚合而来：

```text
function synscore(smiles, max_steps=5):
    result = planner.plan(smiles, max_depth=max_steps)     # 见「多步路径规划」
    routes = result.routes
    solved = [r for r in routes if r.solved]
    if solved:
        best = argmin(solved, key = r.num_steps)           # 步数最少的解
        coverage = 1.0
    elif routes:
        best = argmin(routes, key = (U(r), -r.bb_coverage, r.num_steps))
        coverage = best.bb_coverage
    else:
        return MoleculeReport(solved=False, bb_coverage=0)  # score 记 0

    U = number_of_non_purchasable_leaves(best)
    score = 1 / (U + 1) ** U
    return MoleculeReport(smiles, solved=(U==0), bb_coverage=coverage,
                          min_steps=best.num_steps, score=score, leaves=...)
```

批量打分 `score_batch` **顺序执行**（规划器持 GPU + 共享缓存，不宜朴素多进程；大规模应在
shell 层分片），可通过回调流式落盘，并在 stderr 打印实时 solve_rate。聚合结果
`BatchReport` 给出 `solve_rate`、`mean_bb_coverage`、按路径深度的直方图等。

## 3. 评测

### 3.1 U 分布（1000 ChEMBL）

![U 分布：原始 vs 简化](../figures/udist.svg){ loading=lazy }

| U | 0 | 1 | 2 | 3 | 4 | ≥5 | 无路径 |
|---|---|---|---|---|---|---|---|
| 原始模型 | 818 | 106 | 61 | 10 | 3 | 1 | 1 |
| 简化模型 | 851 | 11 | 21 | 26 | 17 | 72 | 2 |

原始模型在解不出时通常只差 1–2 个不可购起始物（U 集中在 1–2）；简化模型 solved 更多
（851 vs 818），但一旦解不出往往差得更远（72 个靶点 U≥5）——因为它每步都强制拆分，
够不着"一步买到"的收尾。这正是 SynScore 超线性衰减要区分开的两种"未解"。

### 3.2 与结构类基线的定位

结构类可合成性打分很快，但对"明明可购"的分子给不出一致判断：

![结构类基线的单分子打分耗时](../figures/baseline_time.svg){ loading=lazy }

| 打分 | 取值范围 | 中位/均值 | "偏离理想"占比 | 单分子耗时 |
|---|---|---|---|---|
| SAscore | 1.3–7.2 | 中位 2.7 | 32% 分数 >3 | **0.22 ms** |
| RAscore | 0.001–1.0 | 均值 0.91 | 17% 分数 <0.9 | **63 ms** |
| SCScore | 1.0–5.0 | 中位 3.7 | 99% 分数 >2 | **65 ms** |

在 20,000 个**本就可购**的 ZINC 砌块上，三种结构类打分没有一个能给出一致的"平凡可得"判断
（各自把相当比例的可购分子判为偏难）。SynScore 的取舍相反：它**付出一次路径搜索的代价**
换取一个**有路径依据**的判断——贵，但可解释、可给出具体路径。

!!! note "口径说明"
    SynScore 是搜索式指标，其"耗时"即多步搜索时间（见[规划章](planning.md)，中位约
    0.3–0.5 s），与上表的结构类打分**不在同一量纲**，故不做同轴对比；此处只对比"能否对可购
    分子给出一致判断"。

## 4. 局限

- SynScore 的质量完全由底层单步模型与砌块集覆盖决定：模型召回差或砌块集小，都会低估可合成性。
- 单分子需一次搜索，成本远高于结构类打分；大规模筛选需分片并行 + 缓存。
- \(U\) 只数"不可购起始物个数"，不区分这些起始物各自离可购有多远——是刻意的简化。
