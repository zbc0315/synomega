# bb-coverage 合成性评分 对照实验结果（JCIM）

日期：2026-07-26
数据：1000 个目标分子（`benchmarks/search_algo_2026-07/data/targets.smi`）
评分：SynOmega bb_coverage / solved@5（max_steps=5, time_limit=12, max_expansions=150, GPU1 八分片）；
基线 SAscore（RDKit sascorer）、SCScore（Coley, numpy standalone）。
脚本/数据：`benchmarks/synth_eval_jcim/{scripts,data}`。RAscore 待补。

## 概况（n=1000）
- solved@5 率 = 0.766；bb_coverage 均值 = 0.878。

## Spearman 相关（全集，n=1000，含 RAscore 三方）
| bb_coverage vs | rho | p | 备注 |
|---|---|---|---|
| **SAscore** | **-0.522** | 5.9e-71 | SA 低=易，bb 高=易 → 负相关正确 |
| SCScore | -0.209 | 2.7e-11 | 弱 |
| **RAscore** | **+0.518** | 1.1e-69 | RA 高=易，bb 高=易 → 正相关正确，**强** |

- solved vs SAscore -0.508；solved vs SCScore -0.225；SAscore vs SCScore +0.284。
- **RAscore(+0.518)与 SAscore(−0.522)量级/符号高度一致**，两个独立的逆合成可及性分交叉印证 bb_coverage。SCScore 是弱且不稳的离群者。

## 核心论点检验：solved@5==0 子集（二值指标塌缩为 0），n=234
- bb_coverage：**均值 0.477，std 0.278，range [0.00, 0.92]** —— 二值可解性全为 0 的难分子区间，**连续 bb_coverage 仍有大幅方差、仍在区分难易**。
- bb_coverage vs **SAscore**（子集内）：**rho=-0.387, p=9.2e-10** —— 独立复杂度分仍显著相关，方向正确。
- bb_coverage vs **RAscore**（子集内）：**rho=+0.363, p=1.1e-08** —— 第二个独立分**同样在难分子区间显著相关**，符号正确。
- bb_coverage vs SCScore（子集内）：rho=+0.234, p=3.0e-04 —— **符号翻转**（SCScore 不稳，见下）。
- → **两个独立可及性分(SA、RA)在二值塌缩处都仍与 bb_coverage 显著相关**，强化"连续指标在难分子区间仍保信息量"这一核心论点。

> 这就是本论文的核心卖点：`solved@N` 在难分子上退化为常数 0，无区分力；而 `bb_coverage` 在同一区间保留方差并与外部复杂度分相关，仍可用于排序/优选。

## 诚实的caveat
1. **SCScore 是个不稳的基线**：全集仅 -0.209，难分子子集里甚至**翻正**（+0.234）。SAscore 一致且稳健，SCScore 不宜作为主锚。论文中应以 SAscore 为主，SCScore 仅作参考，或补 RAscore。
2. bb_coverage 与 solved 高度同源（都来自同一路线搜索），二者相关是预期的；论文要强调的是"**连续 vs 二值**"的信息量差，而非与外部分的绝对相关强度。

## 分歧案例（SAscore 判易、bb_coverage 判难、solved=0）
多为**稠合多环芳烃/多稠环体系**：SA 因"芳环常见"给低分（判易），但路线搜索给不出可购买叶子（bb_coverage 0.5–0.67）。
- `C(#Cc1c2ccccc2c(C#Cc2ccccc2)c2cc3ccccc3cc12)c1ccccc1`  SA=2.23 bbcov=0.67
- `Oc1c(I)cc2ccccc2c1-c1c(O)c(I)cc2ccccc12`  SA=2.47 bbcov=0.50
- `CN(C)CCN1c2ccccc2CS(=O)(=O)c2ccc(Cl)cc21`  SA=2.54 bbcov=0.50

定性结论：SA 低估稠合多环/含卤芳烃的合成难度；基于真实路线的 bb_coverage 能捕捉到。

## 每方法·每分子耗时（成本对比）
数据列：`base_targets.csv` 含 `sa_sec`/`sc_sec`，`syn_targets.csv` 含 `sec`（均为每分子逐条计时）。

| 方法 | 中位 | 均值 | p95 | max | 说明 |
|---|---|---|---|---|---|
| SAscore | 0.27 ms | 0.29 ms | 0.5 ms | 4.7 ms | 指纹/片段计数，几乎免费 |
| SCScore | 0.92 ms | 0.95 ms | 1.4 ms | 5.1 ms | 1024-bit FP 前向 |
| RAscore (XGB) | 17.4 ms | 17.2 ms | 22.7 ms | — | ECFP counts + XGBoost |
| SynOmega bb_coverage | **1.33 s** | 37.7 s* | 63.7 s | 6839 s* | 真实多步路线搜索 |

三个标量预测器都 ≤25ms；bb_coverage ~1.3s，因其真做路线搜索（比 RAscore ~70×、比 SAscore ~4600×）。

### 干净重测（单进程、无 GPU 争用，代表性子集 n=150）
在 GPU1 单进程重跑（`data/syn_targets_timing.csv`，mols 1-150，与打分同配置 time_limit=12/max_exp=150）：

| 统计 | 值 |
|---|---|
| median | **1.25 s** |
| mean | 8.28 s |
| p90 | 18.4 s |
| p95 | 48.0 s |
| max | 143 s |

**两个重要发现（修正之前的判断）：**
1. **中位数稳健**：干净 1.25s ≈ 争用版 1.33s —— 代表性单分子成本就是 **~1.3s**。
2. **尾部是真实的，不全是争用**：即使单进程无争用，仍有分子搜索拖到 48-143s。根因是 **retro\* 对病态分子（复杂稠环/长链）不严格执行 time_limit=12s**（时间检查在扩展之间，单次扩展可能超时）。GPU 争用是在此**之上**把均值从 8.28s 进一步抬到 37.7s（分片版）。
3. 全 1000 单进程重测**不可行**：个别分子会卡几十分钟（重测在第 151 个分子卡死 40+ 分钟，已终止）。若需干净全量成本，必须给每分子加**硬 wall-clock 超时杀进程**来封住尾部。

结论：bb_coverage 代表性成本 **~1.3s/分子**，比 SAscore/SCScore（亚毫秒–毫秒）贵约 **10³ 倍**，因为它真的在搜路线——"信息量 vs 成本"的权衡：难分子区间它仍能区分（见上），代价是每分子秒级搜索 + 需要硬超时封尾部。

## 后续
- ~~补 RAscore 三方相关~~ ✅ 已完成（RAscore XGB；env 需 numpy<2 + xgboost==1.2.1 + scikit-learn==1.0.2 才能加载官方 pickle 模型）。数据 `data/rascore_targets.csv`，脚本 `scripts/score_rascore.py`。
- 若投稿需干净全量成本：给 `score_synomega.py` 每分子加硬 wall-clock 超时（SIGALRM/子进程），封住 retro\* 未遵守 time_limit 的尾部。
- 外部可制性锚点（专家/实测）仍是投稿必需，当前缺。
