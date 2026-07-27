# SynOmega score vs SAscore/SCScore/RAscore (1000 ChEMBL targets)

数据：1000 个 ChEMBL 药物分子（`../target_set/targets.smi`，种子 20260727，见 `../target_set/`）。
评分：SynOmega bb_coverage / solved@5，统一预算 **top-10 / 8 s / 100 expansions / depth 5 / 60 s 硬超时**
（与效率对照完全一致，用非约束 full 模型；`syn_targets.csv` = `../efficiency_coverage/full.csv`）；
基线 SAscore（RDKit sascorer）、SCScore（Coley numpy standalone）、RAscore（XGB）。
脚本：`score_synomega.py` / `score_baselines.py` / `score_rascore.py`；相关性+CI：`bootstrap_ci.py`。

## 概况（n=1000）
- SynOmega solved@5 率 = **81.8%**；bb_coverage 均值 = 0.919（对全 1000，solved 记 1.0）。

## Spearman 相关（全集 n=1000，10000× bootstrap 95% CI）
| SynOmega bb_coverage vs | rho | 95% CI | 备注 |
|---|---|---|---|
| **SAscore** | **-0.536** | [-0.576, -0.492] | SA 低=易，bb 高=易 → 负相关正确，强 |
| **RAscore** | **+0.503** | [+0.459, +0.544] | RA 高=易 → 正相关正确，强 |
| SCScore | -0.208 | [-0.265, -0.150] | 符号正确但弱，不作主锚 |

三个 CI 均不含 0；SA/RA 强、SC 弱。两个独立可及性分交叉印证 SynOmega 分数的合理性。
（论文将此定位为 sanity check 而非 validity 证明：bb_coverage 在 ~82% solved 靶饱和为 1，
相关性主要由未解出尾部驱动。）

## 每分子耗时（成本）
`base_targets.csv`（`sa_sec`/`sc_sec`）、`rascore_targets.csv`（`ra_sec`）、`syn_targets.csv`（`sec`）逐条计时。

| 方法 | 中位 | 说明 |
|---|---|---|
| SAscore | 0.22 ms | 指纹/片段计数 |
| SCScore | 65 ms | 1024-bit FP 前向（含逐条调用开销）|
| RAscore (XGB) | 63 ms | ECFP counts + XGBoost |
| SynOmega bb_coverage | **0.57 s** | 真实多步路线搜索 |

SynOmega ~0.57s/分子,比最快的 SAscore 慢约 3 个数量级(比 SC/RA 约 9×),因其真做路线搜索。
尾部仍存在(retro* 对病态分子不严格执行 time_limit),硬 60s 超时封尾。

## 变更记录
- 2026-07-27：靶集从"反应库产物"换为 **ChEMBL 35 随机药物分子**（离分布外部测试），预算统一为 8s/100exp，
  相关性重算（SA/RA/SC 与旧值接近）。旧的 solved@5=0 子集论证已从论文移除（bb_coverage 浮点意义不再作卖点）。
- 2026-07-27：**SynOmega 分数改为 `score = bb_coverage + (solved 时 +1)`**（solved→2、unsolved→[0,1)），
  加大 solved/unsolved 间隔（包 `MoleculeReport.score`）。这是逐序列单调变换,**Spearman 相关系数不变**
  （−0.536/+0.503/−0.208 与用 bb_coverage 完全相同,`bootstrap_ci.py` 已改用 score 验证）;solve rate 亦不变。
