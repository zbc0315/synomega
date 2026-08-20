# 单步逆向反应预测

**做什么**：给一个产物分子，预测可能的**反应物**（一步拆分 / 断键）并排序。它是[多步规划](planning.md)
与[可合成性评分](synscore.md)的底层引擎，没有独立命令行子命令；要单独取一步逆合成候选，用 Python。
模型原理与评测见[研究报告 · 单步逆向反应预测](../research/retro.md)；安装见[安装与总览](index.md)。

**输入 / 输出**：输入 = 产物 SMILES；输出 = 排序候选，每个是一组反应物（canonical SMILES 元组）+
分数（模板概率）。

## Python

```python
from synomega.singlestep import TemplateGNN

model = TemplateGNN.default()                 # 首次调用自动下载默认模型
for p in model.predict("CC(=O)Nc1ccccc1O", top_k=5):
    print(round(p.score, 4), p.reactants)     # reactants 是排序好的 canonical 元组
    print(p.smiles)                            # = ".".join(p.reactants)，反应物侧单串 SMILES
    print(p.template_id, p.meta["center_avg"])
```

`predict` 返回 `Prediction` 列表：`reactants`（元组）、`score`（0–1 模板概率）、`template_id`、
`meta["center_avg"]`（反应中心平均置信，用于**同一模板不同匹配位点**之间的二次排序）。默认
`top_k=50`；批量用 `model.predict_batch([...])`。

## 两个后端

| 入口 | 动作空间 | 用途 |
|---|---|---|
| `TemplateGNN.default()` | 全部 64,366 个模板 | 通用单步逆合成、多步规划 |
| `TemplateGNN.simplify()` | 仅"简化型"拆分（把目标拆成 ≥2 个前体） | 可合成性评分推荐后端；多步搜索更省 |

用自己的 checkpoint：`TemplateGNN.from_pretrained("run_dir")`，或环境变量 `SYNOMEGA_MODEL` /
`SYNOMEGA_SIMPLIFY_MODEL` 指向 run 目录。

## 注意

- 单步只看一步；要一路拆到可购砌块请用[多步规划](planning.md)。
- 命中正确模板不一定唯一复现真实反应物（区域 / 位点多解），这是模板法的结构性上限；`center_avg`
  正是用来在这些同模板候选间挑更靠谱的那一个。
