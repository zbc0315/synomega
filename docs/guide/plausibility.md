# 反应合理性评分

对"这组反应物到底能不能给出该产物"打一个 0–1 的合理性分，用来**过滤**明显错误的单步拆分。
经评测其对单步 top-k 召回为净负收益，故**默认关闭**，需要时显式开启。模型与评测见
[研究报告 · 反应合理性评分](../research/plausibility.md)；安装见[安装与总览](index.md)。

```python
import synomega

# 挂进 planner，自动筛掉每一步不合理的候选（只删不重排）
planner = synomega.load_default_planner(plausibility=True, plausibility_threshold=0.4)

# 或直接给一批候选反应打分
from synomega.plausibility import PlausibilityScorer
scorer = PlausibilityScorer.default()
scores = scorer.score_reactions(["CC(=O)O.NC>>CC(=O)NC"])   # 每条返回 0–1
```
