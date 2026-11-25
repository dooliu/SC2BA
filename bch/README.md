# HRL-IM/CBS on SC2BA

本目录从 HRL-IMCBS-test 的 `6b8a5a9` 移植原始 `Q-learning_new.py` 和后续独立训练的 CNN 编码。训练和冻结评测共用 SC2BA 的 `COMMON` 地图、对手控制器、软重置和终止判定。

| 目录或文件 | 职责 |
| --- | --- |
| `upper_state/` | 势力矩阵、编码器、离线训练及码本。上层 Q 表只使用 `UpperEncoder.key(observation)` 的返回值。 |
| `lower_action/` | 五种 CBS 分组、下层状态、11 种宏动作和短期奖励。宏动作输出有序的单位指令。 |
| `learner.py` | 两层 Q 表、更新和快照。保留原下层按当前分组表查找上一次决策的更新语义。 |
| `arena.py` | 观测、宏动作到 SC2BA 的适配，以及开局和结束审计。 |
| `collect.py`、`run_policy.py` | 探针采集、Q 表训练和冻结评测。 |

### 上层方法

| 参数 | 输入与码字 |
| --- | --- |
| `im-hash` | 原始有符号势力图、RdBu 渲染、逐 RGB 通道直方图均衡及量化，得到 24 位十六进制字符串。直接用于 Q 表。 |
| `im-medoid` | 从训练哈希拟合固定数量的 Hamming medoid，部署时分配最近的码字，供等码字预算对照。 |
| `im-grid` | 单通道 IM 矩阵，归一化、居中补零到 128×128，四层 Conv2d 产生嵌入。 |
| `grid-zero`、`grid-density` | 与 `im-grid` 共用主干。第二通道分别是零和只由当前 IM 得出的潜在火力密度。 |
| `unit-set` | 双方单位的绝对和相对坐标，沿用 1×1 Conv1d、掩码 sum/max 聚合与 MLP。 |

各神经编码器独立拟合。几何头预测当前帧的兵力与双向关系直方图。短期 HP 差变化预测头还读取探针分组、宏动作和对手，编码器本身只读取当前状态。用验证布局选择权重，再用训练布局的嵌入标准化并拟合 medoid。部署时仅用冻结的编码器和码本。

移植沿用原势力叠加、裁剪、哈希、分组和奖励定义。SC2BA 使用连续世界坐标供下层决策，势力图按一世界单位一格离散化。历史 `do_randomly` 中的固定攻击点转换到 `COMMON` 的坐标，移动目标限制在场景的可玩区域内。默认每次决策走 8 帧，并采用 SC2BA 各场景的回合上限。旧 MM 地图的观测、权重与 Q 表仍对应原平台；用 SC2BA 探针重新拟合本目录的编码器和策略。

### 开局和数据划分

[默认计划](configs/probes.json)覆盖 `3m`、`8m`，每张图四种开局和三个对手（Bot、QMIX、VDN）。自定义布局可以指定红方出生中心、指定双方中心，或分别列出双方每个 Marine 的生成坐标。各场景可选用 SC2BA 已配备权重的对手。

- 每种「场景 × 布局 × 对手」会采集两个行为策略。`legacy-nearest` 保留旧探针的组内就近攻击，`mixed` 覆盖全部 11 种宏动作。两者都采样五种分组。每个短窗口固定分组规则和宏动作，组成员及命令随实时观测重算。
- 每个布局只属于 `train`、`validation`、`test` 中的一组。同一布局的重复对局、不同对手及两种采集策略共用该划分。可用 `collect --behavior mixed` 仅采集混合策略，或用 `fit --behavior legacy-nearest` 在共用档案上只用旧策略样本拟合。
- `probes.jsonl.gz` 保存观测、行为策略、分组和宏动作、宏动作生成的指令、短期 HP 差变化与窗口长度。`episodes.jsonl` 保存逐局种子、开局实测兵位及血量、终局 HP 和引擎帧。`manifest.json` 保存计划、文件摘要及覆盖统计。定性关系直方图数衡量本次访问到的几何关系类型，保留了双方人数。
- 每个场景在进程内切换对手和布局，逐局检查满血、人数、引擎帧是否连续和终止条件。SC2 对局异常时该次运行保留已写的日志，探针拟合需要完整的归档清单。

### 运行

在已能运行 SC2BA 的 Python 环境中，从仓库根目录执行。已用 Python 3.8、Torch 2.4 CPU 和 SC2 build 97425 验证。

```powershell
python -m pip install -r bch/requirements.txt
python -m bch collect --scenario 3m --seed 7 --repeats 20 --output bch/runs/probes-3m
python -m bch fit --probes bch/runs/probes-3m --scenario 3m --method im-grid --seed 7 --output bch/runs/fit-grid
python -m bch train --scenario 3m --method im-hash --seed 23 --repeats 100 --output bch/runs/train-hash
python -m bch train --scenario 3m --method im-grid --encoder bch/runs/fit-grid/encoder.pt --seed 23 --repeats 100 --output bch/runs/train-grid
python -m bch evaluate --scenario 3m --policy bch/runs/train-hash/policy.json --seed 41 --repeats 30 --output bch/runs/test-hash
python -m bch evaluate --scenario 3m --policy bch/runs/train-grid/policy.json --encoder bch/runs/fit-grid/encoder.pt --seed 41 --repeats 30 --output bch/runs/test-grid
```

`--repeats` 表示每个适用组合的对局数。`fit` 默认用 160 轮、32 维和 32 个码字，并支持调整 `--epochs`、`--latent-dim` 和 `--codes`。做编码对照时复用探针档案，给两种策略传相同的训练与评测种子。跨地图时把拟合和 Q 表训练限制在源地图上，再用同一计划中目标地图的测试布局评测。正式多种子对照用不同的种子和输出目录重复运行。

冻结评测从 `policy.json` 读取 Q 表，运行后核对表内容，按地图与对手写出胜负、步数和访问未见 Q 键的数据。输出目录 `bch/runs/` 用于本地数据、权重与报告。
