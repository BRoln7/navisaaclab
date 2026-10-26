# 连续正常走路筛选与走跑采样调整

本轮针对“跑步自然、走路异常、走跑衔接差”的反馈整理数据。
没有启动训练、策略推理或 GPU 检查；训练和测试命令由用户运行。

## 数据与比例

新默认列表：`data/mimickit/official_smpl/locomotion-curated-v1/motions.yaml`。
原始 PKL、转换数据、`motions-run6.yaml` 和已有 checkpoint 均保留。

| 组别 | 新数据 | 原始来源数 | 时长 | 原 run6 采样概率 | 新概率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 连续正常向前走路候选 | 161 段 | 98 | 487.93 秒 | 走路标签共 57.92% | 55% |
| 较快跑步候选 | 21 段 | 21 | 32.90 秒 | 34.43% | 30% |
| 标注慢跑 | 3 段 | 3 | 38.87 秒 | 4.92% | 10% |
| 真实走跑转换 | 2 段 | 2 | 8.90 秒 | 0.55% | 5% |

总计 187 段、17,245 帧、568.60 秒（9.48 分钟）。
161 段走路来自 98 段原动作的不同连续区间，不是新增独立动作。
跑步加慢跑总概率为 40%，与上一轮的 39.34% 接近。
此次主要改变走路内容质量、慢跑曝光和转换采样，而非再次大幅改变走跑总比例。

## 筛选规则

规则在 `data/cfg/mimickit_amp/motion_curation.yaml`，实现位于
`socnav_sim/mimickit_amp/curation.py`，可用
`python scripts/mimickit_amp/prepare_curated.py` 重现数据处理。

走路片段需同时满足：

- 排除名称明确表示猫步、单腿走、扶栏杆/平衡木、思考式走法、背包、倒退、
  蹲走、跛行等特殊风格的原片段。这只是粗筛，不把所有 walk 标签当正常走路。
- 0.5 秒平滑后的根节点 XY 速度为 0.55–1.65 m/s。
- 根节点朝向与移动方向余弦至少 0.75，剔除明显横向/反向移动区间。
- 官方 SMPL 前向运动学计算的骨盆到头部方向，竖直分量占比至少 0.88。
- 0.75 秒窗口内的平滑速度标准差不超过 0.35 m/s。
- 1 秒平滑移动方向的转速不超过 1 rad/s；根节点瞬时速度不超过 3 m/s。
- 满足全部条件的连续区间至少长 2 秒，不跨越失效帧拼接。

在原 212 段走路标签动作中，98 段有合格连续区间；7 段按特殊名称剔除；
107 段没有符合上述长度和运动学约束的连续区间。
这个结果不意味着被排除的数据本身无效，只表示不用于当前普通向前行走实验。
侧步、走跳、躲闪等另外 8 段暂不纳入本列表。

裁剪后只对 XY 原点做常量平移。高度、根旋转、关节旋转、FPS、帧顺序均保留。
没有跨片段拼接、插值合成或时间缩放。21 段快跑、3 段慢跑及两段转换
按原文件字节完整复制。

组间使用固定概率；组内先平衡原始来源，再按各来源的合法历史时间分配到
它的不同裁剪段。因此一条长录制切成十段，不会获得十倍采样概率。
仍通过 MimicKit 原生 YAML weight 抽样，同时作用于 demo 和参考状态初始化。

## 速度覆盖变化

下表使用合法的 10 帧、0.3 秒历史窗口估计原生连续采样的速度分布。

| 窗口均速 | 原 run6 列表 | 新筛选列表 |
| --- | ---: | ---: |
| <0.2 m/s | 17.10% | 0.54% |
| 0.2–0.8 m/s | 20.04% | 11.23% |
| 0.8–1.4 m/s | 22.81% | 51.50% |
| 1.4–2.0 m/s | 5.37% | 5.51% |
| 2.0–2.5 m/s | 1.74% | 1.49% |
| ≥2.5 m/s | 32.93% | 29.74% |

近静止历史主要留在完整转换片段中。速度阈值是运动学指标，不是接触力步态标签。
这些筛选不能证明足部接触、左右对称或视觉流畅度已改善，需要运行视频检查。
1.4–2.0 m/s 的覆盖仍然有限，两段转换的独立内容仍只有 8.9 秒。
提高它们的概率不增加转换多样性，也不会给纯 AMP 增加速度指令。
本列表聚焦持续走跑，站立和停步的覆盖随之减少；站立起步、停止后的稳定性
需要另外评估，不能从连续运动片段的筛选结果推断。

## 建议的训练命令

为保留当前跑步能力，可从第 1900 轮权重继续，并显式选择新列表：

```bash
./isaaclab.sh -p scripts/mimickit_amp/run.py \
  --mode train --amp_only --headless --device cuda:0 --num_envs 16384 \
  --checkpoint logs/mimickit/smpl-amp/20261019-191801-878449/int_models/model_0000001900.pt \
  --motion_file data/mimickit/official_smpl/locomotion-curated-v1/motions.yaml \
  --max_iterations 1000 --seed 0
```

这里 1000 是新增训练轮数和一次观察预算，不是预计收敛所需轮数。
网络及归一化参数从 checkpoint 加载，优化器、replay 和轮数记录重新初始化。
`symmetry_weight` 保留 0.1；本轮没有同时修改损失、奖励或控制时间间隔。
从头训练则去掉 `--checkpoint`，默认 env 配置已选择新列表。

## 分组评估命令

将 `AMP_CKPT` 设置为要评估的新 `.pt` checkpoint，保留原运行目录的 `params/`。
走路初态与快跑初态分别录像：

```bash
./isaaclab.sh -p scripts/mimickit_amp/run.py \
  --mode play --checkpoint "$AMP_CKPT" --headless --video --device cuda:0 \
  --num_envs 4 --steps 600 \
  --motion_file data/mimickit/official_smpl/locomotion-curated-v1/motions-walking.yaml

./isaaclab.sh -p scripts/mimickit_amp/run.py \
  --mode play --checkpoint "$AMP_CKPT" --headless --video --device cuda:0 \
  --num_envs 4 --steps 600 \
  --motion_file data/mimickit/official_smpl/locomotion-curated-v1/motions-running.yaml
```

另有 `motions-jogging.yaml` 和 `motions-transition-starts.yaml`。
后者只从转换片段的状态初始化，不会命令策略按参考动作完成走跑转换。
受控切换仍需后续直线平滑变速任务。

CPU 回归用例已写入，按约定由用户运行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q tests/test_mimickit_curation.py tests/test_mimickit_amp.py tests/test_mimickit_video.py
```

`selection.csv` 逐条记录原动作的纳入/排除原因，`manifest.json` 记录裁剪范围、
来源与输出哈希、组权重及各列表哈希。相同输入可复用，输入变化时拒绝覆盖旧输出目录。
