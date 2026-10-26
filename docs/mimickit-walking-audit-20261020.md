# MimicKit 接入审查：跑步自然、走路僵硬

后续实施：本报告下文记录的是修正前快照。已增加梯度累积、恢复原版逻辑
批量和 20 万 replay，并将新训练默认设为重建后的均匀数据、10 秒回合、
global_obs=true 的纯 AMP。保留 sym_loss=0.1；新回归用例和 GPU 检查待用户
执行。最新运行说明见 [接入文档](mimickit-smpl-amp.md)。

本次只检查源码、配置和已有日志，并运行 CPU 配置计算；没有启动模拟器、
训练、策略推理或 pytest。没有修改训练代码、配置、数据或 checkpoint。

对照版本：MimicKit `fbe1a36e878f036d6bf5521defb8eb3b8ab4170d`。
本地 upstream checkout 的 `git status --short` 为空。

## 结论与证据边界

暂未发现本接入在 PPO、GAE、判别器标签/损失、风格奖励公式或镜像向量符号上
存在可以确定的接线错误。但发现一处明确改变原版训练节奏的设置：大环境数下
缩小优化 minibatch，仍按相同 epoch 遍历整个 rollout，使 optimizer.step 次数增加。
这不是单纯的显存优化，不能称作训练设置与原版相同。

已有数据回放自然以及 USD/FK 一致，只验证运动学内容和映射。它们不能证明
PD 控制、接触动力学、探索以及对抗学习已经没有问题，也不能证明每个片段
都没有局部异常。

## 已确认：minibatch 上限改变了 optimizer 更新次数

来源：`data/cfg/mimickit_amp/agent.yaml` 的 `batch_size_scaling`，以及
`mimickit_amp/config.py::resolve_agent_runtime`。
上游 `ppo_agent.py::_update_model` 与 `amp_agent.py::_update_model` 按
`ceil(rollout_samples / minibatch_samples) * epochs` 计算更新次数。

在 16384 环境、32 步 rollout（524288 样本）下：

| 网络 | 上游 batch | 当前 batch | 上游每轮更新 | 当前每轮更新 |
| --- | ---: | ---: | ---: | ---: |
| Actor | 65536 | 16384 | 40 | 160 |
| Critic | 32768 | 8192 | 32 | 128 |
| Discriminator | 32768 | 8192 | 32 | 128 |

4096 环境时两者相同；8192 环境时当前更新次数是上游的 2 倍；12288 时为
3 倍；16384 时为 4 倍。这里比较的是同环境数、同一轮 rollout，不是在声称
较小 batch 或更多更新本身一定错误。

三个优化器都使用上游 SGD（momentum=0.9），学习率保持上游值。
改变 optimizer.step 次数会改变参数和 momentum 的演化，不能用“epoch
没变”证明等价。Actor、critic 和 discriminator 都受到影响，不能据此单独
断言判别器比 actor 获得了更多相对训练。

建议修正方向：将逻辑 batch 与显存 microbatch 分开。保持上游逻辑 batch
和 optimizer.step 次数，在一个逻辑 batch 内累积各 microbatch 按样本数
加权的梯度，最后执行一次原生 optimizer 更新。不能只把所有计算图相加后
统一 backward，否则并不释放中间激活；也不能逐 microbatch 调用原生
MPOptimizer.step，因为该函数会 zero_grad、backward 和 step。
权重衰减、momentum、梯度裁剪及分布式聚合均应每个逻辑 batch 执行一次。
Actor 的随机动作 mask、判别器当前/replay/demo 的均值权重及正则项也需保持
一致。应由用户运行 CPU 梯度/更新等价验证，再进行 GPU 小规模检查。

## 已有训练记录说明了什么

工作区可见的最新独立训练目录为：
`logs/mimickit/smpl-amp/20261019-191801-878449`。

保存的 params 显示：16384 环境、纯 AMP、`symmetry_weight=0.1`，动作列表是
`official_smpl/locomotion/motions-run6.yaml`，不是重建后的 AMASS。
如果用户观察的是工作区尚不可见的其他运行，需要根据那个目录重新判断。

该运行最后 100 条训练记录的平均值：

| 指标 | 平均值 |
| --- | ---: |
| Disc_Agent_Acc | 0.993545 |
| Disc_Demo_Acc | 0.999999 |
| Disc_Agent_Logit | -1.46915 |
| Disc_Demo_Logit | 1.45006 |
| Disc_Reward_Mean | 0.41050 |
| Clip_Frac | 0.21669 |
| Sym_Loss（未乘系数） | 0.003579 |

判别器在用于其更新的样本上仍然很容易区分策略和真实动作。它不是独立留出集
上的泛化准确率，也不是走路专属指标；不能由此证明过拟合、梯度消失或者
走路是唯一差的模式。但不能用 episode 接近 600 步证明风格已经学好。
纯 AMP 中环境的 task reward 为零，因此 Train_Return/Test_Return 为零也
不是 AMP 奖励丢失；风格奖励在构造 PPO 数据时加入。

## 数据路径和分布仍需区分

- 当前默认 `env.yaml` 仍指向 `locomotion-curated-v1/motions.yaml`，使用此前
  walk/run/jog/transition = 55/30/10/5 的人为分组概率。
- 可见最新训练实际使用更早的 run6 列表。不能把它的结果归因于新数据或
  当前尚未使用的 curated 默认。
- 新重建的 `data/mimickit/amass-pacer-v1/native/motions.yaml` 共 248 段、
  每段权重为 1。需要显式传入 `--motion_file` 才会使用。
- 新流水线修正了旧 KIT 100 FPS 数据用每 3 帧取样、却标成 30 FPS 的时间轴
  差异。旧 KIT 在播放时相当于原速的 90%。这个真实差异不是走路僵硬的
  单一因果解释。

修改动作采样同时改变参考状态初始化与判别器 demo 分布。暂不将新的成功率
采样、再次提高跑步权重和修正更新节奏混在同一实验中。

## 与上游一致及有意扩展的部分

| 部分 | 对照结果 |
| --- | --- |
| PPO、GAE、终止 bootstrap、网络及损失 | 调用上游，未在本接入重写 |
| 判别器正负标签、BCE、梯度惩罚、logit 正则 | 使用上游实现 |
| 风格奖励 | 使用上游 `2 * -log(max(1-sigmoid(logit), 1e-4))` |
| Demo/replay 放 CPU 与分块计算 | 源码未发现公式或样本对齐错误；非 bitwise 等价声明，未在本轮重跑测试 |
| 动作历史 | 原生 10 帧，控制 30 Hz，跨度 0.3 秒；采样从合法历史时间开始 |
| USD/XML、PD、摩擦、物理频率 | 官方资产；PD 沿用资产设置；控制 30 Hz / 物理 120 Hz，与上游配置一致 |
| 重置 | 保留原生关节状态与历史，补回 root 速度并提交 PhysX；已有运动学检查不能替代动态接触检查 |
| 局部观测 | 从上游 SMPL 示例 global_obs=true 改为 false；有意提供朝向不变的观测，不是同一实验配置 |
| 任务输入 | 额外 20 维；纯 AMP 时为零，不提供步态或目标速度指令 |
| sym_loss | 上游无此项；本接入增加 actor 镜像等变正则，系数 0.1 |

镜像使用极向量 `(x,-y,z)`、轴向量/旋转向量 `(-x,y,-z)`，并交换左右关节。
先在物理观测坐标中镜像，再分别做 RMS；动作先反归一化，镜像后重新归一化。
源码未发现明显符号/归一化顺序错误。它约束策略在镜像状态上的输出，不是
要求同一时刻左右关节角相等。不能因走路僵硬就直接增加系数；即使标量 loss
小，也不能单凭它的绝对值排除梯度影响。用户要求的始终开启保持不变。

## 下一步实验顺序

1. 先修正逻辑 batch/microbatch 的更新节奏，保留 CPU demo/replay 节省显存。
   固定原有数据和其他设置，可单独衡量该修正的影响。
2. 单独比较重建后的均匀动作列表。若直接用新数据从头建立整合基线，也应
   明确它同时改变数据和更新节奏，不能据此给单项改动归因。
3. 给走路初态、跑步初态分别评估，并记录实际速度分布、足部滑动/接触、
   膝关节活动范围及风格分数；初态类别不保证后续策略保持相同步态。
   原来的整体判别器准确率、存活时间不能定位走路质量。
4. 若走路单独数据的受控实验仍然僵硬，再定位物理跟随误差、PD/接触和
   对称正则梯度。暂不增加平滑惩罚或继续提高 sym_loss，避免压低动作幅度。

静态检查不保证步态改善；最终结论需要用户运行修正前后的受控训练和动态检查。
