# MimicKit 官方 SMPL AMP 对照实验

这个实验直接运行固定版本 MimicKit 的官方 `AMPEnv`、`AMPAgent`、SMPL
配置和官方 `smpl_walk.pkl`，用于隔离本项目扩展造成的差异。单轮样本不超过
官方 replay 容量时完全使用原版 agent；超过容量时仅启用本文末尾说明的分块
环形写入兼容层。

固定版本为 `fbe1a36e878f036d6bf5521defb8eb3b8ab4170d`。运行入口会验证 checkout
版本且不允许已修改的受跟踪文件。`run_official.py` 只负责切换到 MimicKit
仓库根目录并调用其原始 `mimickit/run.py`；训练环境和 agent 不使用
`socnav_sim.mimickit_amp` 的子类。

## 实验中没有使用的本项目功能

- 没有 `sym_loss`。
- 没有轨迹观测、task reward、power cost 或轨迹终止。
- 没有额外 20 维策略输入。
- 没有 CPU demo/replay staging、梯度累积或自定义日志器。
- 没有 AMASS/PACER 重建数据、分类权重或成功率采样。
- 没有自定义 Isaac Lab engine 子类。

使用的是 MimicKit 官方默认：单条走路动作、4096 环境、10 秒回合、
`global_obs: true`、10 帧 AMP 历史、30 Hz 控制、120 Hz 物理，以及官方
网络、优化器、奖励和 replay 设置。官方参数文件原本选择 Isaac Gym，命令中
显式换成仓库提供的 Isaac Lab engine，其他 SMPL env/agent 设置不变。

## 安装官方动作

资源文件不随本仓库提交。请从 MimicKit README 的官方 data 链接下载
`MimicKit_Data.zip`，然后执行：

```bash
conda activate navisaaclab
python scripts/mimickit_amp/install_official_smpl_baseline.py \
  data/mimickit/downloads/MimicKit_Data.zip
```

安装器只提取官方 `data/motions/smpl/smpl_walk.pkl`，验证 ZIP CRC，记录文件
SHA-256，且不会覆盖内容不同的已有文件。

walk+run 实验还需要安装同一数据包中的官方跑步动作：

```bash
conda activate navisaaclab
python scripts/mimickit_amp/install_official_smpl_baseline.py \
  data/mimickit/downloads/MimicKit_Data.zip \
  --motion-name smpl_run.pkl
```

当前验证的数据包大小为 `402403772` 字节，SHA-256 为
`a8cd9f1c3451e4bc694a0a16eb1d2639d24cce0314c218771f42bf6e16e2b49f`；其中
`smpl_walk.pkl` 的 SHA-256 为
`a0d65fce7798ac21db3e67061a8005323e9c41fd6d869156cd763bba4b1822ec`，
`smpl_run.pkl` 的 SHA-256 为
`597953d6cfb6747c6cf2d548e7c9f796b7c443e45663b7282fe3624863ff3227`。

## walk+run 双动作实验

`data/cfg/mimickit_official/smpl_walk_run.yaml` 以相同权重列出官方 walk 和
run。环境配置复制官方 `amp_smpl_env.yaml`，只把 `motion_file` 换成上述双动作
清单；网络、优化器、AMP 奖励、回合长度和终止条件均保持官方设置。

建议从头训练，以免 walk-only checkpoint 让新实验在起点就偏向走路。下面按
官方 4096 环境运行 2000 轮；1000 轮时已有中间 checkpoint，可先推理判断：

```bash
RUN_DIR="$PWD/logs/mimickit-official/smpl-walk-run-$(date +%Y%m%d-%H%M%S)"
./isaaclab.sh -p scripts/mimickit_amp/run_official.py \
  --arg_file args/amp_smpl_args.txt \
  --mode train --num_envs 4096 --devices cuda:0 --visualize false \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config "$PWD/data/cfg/mimickit_official/amp_smpl_walk_run_env.yaml" \
  --agent_config data/agents/amp_smpl_agent.yaml \
  --logger tb --save_int_models true --rand_seed 0 \
  --max_iterations 2000 \
  --out_dir "$RUN_DIR"
```

这是无任务、无 gait command 的纯 AMP。策略可从当前身体速度和姿态延续走或
跑，但没有一个显式输入要求它在指定时刻切换步态。因此该实验用于验证同一
策略能否覆盖两种运动分布，不保证推理时可控地选择或切换 walk/run。

## 先做官方小规模检查

命令使用原版入口，因此输出频率、checkpoint 命名和日志格式也保持原版。
包装入口支持 `max_iterations`，并根据环境数量和 `steps_per_iter` 自动换算
MimicKit 使用的总环境步数。下面运行 10 个迭代：

```bash
./isaaclab.sh -p scripts/mimickit_amp/run_official.py \
  --arg_file args/amp_smpl_args.txt \
  --mode train --num_envs 64 --devices cuda:0 --visualize false \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config data/envs/amp_smpl_env.yaml \
  --agent_config data/agents/amp_smpl_agent.yaml \
  --logger tb --save_int_models true --rand_seed 0 \
  --max_iterations 10 \
  --out_dir "$PWD/logs/mimickit-official/smpl-walk-smoke"
```

如果目录已经包含输出，请换一个新目录；原版入口可能继续写入已有目录。

## 官方基线训练

先按官方默认 4096 环境训练。下面运行 1000 个迭代，只是首个评估预算，
不是收敛保证：

```bash
./isaaclab.sh -p scripts/mimickit_amp/run_official.py \
  --arg_file args/amp_smpl_args.txt \
  --mode train --devices cuda:0 --visualize false \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config data/envs/amp_smpl_env.yaml \
  --agent_config data/agents/amp_smpl_agent.yaml \
  --logger tb --save_int_models true --rand_seed 0 \
  --max_iterations 1000 \
  --out_dir "$PWD/logs/mimickit-official/smpl-walk-1000"
```

TensorBoard：

```bash
tensorboard --logdir logs/mimickit-official --port 6006 --samples_per_plugin scalars=999999
```

## 推理判断

原版测试模式会打开窗口，先用 4 个环境检查最终模型：

```bash
OFFICIAL_MODEL="$PWD/logs/mimickit-official/smpl-walk-1000/model.pt"
./isaaclab.sh -p scripts/mimickit_amp/run_official.py \
  --mode test --num_envs 4 --devices cuda:0 --visualize true \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config data/envs/amp_smpl_env.yaml \
  --agent_config data/agents/amp_smpl_agent.yaml \
  --model_file "$OFFICIAL_MODEL" \
  --test_episodes 32 --rand_seed 0
```

无窗口录制一个完整 episode 并直接导出 WebM：

```bash
./isaaclab.sh -p scripts/mimickit_amp/run_official.py \
  --arg_file args/amp_smpl_args.txt \
  --mode test --num_envs 4 --devices cuda:0 --visualize false \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --model_file "$PWD/logs/mimickit-official/smpl-walk-1000/model.pt" \
  --test_episodes 4 --rand_seed 0 \
  --video_out "$PWD/logs/mimickit-official/smpl-walk-1000/videos/preview.webm"
```

`video_out` 会自动启用官方 headless recorder，并在测试结束后保存帧。官方
Isaac Lab recorder 跟拍第 0 个环境；其他环境仍参与测试统计，但不会拼接进画面。

建议全部从仓库根目录执行。运行入口会自动切换到 MimicKit 根目录，因此
`model_file` 和 `out_dir` 应使用绝对路径；上面的 `$PWD` 在启动前完成展开。

这个实验只验证官方单动作 walking 基线：

- 若官方走路自然，说明 MimicKit AMP、官方 SMPL 资产和当前 Isaac Lab 后端
  至少能完成单动作走路。随后应逐项加入新数据、对称损失、CPU staging 和
  多动作训练，定位是哪一步使走路变僵。
- 若官方从头训练也僵硬，先比较官方预训练模型（如果数据包提供）并确认训练
  预算，再排查 Isaac Sim/Isaac Lab 版本和动力学差异。不能立刻断言 AMP
  算法本身错误。
- 即使官方单动作自然，也不能证明多动作 AMP 必然自然；单动作判别问题更简单，
  且该对照不包含跑步与走跑切换。

## 可变环境数量兼容

包装入口根据 `num_envs * steps_per_iter` 自动判断单轮 replay 写入量。官方
replay buffer 容量固定为 200000；当单轮样本超过容量时，入口将原版随机
replay 样本拆成不超过 `min(4096, buffer容量)` 条的多次环形写入。因此 8192、
12288、16384 或其他环境数使用同一入口，无需修改 agent YAML。replay 容量、
采样数量、网络、loss 和优化参数保持不变；当单轮样本不超过容量时完全使用
原版插入函数。可用环境上限仍由物理仿真、rollout/demo 张量和 GPU 显存决定。
