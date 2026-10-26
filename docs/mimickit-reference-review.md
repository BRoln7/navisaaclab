# 分层检查所有参考动作

目标是先检查数据在当前 MimicKit 官方 SMPL USD 上的表现，不加载策略，
不构造判别器或优化损失。自动指标用于排查顺序，不能自动判定动作是否自然。

## 第一步：全部动作做 CPU 统计

```bash
conda activate navisaaclab
python scripts/mimickit_amp/review_motions.py
```

默认检查重新处理的 `data/mimickit/amass-pacer-v1/native/motions.yaml`。
每一帧参与指标计算，不修改、筛掉或重新加权原数据。
`--motion_file` 可切换到其他具有配套转换 manifest 的原生动作列表。

输出到 `logs/motion-review/amass-pacer-v1/`：

- `metrics.csv`：全部片段的名称、速度、根加速度、关节/根旋转变化、足底高度、
  低足位移速度代理值、躯干倾角、前向一致性及极值发生时间。
- `review_plan.json`：代表片段、极值片段及其原始编号/文件/检查时间窗口。
- `summary.json`：分类段数、需看的批次数及 CUDA 未初始化的记录。

关节旋转变化按四元数的实际夹角计算，避免把 expmap 的正负 π 分支误认为跳变。
足底高度来自官方 XML 的碰撞盒。低足位移速度只在足部低于 6 cm、垂直变化
较小且身体移动时统计；滚动接触和短片段仍可能产生较大值，不能等同真实滑脚。
跑步腾空不是异常。特殊侧步、单腿走、猫步等应结合源动作名称解释。

## 第二步：四格回放代表片段

```bash
./isaaclab.sh -p scripts/mimickit_amp/view_motions.py \
  --headless --video --device cuda:0
```

默认最多 20 个不同片段：慢走/正常走/快走各两段、四种速度的跑步、两段慢跑、
两段完整走跑转换，以及六种指标的待复查极值。某类不足时会少选，不伪造或重复补齐。
同一片段只占一个名额，其余类别不足不会自动证明数据缺失。

复用官方 `ViewMotionEnv` 的插值、关节映射及状态赋值，使用现有 SMPL 后端。
每帧提交根/关节状态并渲染，不推进物理时间，不调用策略。
四个环境相互分开，避免别的动作进入镜头。运行中也检查 USD 的实际刚体位置
与参考 FK 的误差，位置超过 5 cm 或旋转超过 0.15 rad 会报错，防止用错模型映射继续观察。

普通片段默认最多看 8 秒，极值片段会把检查窗口放在对应事件附近。
两段明确标注的 walk-to-run/run-to-walk 保留完整时间区间。
没有把独立走路/跑步片段拼接成转换；也没有循环短跑片段。
短片段结束后保持末帧，并在对应格子标记 `END (held)`。
批次切换时的姿态跳变和 END 后的静止属于播放器行为，不是连续动作缺陷。

输出到终端打印的 `logs/motion-review/video-时间戳/`：

- `reference-motions.webm`：四格参考动作视频。
- `batches.json`：视频帧范围、每格完整动作名称、源编号、选择原因和检查时间。
- `playback.json`：实际帧数、批次数及最大 USD/FK 位置误差。

观察重点：脚是否异常拖地/悬空，膝与躯干姿态是否原本就怪，左右步态是否明显
不对称，完整转换是否在源数据内已发生不连续。参考回放正常只证明参考与映射
这一层通过检查，不能证明 PD/接触动力学或策略训练正确。

## 定位问题后重看

只回放代表套件中的第 3 批（编号从 0 开始）：

```bash
./isaaclab.sh -p scripts/mimickit_amp/view_motions.py \
  --headless --video --device cuda:0 --batch 2
```

所有 248 段各检查一个窗口，自动分成 62 批：

```bash
./isaaclab.sh -p scripts/mimickit_amp/view_motions.py \
  --headless --video --device cuda:0 --suite all
```

查看所有片段的完整时长，或只对可疑批次添加这个选项：

```bash
./isaaclab.sh -p scripts/mimickit_amp/view_motions.py \
  --headless --video --device cuda:0 --suite all --batch 12 --full_clips
```

这些命令由用户执行。本轮没有启动 GPU 回放、训练或 pytest。
新增 CPU 回归测试可手动运行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest tests/test_motion_review.py -q
```
