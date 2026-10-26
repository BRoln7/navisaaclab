# 从原始 AMASS 重新准备 MimicKit AMP 数据

建议重新建立一份可追溯的数据版本，先验证参考动作，再训练。数据决定 AMP
判别器要学习的运动分布，但控制器、观测、动作接口、初始化、训练过程同样
影响结果。目前不能断言“算法已经完全正确，只剩数据问题”。

## 当前数据的审计结果

2026-10-20 在 CPU 上读取原始项目 PKL，没有启动仿真、训练、推理或 CUDA。
结果在 `logs/motion-analysis/20261020/pacer-source-audit.json`。

- 当前文件包含 246 段动作。PACER 上游 `amass_run_data` 有 285 个条目、
  276 个唯一名称；项目中的同名清单与上游清单逐项相同。
- 246 段全部在该清单内。缺少的 30 段主要为上下楼梯，另有扶手、蹲走到跑步、
  侧走转身片段。这不构成“缺少 30 段必要普通走跑动作”的证据。
- 用保留的 `pose_aa`、关节父子关系和 PACER 的 upright 公式独立重建
  `pose_quat_global`，所有动作最大旋转误差约 `1.63e-15 rad`。
  保存的 local/global 四元数关系最大误差约 `8.45e-8 rad`。
  因此暂未发现这一步存在关节顺序或 upright 变换错误。
- 所有最终片段标记 30 FPS，但文件没有保存原始捕获帧率与重采样轨迹。
  用户确认原始 NPZ 和中间数据库没有保留，无法仅凭最终文件验证时间轴。
- 最终数据全部为 neutral、beta=0，与 PACER 转换脚本的做法一致。
  这不是损坏证据，但无法从这些字段恢复原始人体形状。
- 当前 MimicKit 转换在官方骨架上保持旋转，用每片段固定高度平移贴地。
  平移范围约 -9.23 至 +12.32 cm。这不是接触约束重定向，也不证明足底接触
  正确；往返 FK 数值一致同样不等于动作物理可行。

此前的跑步加权和连续走路筛选调整的是采样分布，不会修复原始时间轴、
骨架适配或接触问题。已有数据不应直接认定为错误，也不应直接认定为充分。

## 下载原始数据

官方入口：<https://amass.is.tue.mpg.de/download.php>。
需要在 AMASS 网站注册、登录并取得所选数据集的下载权限。
使用 **SMPL+H G / gender-specific**，与 PACER 原始读取链路对应；
不使用 SMPL-X，也不把第三方预处理数据当作原始 AMASS。

```bash
conda activate navisaaclab
python scripts/mimickit_amp/download_amass.py
```

脚本会在终端询问 AMASS 用户名和密码，密码隐藏输入，不写入命令参数、
下载 URL、日志或凭据文件。默认保留 TLS 证书校验。
密码仅用于官方 HTTPS 下载服务；脚本不会代替用户注册或接受网站许可。

默认下载 PACER 清单涉及的 8 个来源，而非整个 AMASS：

| 官方归档名 | 解压后的 PACER 目录名 |
| --- | --- |
| ACCAD | ACCAD |
| BMLrub | BioMotionLab_NTroje |
| EyesJapanDataset | Eyes_Japan_Dataset |
| HumanEva | HumanEva |
| KIT | KIT |
| SFU | SFU |
| TotalCapture | TotalCapture |
| Transitions | Transitions_mocap |

可以先下载一个子集确认账号和网络正常，再运行默认命令补齐：

```bash
python scripts/mimickit_amp/download_amass.py --datasets ACCAD
```

默认输出：

```text
data/amass_raw/pacer_smplh_g/
  archives/                 # 原始 tar.bz2、断点文件、归档 SHA256
  raw/ACCAD/                # 原始 NPZ；其余七个来源同级
  metadata/ACCAD.csv        # 逐片段路径、原始 FPS、帧数、维度、文件大小
  metadata/ACCAD.json       # 来源 URL、归档哈希、FPS 分布、导出时间
```

脚本只依赖标准库和 NumPy，不导入 Torch、Isaac Sim 或训练模块。
支持断点续传和重试；服务器不支持续传时重新下载当前归档。
HTTP 登录网页会被识别，避免误存为数据。下载完成后计算本地 SHA256；
这属于可追溯记录，不是 AMASS 发布的官方校验和。
解压读取 bzip2 校验尾部，拒绝路径越界和链接文件，保留原始归档。
逐片段只读取数组头与 FPS，不修改姿态、轨迹、帧率或关节数据。
请为原始归档和解压结果同时预留空间；脚本会按实际传输/成员大小检查剩余空间。

重复运行会复用已有归档，并跳过有匹配来源记录的已解压子集。
跳过已解压目录不代表重新逐字节审计其所有 NPZ；如需独立重建可指定新的输出目录。
手工从官网获得的同名归档也可放入 `archives/` 后使用：

```bash
python scripts/mimickit_amp/download_amass.py --extract-only
```

其他选项：`--output /path/to/amass`、`--no-extract`、`--plan`。
`--plan` 仅打印范围和官方下载 URL，不联网。
归档下载地址采用官方 `download.php` 接口；已对照现有公开下载实现，
但本次没有使用用户账号实际下载，不能保证账号权限或服务端接口永不变化。
遇到权限/登录错误，应在官网下载页确认该账号可下载对应 **SMPL+H G** 子集。

## 下载后如何重建

PACER 官方说明的处理顺序是：

1. [process_amass_raw.py](https://github.com/nv-tlabs/pacer/blob/main/uhc/data_process/process_amass_raw.py)：读取原始 NPZ。
2. [process_amass_db.py](https://github.com/nv-tlabs/pacer/blob/main/uhc/data_process/process_amass_db.py)：时间采样、异常片段筛选、高度处理。
3. [convert_amass_isaac.py](https://github.com/nv-tlabs/pacer/blob/main/uhc/data_process/convert_amass_isaac.py)：关节顺序、neutral 骨架和 upright 转换。

下一步需要把这一链路做成适配当前 Python/MimicKit 的可追溯处理流程，
而不是直接运行上游包含交互断点和历史路径的研究脚本。
尤其上游第二步使用 `int(source_fps / 30)` 取整数步长，第三步写死 30 FPS：
当源帧率不是 30 的整数倍时，这不能保证真实时间轴一致。
例如源为 100 FPS，隔 3 帧取样实际约为 33.33 FPS，标作 30 FPS 会改变动作速度。
这是上游处理方式的风险，尚不能据此证明现有 246 段受到了该问题影响。

新版本应保留原始 FPS、源文件哈希、时间对应关系，用一致的时间坐标对旋转
和位移重采样；不要单独改 demo 的 dt 来让动作“更快”。
接着以当前 **MimicKit 官方 SMPL USD/XML** 为目标检查腿长、根高度、
足部接触、滑脚、关节速度尖峰和左右步态。不能只比较四元数往返误差。

先选少量正常走路、持续跑步和完整走跑转换做参考动作回放，确认数据自身
视觉表现正常；再建立独立的 `locomotion-v2` 数据目录与固定采样清单。
同一组算法/奖励/对称损失参数做对照，才能区分重建数据带来的变化。
纯 AMP 没有速度指令，数据包含走跑不等于策略必须主动按需切换，仍需另行评估。

当前下载脚本不执行上述重采样、重定向、回放或训练，也不修改当前训练配置。
本次只检查了脚本语法和代码；实际下载与训练/测试命令由用户执行。

## 下载完成后的处理命令

已添加独立入口，调用 `third_party/pacer_preprocessing/` 中保留并注明来源的
PACER 处理函数。现有 `uhc/data_process/`、旧 PKL 和训练配置均保留。
适配内容和来源哈希见该目录的 `README.md`、`sources.json`。

```bash
conda activate navisaaclab
python scripts/mimickit_amp/process_amass.py --time_mode exact
```

执行流程：

1. 使用 PACER 原 `amass_run_data` 清单，提前读取 276 个独立原始动作，
   保存 `raw_selected.pkl`，保留捕获 FPS、全部 156 个姿态参数和原始位移。
2. `exact` 模式用真实时间网格重采样到 30 Hz：旋转用 SLERP，位移线性插值。
   60/120 Hz 来源直接按整数步长取样；100 Hz 来源按真实时间插值。
   不修改原始数据，时间对齐结果独立保存在 `time_aligned.pkl`。
3. 调用 PACER 的 `process_qpos_list` 和 `fix_height_smpl_vanilla`：
   按官方异常标注剔除/截断，去除手指自由度，用对应原始性别/形状的 SMPL
   首帧网格最低点进行整段高度平移，写出 `processed_smpl.pkl`。
4. 调用 PACER 的 `run`：neutral/beta=0、关节重排、root offset、upright
   旋转，写出 `pacer_upright.pkl`。复用已有固定 neutral PACER XML，避免
   每段重新生成并覆盖同一 XML；会检查该 XML 的骨盆偏移及骨架顺序。
5. 用现有的 MimicKit 官方骨架转换器写出 `native/motions.yaml` 和原生动作。
   USD 仍是当前 MimicKit 官方 SMPL，不使用 PACER XML 作为运行时人体。
   此步保持现有的每段固定高度修正和 FK 往返检查，不是接触约束重定向。

当前下载数据完整覆盖 276 个唯一名称。根据原始下载清单，213 段为 100 FPS、
60 段为 120 FPS、3 段为 60 FPS。这是筛选前参考动作池，不等于原训练时的
`slim` 子集，也不是严格的 AMASS train/test 划分；其中仍有楼梯等特殊动作。
本轮不重新加权走跑，所有导出的动作权重先设为 1。

**新增确认：** 将旧数据的 185 段 KIT 动作与重新下载的原始数据比较，全部
逐帧对应 `raw_100fps_poses[::3]`，但旧 PKL 标记 30 FPS。
因此这 185 段的播放速度确实为原始时间轴的 90%，不再只是代码层面的假设。
这不证明 10% 的时间差就是所有步态问题的原因。

官方 `download_data.sh` 提供 occlusion v2，而原版处理入口引用 v3。
新入口默认自动下载官方公开提供的 **v2**，并将实际文件 SHA256 写入构建记录。
如果手头有 v3，可用 `--occlusion /path/to/v3.pkl` 指定现有文件。
不会在缺少标注时默默改为空筛选表。下载失败时可从
[官方脚本](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/download_data.sh)
取得文件后使用 `--occlusion`。

默认输出到独立目录 `data/mimickit/amass-pacer-v1/`：

```text
build.json                  # 原始文件/模型/骨架/处理代码/标注的哈希及阶段状态
raw_selected.pkl
time_aligned.pkl
processed_smpl.pkl
pacer_upright.pkl
native/manifest.json         # 每段来源名称、FPS、固定高度平移与 FK 误差
native/motions.yaml
native/clip_*.pkl
clips.csv                   # 源 FPS/帧数、最终 FPS/时长、固定高度平移与 FK 误差
report.json                 # 筛选前后数量、被剔除名称、总时长和最终动作列表
```

中断后重新执行同一命令，会校验并复用完成阶段；输入或设置变化则需要新的
`--output`，避免混用缓存。原生转换在临时目录完成后发布，列表使用最终路径。

仅检查覆盖范围和原始 FPS（不会下载标注或生成数据）：

```bash
python scripts/mimickit_amp/process_amass.py --plan
```

原版时间轴对照模式（明确保留整数抽帧和固定 30 FPS 标签）：

```bash
python scripts/mimickit_amp/process_amass.py \
  --time_mode pacer_legacy \
  --output data/mimickit/amass-pacer-legacy-v1
```

该模式仍使用所选标注版本及当前骨架适配，不宣称逐字节复现原版训练数据。
不建议把原版时间轴模式作为修正时间轴后的默认训练集。

本轮只做静态检查与原始/旧数据只读比较；没有运行处理入口、单元测试、仿真
或训练。可手动运行本次新增的 CPU 测试：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest tests/test_amass_processing.py -q
```

处理完成后先检查 `report.json` 和参考动作。当前 276 段候选含有楼梯和特殊动作，
且 v2/v3 的筛选结果可能不同，不能未经检查就自动替换现有普通平地走跑训练集。
