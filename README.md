# ResNet 退化问题复现（CIFAR-10）

复现论文 **Deep Residual Learning for Image Recognition**（He et al., CVPR 2016）
中的核心实验：**plain 深层网络随深度增加反而变差（退化问题），而残差网络可以随深度稳定获益。**

**Team group 11**：Chen Letian 2330026013 · Zhang Yiyang 2430024331 ·
Liu Chuanliang 2330013021 · Liu Yilin 2430026139

## 交付物

| 文件 | 内容 |
|---|---|
| `milestone1/reading.pdf` | 里程碑 1：阅读（7 页） |
| `milestone2/milestone2.pdf` | 里程碑 2：首次运行（6 页） |
| `figures/fig6_plain.png`、`figures/fig6_resnet.png` | 论文 Fig. 6 的复现对比 |
| `figures/fig7.png`、`figures/gradients.png` | 论文 Fig. 7 响应幅度、Sec. 4.1 梯度范数 |
| `figures/extension_a_lr.png` | 扩展 A：误差随初始学习率的变化 |
| `reports/summary.md` | 12 次核心复现的种子均值与极差 |
| `reports/extension_a_status.md` | 扩展 A 完整分析（H1/H2 判定） |
| `reports/extension_a_tuning.md` | 扩展 A 网格与选优 |

## 主要结果

64,000 迭代处的测试误差（核心复现，三个种子均值）：

| 网络 | 本项目 | 论文 Table 6 |
|---|---|---|
| plain-20 | 9.87% | 9.7% |
| plain-56 | 14.43% | 13.3% |
| ResNet-20 | 8.68% | 8.75% |
| ResNet-56 | 7.36% | 6.97% |

- **退化复现**：plain-56 的训练误差比 plain-20 高 2.04 个百分点（2.47 vs 0.43），
  超过三种子极差 0.70 —— 深层 plain 网络连训练集都拟合得更差，不是过拟合。
- **残差逆转**：ResNet-56 的测试误差比 ResNet-20 低 1.31 个百分点（7.36 vs 8.68），
  超过极差 0.34。
- **扩展 A（调参）**：plain-20 选出的初始学习率为 0.1，plain-56 为 0.03（H1 成立）；
  plain-56 在 0.3 和 0.5 上直接发散（迭代 23 / 14），plain-20 在同档位正常收敛。
  调优后退化依然超出种子波动（H2 成立）。

## 目录结构

```
src/
  models.py        # PlainCIFAR / ResNetCIFAR（BasicBlock，shortcut A/B/C）
  data.py          # 45k/5k 划分、逐像素均值、GPU 上 pad4+裁剪+翻转
  train.py         # 按迭代训练；每 epoch 与 64k 评估；AMP；发散检测
  plot.py          # Fig. 6 风格曲线：每种子细线 + 均值粗线
  plot_tuning.py   # 扩展 A：误差随初始学习率的变化
  analyze.py       # 逐层响应标准差（Fig. 7）、逐层梯度范数
  summarize.py     # 64k 训练/验证/测试误差、均值、极差与差值判定
  tune.py          # 扩展 A 选优表（按训练误差最低）
  epoch_table.py   # 单次运行的逐 epoch 表格
  milestone2_tex.py# 从 run 输出生成里程碑 2 的 LaTeX
scripts/
  run_core.sh        # 12 次核心复现（断点续跑队列）
  run_extension_a.sh # 扩展 A 网格（断点续跑队列）
  fetch_cifar.sh     # 断点续传 + MD5 校验的数据下载
runs/            # 每次运行：args.json / metrics.csv / results.json / last.pth
figures/         # 生成的图
reports/         # 汇总表格
data/            # CIFAR-10（首次运行自动下载）
```

## 环境

独立虚拟环境 `.venv`（CUDA 12.6 版 PyTorch，适配 RTX 4060）：

```bash
source .venv/Scripts/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 训练协议（任务书 + 论文）

| 项目 | 设置 |
|---|---|
| 数据划分 | 五个训练 batch 按顺序拼接，最后 5,000 张作验证集，前 45,000 张训练 |
| 归一化 | 在 **45,000** 张训练图上计算逐像素均值（不除标准差），在增强之后相减 |
| 增强 | 零填充 4，随机 32×32 裁剪，随机水平翻转；测试用原始单视图 |
| 优化 | SGD momentum 0.9，weight decay 1e-4，batch 128，He（fan-in）初始化 |
| 学习率 | 0.1，在 32k / 48k 迭代各 ÷10，64,000 迭代结束（110 层才用 warm-up） |
| 一个 epoch | 351 次迭代（45,000 // 128，丢掉最后一个不满的 batch），日程按迭代计 |
| 评估 | 每 epoch 结束及 64k：eval 模式下 45k / 5k / 10k 误差；训练模式下本 epoch 增强 batch 的 running 误差；64k 用最后 351 个 minibatch |
| 精度 | 每次运行都用 autocast float16 + GradScaler，评估用 float32 |
| 发散 | loss 出现 NaN/Inf 即停止并记录迭代数，不补跑、不计入均值 |

## 运行

```bash
# 单个运行（例：ResNet-20，种子 0）
python -m src.train --arch resnet --depth 20 --seed 0 --out runs/resnet20-s0

# 12 次核心复现 + 出图 + 汇总（断点续跑，已完成的不重跑）
bash scripts/run_core.sh

# 扩展 A：五档初始学习率 × 两个深度，seed 0
bash scripts/run_extension_a.sh
```

一次运行的产物：

- `metrics.csv`：每个 epoch 一行 —— 迭代数、lr、running 训练误差、45k 训练误差、5k 验证误差、10k 测试误差、epoch loss、累计耗时
- `results.json`：最终（64k）三项 eval 误差、running 误差、是否发散、参数量、总耗时
- `last.pth`：模型权重

> `src/train.py` 只在跑满 64k 时写 `last.pth` 和 `results.json`，中途中断的运行无法续跑，必须重来。
> 本机并发上限为 3 个进程：8 个并行时曾触发 `CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED`
> 和 `CUDA error: an illegal memory access` 各毁掉一个运行。56 层网络单进程已吃满 GPU，
> 并行两个只提升约 5% 总吞吐；20 层网络则能从并发中获益。

## 出图与汇总

```bash
python -m src.plot --network "plain-20:runs/plain20-s*" --network "plain-56:runs/plain56-s*" \
    --readoff "plain-20:1.5:9.7" --readoff "plain-56:5.9:13.3" \
    --title "plain networks (cf. Fig. 6 left)" --out figures/fig6_plain.png

python -m src.summarize runs/plain20-s* runs/plain56-s* runs/resnet20-s* runs/resnet56-s* \
    --out reports/summary.md

# 论文 Fig. 7：逐层响应幅度；Sec. 4.1：逐层梯度范数
python -m src.analyze responses --runs runs/plain20-s0 runs/plain56-s0 runs/resnet20-s0 runs/resnet56-s0 \
    --out figures/fig7.png
python -m src.analyze gradients --runs runs/plain20-s0 runs/plain56-s0 --out figures/gradients.png

# 扩展 A：误差随初始学习率的曲线、选优表
python -m src.plot_tuning --out figures/extension_a_lr.png
python -m src.tune runs/plain20-lr*-s0 runs/plain20-s0 runs/plain56-lr*-s0 runs/plain56-s0 \
    --out reports/extension_a_tuning.md
```

## 与论文的对应关系

| 论文 | 本项目 |
|---|---|
| Fig. 6 左：plain-20 vs plain-56 | `runs/plain20-s*` vs `runs/plain56-s*` |
| Fig. 6 中：ResNet-20 vs ResNet-56 | `runs/resnet20-s*` vs `runs/resnet56-s*` |
| Table 6：ResNet-20/56 测试误差 8.75 / 6.97 | `reports/summary.md` |
| Fig. 7：逐层响应标准差 | `src/analyze.py responses` |
| Sec. 4.1：梯度范数健康（非梯度消失） | `src/analyze.py gradients` |
| Table 3：shortcut A/B/C | `--shortcut A|B|C` |

## 与论文的已知差异

1. 训练集为 45,000（论文未说明是否用全部 50,000），预期误差略高于论文。
2. 每 epoch 的评估在 eval 模式下用未增强图像，论文未说明其 Fig. 6 训练误差的测法。
3. 混合精度（float16 autocast）用于训练，论文未提及；任务书允许，且每一次运行都使用。
4. 单卡 batch 128，论文为“两张 GPU 上”的 batch 128（每卡 64）。

## 引用

- He, K., Zhang, X., Ren, S., Sun, J. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Krizhevsky, A. *Learning multiple layers of features from tiny images.* Technical report, 2009.
- Prince, S. J. D. *Understanding Deep Learning.* MIT Press, 2023. §11.1–11.4.
- 所有代码由本组编写；预训练权重、第三方数据集均未使用。
