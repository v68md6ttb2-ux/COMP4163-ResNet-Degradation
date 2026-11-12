# ResNet 退化问题复现（CIFAR-10）

复现论文 **Deep Residual Learning for Image Recognition**（He et al., CVPR 2016）
中的核心实验：**plain 深层网络随深度增加反而变差（退化问题），而残差网络可以随深度稳定获益。**

- 论文原文：`1512.03385v1.pdf`
- 任务书：`handout.pdf`；选题卡：`resnet-degradation-card.pdf`
- 里程碑 1（阅读）：`milestone1/reading.pdf`
- 数据集：CIFAR-10（论文 Sec. 4.2）
- 网络：$6n+2$ 层，$n=3 \Rightarrow 20$ 层，$n=9 \Rightarrow 56$ 层
- 对比：plain-20 / plain-56 / ResNet-20 / ResNet-56，各 3 个种子（12 次运行）

## 目录结构

```
src/
  models.py      # PlainCIFAR / ResNetCIFAR（BasicBlock，shortcut A/B/C）
  data.py        # 45k/5k 划分、逐像素均值、GPU 上 pad4+裁剪+翻转
  train.py       # 按迭代训练；每 epoch 与 64k 评估；AMP；发散检测
  plot.py        # Fig. 6 风格曲线：每种子细线 + 均值粗线
  analyze.py     # 逐层响应标准差（Fig. 7）、逐层梯度范数
  summarize.py   # 按任务书格式汇总 64k 训练/验证/测试误差、均值、极差
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

## 训练

```bash
# 单个运行（例：ResNet-20，种子 0）
python -m src.train --arch resnet --depth 20 --seed 0 --out runs/resnet20-s0

# 全部 12 次核心运行 + 出图 + 汇总
bash scripts/run_core.sh
```

一次运行的产物：

- `metrics.csv`：每个 epoch 一行 —— 迭代数、lr、running 训练误差、45k 训练误差、5k 验证误差、10k 测试误差、epoch loss、累计耗时
- `results.json`：最终（64k）三项 eval 误差、running 误差、是否发散、参数量、总耗时
- `last.pth`：模型权重

## 出图与汇总

```bash
python -m src.plot --network "plain-20:runs/plain20-s*" --network "plain-56:runs/plain56-s*" \
    --readoff "plain-20:1.5:9.7" --readoff "plain-56:5.9:13.3" \
    --title "plain networks (cf. Fig. 6 left)" --out figures/fig6_plain.png

python -m src.summarize runs/plain20-s* runs/plain56-s* runs/resnet20-s* runs/resnet56-s* \
    --out reports/summary.md

# 论文 Fig. 7：逐层响应幅度；Sec. 4.1：逐层梯度范数
python -m src.analyze responses --runs runs/plain20-s0 runs/plain56-s0 runs/resnet20-s0 runs/resnet56-s0 \
    --out figures/fig7_responses.png
python -m src.analyze gradients --runs runs/plain20-s0 runs/plain56-s0 --out figures/gradients.png
```

## 与论文的对应关系

| 论文 | 本项目 |
|---|---|
| Fig. 6 左：plain-20 vs plain-56 | `runs/plain20-s*` vs `runs/plain56-s*` |
| Fig. 6 中：ResNet-20 vs ResNet-56 | `runs/resnet20-s*` vs `runs/resnet56-s*` |
| Table 6：ResNet-20/56 测试误差 8.75 / 6.97 | 修改 `--depth` 即可 |
| Fig. 7：逐层响应标准差 | `src/analyze.py responses` |
| Sec. 4.1：梯度范数健康（非梯度消失） | `src/analyze.py gradients` |
| Table 3：shortcut A/B/C | `--shortcut A|B|C` |

## 与论文的已知差异

1. 训练集为 45,000（论文很可能用全部 50,000），预期误差略高于论文。
2. 每 epoch 的评估在 eval 模式下用未增强图像，论文未说明其 Fig. 6 训练误差的测法。
3. 混合精度（float16 autocast）用于训练，论文未提及。

## 预期结果（论文 Table 6 / Fig. 6）

- plain-20 → plain-56：**训练误差和测试误差都上升**（退化问题，非过拟合）。
- ResNet-20 → ResNet-56：**两者都下降**；ResNet-56 单模型测试误差约 **6.97%**。
- ResNet-56 参数量 0.85M，与 plain-56 完全相同。
