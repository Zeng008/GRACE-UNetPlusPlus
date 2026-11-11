# GRACE-UNet++ 代码说明

本仓库仅整理本论文提出的模型，以及运行、训练和评估该模型所需的代码。

包括：GFM、CGRR、完整 GRACE-UNet++、ACDC/MSD 数据加载与预处理、两阶段训练、
原实验五折划分、评估与预测入口，以及论文中的去模块和细化策略消融。
不包含其他论文的对比模型、与本文无关的探索分支、原始医学数据或训练权重。

文件结构：

| 位置 | 内容 |
| --- | --- |
| `grace_unetpp/model.py` | GFM、CGRR 和完整网络 |
| `grace_unetpp/datasets/` | ACDC、MSD 数据处理 |
| `train.py` | 五折训练与自身消融 |
| `evaluate.py`、`predict.py` | 测试集评估与预处理张量预测 |
| `summarize.py` | 明确区分切片级和病例级汇总 |
| `configs/`、`splits/` | 运行参数与实际五折编号 |
| `tests/` | 数学算子、梯度、网络和数据流程检查 |

安装、数据格式和运行命令见英文 README。ACDC 使用 HDF5 切片，MSD Spleen 使用
NIfTI 体数据。训练参数和五折编号分别见 `configs/` 与 `splits/`，评估结果可按切片级
或病例级进行汇总。
