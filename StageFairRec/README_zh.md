# StageFairRec 中文使用说明

这是依据所提供论文重建的代码库，包含主要方法的完整训练与评测链路。
**它不是从作者实验服务器导出的原始代码，也尚未复现论文表格数值。**

## 已包含

源端 LoRA 公平语义学习、用户/课程门控融合、融合后敏感残差分解、TLGC；
PMF、DeepModel、SASRec、BERT4Rec；六种配置（纯骨干及五种增强变体）；
固定负采样评测、HR/NDCG、群体差异、四类攻击器、检查点与断点续训。

SM、FFVAE、AFRL 不在此仓库中，避免把简化重写误当成这些论文的原方法。
真实数据、7B 权重、作者原始结果与检查点均未附带。

## 最快运行

在仓库根目录创建 Python 3.10+ 环境，并按机器安装 PyTorch，然后运行：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/smoke.py --out runs/smoke
```

演示自动生成合成数据并运行四种骨干及所有增强变体。演示语义编码器是轻量
测试替身，不代表真正的大模型。GPU 大模型实验另安装 `.[llm,attack,dev]`。
完整命令见英文 [README](README.md)。

## 真实数据准备

需提供三个 UTF-8 CSV：

- `courses.csv`：`course_id` 与课程文本字段。
- `users.csv`：`user_id,gender,age`。gender 必须事先明确转换成 0/1。
- `interactions.csv`：`user_id,course_id,timestamp`，时间为数值时间戳。

程序执行每用户 8:1:1 时间切分，剔除少于 10 次交互的用户；固定抽取 99 个
未交互课程作为评测负例。年龄分组边界必须显式传入，程序不会代替作者猜测。
详细说明和 JSONL 转换脚本见 [DATA.md](docs/DATA.md)。

## 论文与代码的对应

| 论文内容 | 文件 |
|---|---|
| 式 (2)–(6)、(18)：LLM 编码与源端公平 | `semantics.py`、`models.py` |
| 式 (7)–(10)：门控融合、敏感残差相减 | `models.py` |
| 式 (11)–(13)：融合后对抗目标 | `engine.py` |
| 式 (14)–(16)：TLGC | `losses.py` |
| 式 (17)、(19)：推荐与联合优化 | `models.py`、`engine.py` |
| 式 (20)：群体推荐质量差值 | `engine.py` |
| 属性恢复攻击 | `attacks.py` |

所有实现文件位于 `stagefairrec/`。论文没有交代清楚的参数和结构选择集中列在
[IMPLEMENTATION.md](docs/IMPLEMENTATION.md)，包括年龄分箱、Norm 类型、
DeepModel 结构、各骨干损失适配、攻击数据的用户重叠问题等。

## 上传 GitHub

压缩包解压后，上传 **StageFairRec 文件夹内的内容** 到仓库根目录。
保留 `.github`、`.gitignore` 等文件，不上传合成运行目录、真实用户数据和模型缓存。
MIT 许可证已包含。正式作为作者开源代码发布前，请将引用信息补充为核实后的作者
与论文信息，并用实际数据核对结果；当前 README 如实标注了“依据论文重建”。
