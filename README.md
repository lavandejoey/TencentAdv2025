# TencentAdv2025

## **项目结构**
```text
TencentAdv2025/
│
├─ environment.yml           # conda 环境配置
├─ README.md                 # 项目说明
│
├─ src/                      # 源码
│   ├─ train.py              # 训练主脚本
│   ├─ infer.py              # 推理脚本（ln to EVAL_INFER_PATH）
│   ├─ model.py              # 模型定义
│   ├─ dataset.py            # 数据加载与预处理
│   └─ utils.py              # 通用工具
│
├─ config/                   # 配置文件（超参、路径等）
│   └─ default.yaml  
│
├─ data/                     # 训练数据挂载点（ln to TRAIN_DATA_PATH）
│   └─ …  
│
├─ checkpoints/              # 训练模型（ln to TRAIN_CKPT_PATH）
│   └─ global_step_10000/    # ⭐ 必须以 global_step 开头
│       ├─ ckpt.pt
│       └─ …
│
└─ logs/                     # TensorBoard 日志（ln to TRAIN_TF_EVENTS_PATH）
    └─ events.out.tfevents.…
```

## **Angel 平台使用简要笔记**

  ---
  - 环境变量配置

    | 环境变量 | 描述 |
    | --- | --- |
    | USER_CACHE_PATH | 用户缓存路径（20 GB）；训练/评测均提供，可用于任务间共享文件 |
    | TRAIN_DATA_PATH | 训练数据集根目录 |
    | TRAIN_CKPT_PATH | 训练产出模型检查点根目录，检查点必须放在以 `global_step` 开头的子目录下，平台才能识别并用于评测 |
    | TRAIN_TF_EVENTS_PATH | TensorBoard Events 日志输出路径，用于训练可视化 |
    
  - 评测
  
    | 环境变量 | 描述 |
    | --- | --- |
    | MODEL_OUTPUT_PATH | 已训练模型（.ckpt）根目录 |
    | EVAL_DATA_PATH | 评测数据及检索索引根目录 |
    | EVAL_INFER_PATH | 用户上传的推理脚本所在目录，脚本须命名为 `infer.py` |
    | EVAL_RESULT_PATH | 评测输出目录，最终结果保存在 `result.json` |
  
  ---

  - 训练与评测规范
    - **训练容器内**，通过以上环境变量挂载数据与输出目录。
    - **模型检查点**：必须输出到 `TRAIN_CKPT_PATH` 并位于形如 `global_step=xxx` 的子目录内。
    - **训练可视化**：若需在平台界面查看 Loss 等指标，需将 TensorBoard Events 写入 `TRAIN_TF_EVENTS_PATH`。
    - **评测脚本**：统一命名 `infer.py`，提交后平台状态由“待评测”→“评测中/失败/成功”，通过后可在界面下载评测结果。

  ---

  - 提交与排名规则
    - **提交频次**：每日（北京时间 12 点—次日 12 点）最多 3 次。
    - **排行更新**：使用北京时间 12 点前提交的历史最优成绩，于当日 15 点在榜单更新。

  ---

  - 环境

    | **GPU**：20% 单 GPU 算力，约 19 GiB； | **OS**：Ubuntu 22.04；                                    |
    |:-------------------------------|:--------------------------------------------------------|
    | **CPU**：9 核；                   | **NVIDIA 驱动**：535.247.01；**CUDA**：12.3；**cuDNN**：9.5.1； |
    | **内存**：55 GiB。                 | **cuBLAS**：12.3.4.1-1；**NCCL**：2.20.3-1+cuda12.3；       |
    |                                | **Conda**：25.3.1；**Python**：3.10.16。                    |