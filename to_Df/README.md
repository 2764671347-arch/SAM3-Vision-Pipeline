# SAM3 最小运行环境

这个目录只保留了运行以下脚本所需的最小内容：

- `inference.py`
- `eval.py`
- `count_model_params.py`
- `sam3/`
- `assets/bpe_simple_vocab_16e6.txt.gz`

不包含训练、可视化示例、数据集脚本、完整仓库文档，也不包含模型权重。

## 目录结构

```text
to_Df/
├── assets/
│   └── bpe_simple_vocab_16e6.txt.gz
├── model/
│   └── sam3.pt            # 需要自行放置，或运行时通过参数指定
├── sam3/
├── count_model_params.py
├── eval.py
├── inference.py
├── requirements.txt
└── README.md
```

## 安装

```bash
cd /data/liupengli/WorkSpace/SAM3-demo/to_Df
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

如果你的权重是 `.safetensors`，再执行：

```bash
pip install safetensors
```

## 权重放置

默认会读取：

```text
./model/sam3.pt
```

你也可以在运行时显式指定：

```bash
python inference.py --checkpoint-path /path/to/sam3.pt ...
python count_model_params.py --model-path /path/to/sam3.pt
```

## 推理

输入数据目录要求：

```text
data_dir/
├── images/
└── labels/
```

其中 `labels/*.txt` 使用 YOLO 格式：

```text
class_id cx cy w h
```

示例：

```bash
python inference.py \
  --data-dir /path/to/data \
  --output-dir /path/to/output \
  --checkpoint-path /path/to/sam3.pt \
  --device cuda \
  --class-names "0:person,1:car"
```

如果不传 `--device`，脚本会自动在 `cuda` 和 `cpu` 之间选择。实际推理强烈建议使用 GPU。

## 评估

```bash
python eval.py \
  --gt gt_master.json \
  --predictions predictions.json \
  --output result.json
```

## 参数统计

统计本地 `sam3.pt`：

```bash
python count_model_params.py
```

显式指定路径：

```bash
python count_model_params.py \
  --model-path /path/to/sam3.pt \
  --json-output ./model_stats.json
```

如果只想按 checkpoint 直接统计，不尝试实例化本地 SAM3：

```bash
python count_model_params.py \
  --model-path /path/to/other_checkpoint.pt \
  --mode checkpoint
```

## 这份最小环境里额外做过的兼容性处理

- 去掉了原脚本里的绝对路径硬编码
- `count_model_params.py` 改成了命令行参数形式
- 修复了 `PositionEmbeddingSine` 在无 GPU 环境下无法初始化的问题
- 切断了推理路径对 `decord`、视频推理模块等非必要依赖的强制导入
