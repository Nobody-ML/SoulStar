# SoulStar - 知心心理咨询大模型
<div align="center">
  <img src="./assets/logo.png" width="600"/>
  <!-- <a href="https://github.com/Nobody-ML/SoulStar/tree/main/">
    <img src="assets/logo.png" alt="Logo" width="600">
  </a> -->

  <!-- [![Logo](assets/logo.png)](https://github.com/Nobody-ML/SoulStar/tree/main/) -->

  <h3 align="center">SoulStar</h3>
  <br /><br />

  [![license](https://img.shields.io/github/license/Nobody-ML/SoulStar.svg)](https://github.com/Nobody-ML/SoulStar/blob/main/LICENSE)

  <!-- 🔍 模型开源地址：
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤗%20Huggingface)]()
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤖%20ModelScope)]() -->

</div>

## 📖 目录
- [SoulStar - 知心心理咨询大模型](#soulstar%20-%20知心心理咨询大模型)
  - [🎉 更新](#🎉%20更新)
  - [📝 简介](#📝-简介)
  - [🛠️ 快速开始](#🛠️-快速开始)
    - [1. 算力要求](#1-算力要求)
    - [2. 基于 transformers 使用模型](#2-基于-transformers-使用模型)
    - [3. 通过网页前端体验 demo](#3-通过网页前端体验-demo)
    - [4. 基于 LMDeploy 高性能部署](#4-基于-lmdeploy-高性能部署)
  - [🧾 数据构建](#🧾-数据构建)
  - [🧑‍💻 微调指南](#🧑‍💻-微调指南)
  - [📚 应用体验](#📚-应用体验)
  - [🎖️ 致谢](#🎖️-致谢)
  - [开源许可证](#开源许可证)



## 🎉 更新
- 【2024.2.26】基于 internlm2-chat-7b-qlora 微调出第一版模型


## 📝 简介

SoulStar 是一个心理咨询大模型，通过 `Large Language Model` 指令微调而来，内核为**温柔、知心的大姐姐**，她能分析用户的咨询内容和需求，给予详实的且具有人文关怀的长回复或详细建议，并且回复中附有 emoji 表情和颜文字(~(\*╹▽╹\*)、✿✿ヽ(ﾟ▽ﾟ)ノ✿ 、(\*^\_^\*))以拉近与用户的距离，让用户在咨询中得到心理上的支持和帮助。

目前支持模型及微调方式列表如下：
|         基座模型          |   微调方式   |
| :-------------------: | :------: |
|   InternLM2-Chat-7B   |  qlora   |
|          ……           |    ……    |

项目持续开发中，欢迎  Star⭐、PR 和 Issue。

## 🛠️ 快速开始

### 1. 算力要求
- 推理要求显存至少16G

### 2. 基于 transformers 使用模型
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("internlm2-chat-7b-soulstar", trust_remote_code=True)
# 设置`torch_dtype=torch.float16`来将模型精度指定为torch.float16，否则可能会因为您的硬件原因造成显存不足的问题。
model = AutoModelForCausalLM.from_pretrained("internlm2-chat-7b-soulstar", device_map="auto",trust_remote_code=True, torch_dtype=torch.float16)

model = model.eval()
response, history = model.chat(tokenizer, "请问你是谁呀？", history=[])
print(response)

response, history = model.chat(tokenizer, "我最近真的好焦虑，课业上给我的作业总是错的，考试时好时坏，我压力真的好大，父母也老是因为学习上的事打骂我，我是不是该放弃学习了？我也没什么朋友，我也想和别人一起玩，一起学习，但是我感觉总是开不了口，一直都是一个人，我该怎么办才好啊，感觉我的人生真的很糟糕，看不到什么希望。", history=history)
print(response)
```
### 3. 通过网页前端体验 demo
```bash
pip install streamlit
pip install transformers
streamlit run ./demo/web_demo.py --server.address 127.0.0.1 --server.port 6006
```

### 4. 基于 LMDeploy 高性能部署
```shell
# 使用命令行
lmdeploy chat turbomind /root/model/soulstar  --model-name internlm2-chat-7b
```

## 🧾 数据构建
- 详情请见[数据构建](./datasets/README.md)

## 🧑‍💻 微调指南
- 详情请见[微调指南](./finetune_config/xtuner_config/README.md)

## 📚 应用体验
- 应用部署在 OpenXLab 应用中心，可前往体验

## 🎖️ 致谢
- [OpenXLab](https://openxlab.org.cn/home)
- [InternLM](https://github.com/InternLM/InternLM/tree/main)

## 开源许可证

该项目采用 [Apache License 2.0 开源许可证](LICENSE)。同时，请遵守所使用的模型与数据集的许可证。
