---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': AiArtData
          '1': RealArt
  splits:
  - name: train
    num_bytes: 2619946661.04
    num_examples: 152710
  download_size: 1372383838
  dataset_size: 2619946661.04
---
# Dataset Card for "AI-Generated-vs-Real-Images-Datasets"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)