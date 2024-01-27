# Textual Prompt Guided Image Restoration

Qiuhai Yan, Aiwen Jiang, Kang Chen, Long Peng, Qiaosi Yi and Chunjie Zhang, "Textual Prompt Guided Image Restoration", arXiv, 2023 

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2312.06162)

> **Abstract:** Image restoration has always been a cutting-edge topic in the academic and industrial fields of computer vision. Since degradation signals are often random and diverse, ”all-in-one” models that can do blind image restoration have been concerned in recent years. Early works require training specialized headers and tails to handle each degradation of concern, which are manually cumbersome. Recent works focus on learning visual prompts from data distribution to identify degradation type. However, the prompts employed in most of models are non-text, lacking sufficient emphasis on the importance of human-in-the-loop. In this paper, an effective textual prompt guided image restoration model has been proposed. In this model, task-specific BERT is fine-tuned to accurately understand user’s instructions and generating textual prompt guidance. Depth-wise multi-head transposed attentions and gated convolution modules are designed to bridge the gap between textual prompts and visual features. The proposed model has innovatively introduced semantic prompts into low-level visual domain. It highlights the potential to provide a natural, precise, and controllable way to perform image restoration tasks. Extensive experiments have been done on public denoising, dehazing and deraining datasets. The experiment results demonstrate that, compared with popular state-of-the-art methods, the proposed model can obtain much more superior performance, achieving accurate recognition and removal of degradation without increasing model’s complexity.

## Installation
The project is built with Python 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
~~~
pip install -r requirements.txt
~~~

## Results (TextPromptIR)
Performance results of the TextPromptIR framework trained under the all-in-one setting
<summary><strong>Table</strong> </summary>

<summary><strong>Visual Results</strong></summary>

## Data Download and Preparation
Denoising: [BSD400](https://drive.google.com/drive/folders/1O1Z8yEbLzndLzk9jK233r8DEI-3Xmeoe?usp=drive_link), [WED](https://drive.google.com/drive/folders/1p7ax2daKZOjHyMA7UFZ3lcoRBeWtTmxn?usp=drive_link), [Urban100](https://drive.google.com/drive/folders/1QgXBf3LOKwZnnWQQBqDt56T630mq_o7v?usp=drive_link), [CBSD68](https://drive.google.com/drive/folders/1mgEDilXcRkE6bkQoGkK-wrf-OhkC2CpI?usp=drive_link)

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1RjrjuGBK0jsQ5a5j1k-clsdxZkrqPQE2?usp=drive_link)

Dehazing: [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) (OTS)

The training data should be placed in ``` data/Train/{task_name}``` directory where ```task_name``` can be Denoise,Derain or Dehaze.
After placing the training data the directory structure would be as follows:
```
└───Train
    ├───Dehaze
    │   ├───original
    │   └───synthetic
    ├───Denoise
    └───Derain
        ├───gt
        └───rainy
```

The testing data should be placed in the ```test``` directory wherein each task has a seperate directory. The test directory after setup:

```
├───dehaze
│   ├───input
│   └───target
├───denoise
│   ├───bsd68
│   └───urban100
└───derain
    └───Rain100L
        ├───input
        └───target
```
## Citation
If you find this project useful for your research, please consider citing:
~~~
@article{yan2023textual,
  title={Textual Prompt Guided Image Restoration},
  author={Yan, Qiuhai and Jiang, Aiwen and Chen, Kang and Peng, Long and Yi, Qiaosi and Zhang, Chunjie},
  journal={arXiv preprint arXiv:2312.06162},
  year={2023}
}
~~~

## Contact
Should you have any question, please contact Qiuhai Yan (yanqiuhai16@163.com).
