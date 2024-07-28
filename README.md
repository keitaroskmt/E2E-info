# E2E-info
Supplementary codes for our paper ["End-to-End Training Induces Information Bottleneck through Layer-Role Differentiation: A Comparative Analysis with Layer-wise Training"](https://openreview.net/forum?id=O3wmRh2SfT&noteId=3X3EJITPUQ), TMLR2024.

It provides useful implementations for **various layer-wise training methods** and **HSIC-based analyses**.

## Getting Started
### Layer-wise Training
Under work

### Forward-Forward Algorithm
Train models with Forward-Forward algorithm by [Hinton, 2022](https://arxiv.org/abs/2212.13345) with
```
Python main_ff.py
```

Currently, toy MLP and CNN models are supported.
You can try new models by adding new classes under `src/models/forward_forward_model.py`.

Embedding label information into the inputs is one of the characteristics of the Forward-Forward algorithm, and it is supported in several ways in addition to the original paper.
For more details, please refer to the descriptions in the `LabelEmbedder` class under `src/models/forward_forward_block.py`.

For example, settings `method=top-left` embeds class information as in the original paper like

<a href="url"><img src="https://github.com/keitaroskmt/E2E-info/blob/1c62cb9223ee76d020eed0235491cf9c3419d071/images/ff_top_left.png" width=400></a>

We can also provide class information by subtracting the class prototypes as follows.

<a href="url"><img src="https://github.com/keitaroskmt/E2E-info/blob/1c62cb9223ee76d020eed0235491cf9c3419d071/images/ff_subtract.png" width=400></a>


## TODO List
#### Overall
- [ ] Add documents to run main files
- [ ] Test on GPU machine
- [ ] Add codes to reproduce nHSIC dynamics

#### Training Algorithms
- [x] Layer-wise training
- [x] Sequential layer-wise training
- [ ] Signal Propagation
- [x] Forward-Forward algorithm

#### Architecture
- [x] ResNet
- [x] VGG
- [ ] Vision transformer
