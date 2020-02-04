# SAE
Code for ACL 2019 paper "[Multi-hop Reading Comprehension across Multiple Documents by Reasoning over Heterogeneous Graphs](https://www.aclweb.org/anthology/P19-1260/)"

Based on PyTorch

## Overview
Evaluation code for HDEGraph on [WikiHop leaderboad](http://qangaroo.cs.ucl.ac.uk/leaderboard.html) with pretrained models. 

## Installation
1. git clone

2. Install PyTorch. The code has been tested with PyTorch >= 1.0

2. Install the requirements

3. Download [pretrained models](https://drive.google.com/open?id=10tk8Ny-KklmLm7oPfpRcDkWUdTcWxT-7). Put zip file into the same folder with `run.py`, and unzip it.

## Running
Run
```
python run.py input_file output_file
```

`input_file` can be WikiHop dev file or other data sets organized in the same format with WikiHop.

`output_file` is the file where predictions locate at

## Citation
```
@inproceedings{tu2019multi,
  title={Multi-hop Reading Comprehension across Multiple Documents by Reasoning over Heterogeneous Graphs},
  author={Tu, Ming and Wang, Guangtao and Huang, Jing and Tang, Yun and He, Xiaodong and Zhou, Bowen},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages={2704--2713},
  year={2019}
}
```
