# Hybrid Code Networks (unofficial)

Hybrid Code Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning  
Jason D. Williams, Kavosh Asadi, Geoffrey Zweig  
https://arxiv.org/abs/1702.03274


This repo contains the implementation of Hybrid Code Networks in PyTorch.

<img src="https://user-images.githubusercontent.com/166852/33999718-389cdb26-e0b9-11e7-8708-140da0803a5b.png" >

## Setup

You need to download following data.

- [Dialog bAbI Tasks Data 1-6](https://fb-public.box.com/s/chnq60iivzv5uckpvj2n2vijlyepze6w)
```
./dialog-bAbI-tasks/dialog-babi-kb-all.txt
./dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-trn.txt
./dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-tst-OOV.txt
```
- [Google News 300 dim (100B)](https://github.com/3Top/word2vec-api) 
```
./data/GoogleNews-vectors-negative300.bin
```

- Libraries
```
$ pip install tqdm
```

## Train & Test
```
$ python train.py --help
$ python train.py
$ python train.py --test 1
```

## TODOs
- [ ] test Task6
- [ ] Reinforcement Learning
