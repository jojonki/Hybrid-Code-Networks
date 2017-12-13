# Hybrid-Code-Networks

Hybrid Code Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning  
Jason D. Williams, Kavosh Asadi, Geoffrey Zweig  
https://arxiv.org/abs/1702.03274

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

## Train & Test
```
$ python train.py --help
$ python train.py
$ python train.py --test 1
```

## TODOs
- [ ] test Task6
- [ ] Reinforcement Learning
- [ ] Action mask
