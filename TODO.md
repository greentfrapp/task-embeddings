# TODO

- [x] Implement on CIFAR-FS
- [ ] List hyperparameters and combinations
- [ ] Implement on miniImagenet
- [ ] Try using layer_norm instead of batch_norm in the networks
- [ ] Compare with cosine similarity ie. train with normalized feature and weight vectors and train a scale value
- [ ] Compare with ablation of attention component
- [ ] For sinusoid task, use k >= f samples to calculate closed-form solution to the weights matrix
- [ ] Try multimodal regression and see if attention helps the model to perform better; possibly refer to [PLATIPUS](https://arxiv.org/abs/1806.02817)
- [ ] Implement on RL tasks, refer to [MAML](https://arxiv.org/abs/1703.03400) and [SNAIL](https://arxiv.org/abs/1707.03141); MAML RL repo [here](https://github.com/cbfinn/maml_rl); MAESN [here](https://arxiv.org/abs/1802.07245)
- [ ] Implement PyTorch version

# Ideas

- [ ] Increase number of layers in "decision" component, use SGD to train that part instead of simply attention; similar to a semi-MAML
- [ ] Alternately backprop feature extractor and attention components
- [ ] Add noise to output of feature extractor before passing to component