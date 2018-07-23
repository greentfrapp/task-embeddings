# TODO

- [x] Implement on CIFAR-FS
- [ ] List hyperparameters and combinations
- [ ] Implement on miniImagenet
- [x] Try using layer_norm instead of batch_norm in the networks
- [ ] Compare with cosine similarity ie. train with normalized feature and weight vectors and train a scale value
- [x] Compare with ablation of attention component; *see multimodal regression below*
- [ ] For sinusoid task, use k >= f samples to calculate closed-form solution to the weights matrix
- [x] Try multimodal regression and see if attention helps the model to perform better; possibly refer to [PLATIPUS](https://arxiv.org/abs/1806.02817); *attention helps for both multimodal and sinusoid regression*
- [ ] Implement on RL tasks, refer to [MAML](https://arxiv.org/abs/1703.03400) and [SNAIL](https://arxiv.org/abs/1707.03141); MAML RL repo [here](https://github.com/cbfinn/maml_rl); MAESN [here](https://arxiv.org/abs/1802.07245)
- [ ] Implement PyTorch version

# Ideas

- [ ] Increase number of layers in "decision" component, use SGD to train that part instead of simply attention; similar to a semi-MAML
- [x] Alternately backprop feature extractor and attention components
- [x] Add noise to output of feature extractor before passing to component

# Results

- [ ] Plot training and validation curves for multimodal, sinuosoid and omniglot with and without attention
- [ ] Visualize tSNE for omniglot activations
- [ ] Experimental parameters on CIFAR - pretraining - None, 100 steps, 500, 1000, 2000
- [ ] Experimental parameters on CIFAR - attention layers - 1, 2, 3, 4, 5
- [ ] Experimental parameters on CIFAR - hidden dim - 64, 128, 256
- [ ] Experimental parameters on CIFAR - num_classes - 1-5, 5, 10, 15
- [ ] Experimental parameters on CIFAR - num_shot_train - 1-5, 5, 10, 15
