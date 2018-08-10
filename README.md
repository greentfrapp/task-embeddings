# Features to Weights

#### or The Surprising Effectiveness of Cosine Similarity for Few-shot Classification and Regression

This repository started as a potential idea for metalearning. While I eventually did not find anything worth a full publication, I did find several surprising results, which are documented below.

Note: the following assumes understanding of the metalearning problem. Otherwise, see [here](http://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/) for a concise (and still pretty relevant) summary of the field by Chelsea Finn.

## Rethinking Weights

Consider a simple classifier with a 3-dimensional input ![\mathbf{x}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bx%7D), a 3-dimensional output ![\mathbf{y}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7By%7D) (one-hot over 3 classes), such that (assuming no bias):

![\mathbf{y}=\sigma(\mathbf{x}^\top\mathbf{W})](http://latex.codecogs.com/gif.latex?%5Cmathbf%7By%7D%3D%5Csigma%28%5Cmathbf%7Bx%7D%5E%5Ctop%5Cmathbf%7BW%7D%29) 

The weights matrix ![\mathbf{W}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BW%7D) (assuming no bias) has shape (3, 3). The magnitude of ![W_{i,j}](http://latex.codecogs.com/gif.latex?W_%7Bi%2Cj%7D) (where each column corresponds to a class) indicates the importance of ![x_i](http://latex.codecogs.com/gif.latex?x_i) in determining ![y_j](http://latex.codecogs.com/gif.latex?y_j). The sign of ![W_{i,j}](http://latex.codecogs.com/gif.latex?W_%7Bi%2Cj%7D) indicates the direction of the relationship between ![x_i](http://latex.codecogs.com/gif.latex?x_i) and ![y_j](http://latex.codecogs.com/gif.latex?y_j).

That is about as much we can say about the weights matrix.

However, suppose we normalize both ![\mathbf{x}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bx%7D) and each ![\mathbf{W}_{:,j}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BW%7D_%7B%3A%2Cj%7D), before performing the dot product. 

Let's consider just the weights that affect the first class (![\mathbf{W}_{:,1}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BW%7D_%7B%3A%2C1%7D)) and denote the normalized vectors as ![\mathbf{\hat{x}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7Bx%7D%7D) and ![\mathbf{\hat{W_{:,1}}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7BW_%7B%3A%2C1%7D%7D%7D).

The dot product effectively computes the cosine similarity between ![\mathbf{\hat{x}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7Bx%7D%7D) and ![\mathbf{\hat{W_{:,1}}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7BW_%7B%3A%2C1%7D%7D%7D).

- If the dot product gives 1, ![\mathbf{\hat{x}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7Bx%7D%7D) and ![\mathbf{\hat{W_{:,1}}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7BW_%7B%3A%2C1%7D%7D%7D) are identical.
- If the dot product gives 0, ![\mathbf{\hat{x}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7Bx%7D%7D) and ![\mathbf{\hat{W_{:,1}}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7BW_%7B%3A%2C1%7D%7D%7D) are orthogonal
- If the dot product gives -1, ![\mathbf{\hat{x}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7Bx%7D%7D) and ![\mathbf{\hat{W_{:,1}}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7BW_%7B%3A%2C1%7D%7D%7D) are in opposite directions

We can also interpret ![\mathbf{\hat{W_{:,1}}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7BW_%7B%3A%2C1%7D%7D%7D) as a template, prototype or 'normal' example of class 1. Then it makes perfect sense to classify whether ![\mathbf{\hat{x}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7Bx%7D%7D) belongs to class 1 by calculating the cosine similarity between ![\mathbf{\hat{x}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7Bx%7D%7D) and ![\mathbf{\hat{W_{:,1}}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7BW_%7B%3A%2C1%7D%7D%7D).

#### Here's the kicker.

Suppose we have trained a model on the 3 classes. Then we discover (oh shit!) that there's a 4th class that we missed out. If we had trained the model using the normalized vectors (above), there is an easy way to get the weights for the 4th class without retraining the entire weights matrix.

We simply calculate the average input feature vector over all the examples of the 4th class, then normalize it. This gives us the template or prototype for the new class. Then we can simply set that as an additional row in the weights matrix, ![\mathbf{\hat{W_{:,4}}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7BW_%7B%3A%2C4%7D%7D%7D).

This is the concept used by Gidaris & Komodakis ([2018](https://arxiv.org/abs/1804.09458)) in tackling few-shot classification. It is also slightly related to Prototypical Networks by Snell et al. ([2017](https://arxiv.org/abs/1703.05175)) although they use Euclidean distance from the prototypes as a measure for classification.

## Vanilla Cosine Similarity for Few-shot Classification

Rather than directly porting the feature vector to the weights matrix, Gidaris & Komodakis ([2018](https://arxiv.org/abs/1804.09458)) trained a weights generator, which ranged from simple element-wise (Hadamard) product to an attention-based model.

The authors also pretrained the feature extractor on all 64 classes in the miniImageNet training set, just as you would train a regular classifier, except we add in the normalization mentioned above.

I was unable to reproduce the results in the paper with a Tensorflow implementation (the authors did release their [PyTorch code](https://github.com/gidariss/FewShotWithoutForgetting)). However, I did find that simple vanilla cosine similarity (no pretraining, no weights generator, no nothing) worked surprisingly well!

Here's a visual outline of using vanilla cosine similarity for metalearning.

## Results on Omniglot, CIFAR-FS and miniImageNet

#### Omniglot Accuracy

Using 4 convolution blocks, with 64 filters each.

| Method | 5 way 1 shot | 20 way 1 shot |
|---|:---:|:---:|
| Siamese Net (Koch, 2015) | 97.3 | 88.2 |
| Matching Nets (Santoro et al., 2016) | 98.1 | 93.8 |
| MAML (Finn et al., 2017) | 98.7 ± 0.4 | 95.8 ± 0.3 |
| SNAIL (Mishra et al., 2017) | **98.96 ± 0.2** | 97.64 ± 0.3 |
| Vanilla Cosine | **99.3 ± 0.3** |  |

#### CIFAR-FS Accuracy

Using 4 convolution blocks, with 64 filters each.

| Method | 5 way 1 shot | 5 way 5 shot |
|---|:---:|:---:|
| MAML (Finn et al., 2017) | **58.9 ± 1.9** | 71.5 ± 1.0 |
| Proto Net (Snell et al., 2017) | 55.5 ± 0.7 | 72.0 ± 0.6 |
| Relation Net (Sung et al., 2018) | 55.0 ± 1.0 | 69.3 ± 0.8 |
| R2-D2 (Bertinetto et al., 2018) | **60.0 ± 0.7** | **76.1 ± 0.6** |
| Vanilla Cosine | **59.6 ± 1.0** | **76.5 ± 0.9** |

#### CIFAR-FS Accuracy

Using 4 convolution blocks, with [96, 192, 384, 512] filters.

| Method | 5 way 1 shot | 5 way 5 shot |
|---|:---:|:---:|
| MAML (Finn et al., 2017) | 53.8 ± 1.8 | 67.6 ± 1.0 |
| Proto Net (Snell et al., 2017) | 57.9 ± 0.8 | 76.7 ± 0.6 |
| R2-D2 (Bertinetto et al., 2018) | **64.0 ± 0.8** | 78.9 ± 0.6 |
| Vanilla Cosine | **64.78 ± 1.0** | **80.5 ± 0.8** |

#### miniImageNet Accuracy

Using 4 convolution blocks, with [96, 192, 384, 512] filters.

| Method | 5 way 1 shot | 5 way 5 shot |
|---|:---:|:---:|
| MAML (Finn et al., 2017) | 40.9 ± 1.5 | 58.9 ± 0.9 |
| Meta LSTM (Ravi & Larochelle, 2017) | 43.4 ± 0.8 | 60.6 ± 0.7 |
| Proto Net (Snell et al., 2017) | 42.9 ± 0.6 | 65.9 ± 0.6 |
| Relation Net (Sung et al., 2018) | 50.4 ± 0.8 | 65.3 ± 0.7 |
| R2-D2 (Bertinetto et al., 2018) | **51.2 ± 0.6** | **68.2 ± 0.6** |
| Vanilla Cosine | **52.4 ± 1.8** | **69.9 ± 1.7** |

SOTA results on this dataset uses ResNet-like architectures.

| Method | 5 way 1 shot | 5 way 5 shot |
|---|:---:|:---:|
| LEO (Rusu et al., 2018) | **60.06 ± 0.05** | **75.72 ± 0.18** |

## Few-shot Regression

Extending the method to few-shot regression required some major modifications, which ruined the whole 'vanilla cosine similarity is great' vibe. But the results are still pretty surprising.

Cosine similarity only outputs values ranging [-1, 1]. This obviously does not work for most regression tasks, which can have infinite output range. However, we can reuse the idea of using extracted training features to generate the weights. I use an attention mechanism, adapting the Transformer architecture ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) to convert extracted training features and training labels into weights (ie. a weights generator). The generated weights are then used for prediction.

Here's a visual outline of the method.

One way to interpret this is to see the weights generator as a *systems of linear equations* solver. Suppose our extracted feature vector is 3-dimensional. Assuming we have a simple linear layer to generate a final scalar prediction, we basically have to solve for the coefficients of each of the 3 values in the feature vector. The weights generator learns to solve for the coefficients after training on different tasks. 

Assuming no noise, we only require 3 equations (pairs of training samples + labels) to solve for 3 coefficients. But in the case of metalearning, we have few examples (typically 1 to 10) to solve for many coefficients (40 in the sinusoid toy task).

Another way to interpret this is by analogy to a human. Consider the sinusoid toy problem (below). If a human was presented with 5 points from the function, it might be difficult for them to infer the correct function. However, if we provide information about the task distribution (such as, 'The functions are all sinusoids.') we can greatly reduce the solution space and make it possible for the human to give a good prediction. In that sense, both the feature extractor and the weights generator contain learned information about the task distribution.

## Results on Sinusoid Toy Problem

## Ablation Study

## Thoughts

The overarching theme in the above methods involves learning a general feature extractor and then converting training features to weights. This is rather analogous to how humans learn so quickly in new environments (relative to AI).

For instance, when exposed to a new [Sonic the Hedgehog game](https://blog.openai.com/retro-contest/), we do not pick up the game from scratch. We utilize *trained feature extractors* and recognize that the pixels translate to higher-level features in the form of platforms, spikes and rings etc. Then we just have to learn what these features mean eg. spikes are bad, rings are good.

It might be the case that the feature extractor needs to adapt when encountering new tasks. This will then be better modeled by algorithms such as MAML ([Finn et al., 2017](https://arxiv.org/abs/1703.03400)), where the entire network is updated for every task. But the above experiments seem to indicate this might not be necessary.

Another concern is that in the above methods, the weights are only generated for the single final layer. This assumes that the output of the trained feature extractor only requires a single layer's transformation to arrive at the final prediction. On a deeper level, this also builds on the assumption that all other layers other than the final layer can be shared between tasks. Such assumptions might not necessarily hold and it will be interesting to consider if we can extend this model to more accommodate more than one post-feature-extractor layer.

## Instructions

## Notes

I am sharing this via GitHub because I thought the results were cool but not really significant enough to warrant a full publication.

Citations to this are not required but greatly appreciated!
