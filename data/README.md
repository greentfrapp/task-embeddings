# Few-shot Datasets

Paths described below are relative to the current directory `data` eg. `omniglot_resized` refers to `data/omniglot_resized/`.

## Omniglot

First download the omniglot dataset from https://github.com/brendenlake/omniglot and extract the contents of both `images_background` and `images_evaluation` into `omniglot_resized` so you should have paths like `omniglot_resized/Alphabet_of_the_Magi/character01/0709_01.png`.

Then, run the following:

```
$ cd omniglot_resized/
$ python resize_images.py
```

This resizes the images to 28 by 28 pixels.

## CIFAR-FS

First download the CIFAR-100 dataset from http://www.cs.toronto.edu/~kriz/cifar.html

***Download CIFAR-100 and NOT CIFAR-10***

Extract the contents of cifar-100-python to `cifar`.

Then run the following:

```
$ cd cifar/
$ python proc_images.py
```

This transposes the 32 by 32 pixel images to be channel-last and places them in the `test`, `train` and `val` folders.

## MiniImageNet

I can't seem to find a good source for this dataset so the simplest way to do this is to just download the dataset from my Google Drive [here](https://drive.google.com/file/d/16pifyDIvxxI0ILEtw587-Kpx1HcaU9e3/view?usp=sharing).

Then just extract the entire `miniImagenet` folder into the current folder (`data`).

## Additional Notes

When running experiments, the `make_data_tensor` method in `data_generator.py` actually generates 2e+5 training tasks by randomly selecting combinations samples to make N-way k-shot tasks. This can take a while, possibly more than 30 minutes for CIFAR-FS and MiniImageNet. 

An alternative is to generate a set of 2e+5 training tasks once, save the list of filenames as a pickle file and then load from this file when running experiments. This is significantly faster, taking less than a minute to load after the first time. 

A separate pickle file has to be generated for each type of task (5way1shot vs 5way5shot) and for each datasource.

**Once saved, be careful to load the correct pickle file ie. way and shot used in the experiment must match the loaded file.** This is a potential source of a hidden bug, which will cause the model to fail to learn anything.

The training set can be saved easily by calling the `data_generator.py` script and specifying the `--save` flag:

```
$ python data_generator.py --save --savepath='my_training_set.pkl' --datasource='cifar' --num_classes=5 --num_shot_train=1 --num_shot_test=1
```

Subsequently, when calling the `make_data_tensor` method with an initialized `DataGenerator` object, just specify the arguments `load=True` and `savepath='my_training_set.pkl'`.

This only works for the training set, since the validation and test sets are significantly faster to generate (<< 1 minute).
