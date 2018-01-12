# WaifuNet
Training a convolutional neural network with cute anime girls.

There is some old stuff I did last year under the `old` branch. This was just a simple model with prediction over
just four non-overlapping classes. Also the very simple VGG network architecture was used. But now let's try something
more larger scale.

## Goals

1. Train a standard ResNet50 model with 2 million pictures from danbooru. Goal is multi-label tagging, meaning
there can be multiple correct labels that can be assigned 

2. Distributed training with data parallelism to cope with the large amount of data and the lack of better hardware.
It is easier to find a small cluster of mediocre PCs than one really good PC.

3. **Final Goal:** Joint network to remove watermarks from anime pictures (use data augmentation here)
and enlarge them (super resolution). For this I need to read more about GANs and the U-net architecture.
