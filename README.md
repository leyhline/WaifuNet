Trying out some different network architectures for deep learning. 

Continuation from **gelbooru-analysis** repository.

Step 4: Network architecture
--------------
~~Use TensorFlow. (Yuck! Google!)~~

~~Better: Use [Keras](https://keras.io/) with TensorFlow backend.~~

Seems like Keras got integrated into TensorFlow. Furthermore it seems like Keras sucks for more involved layers. Conclusion: Use TensorFlow.

And read a lot of papers/tutorials.

Ideas: 
* My own [Deepdreaming](https://en.wikipedia.org/wiki/Deepdreaming) Network.
* Image recognition/ automatic labeling.
* Generating networks for my very own AI generated Waifu.

Step 5: Training and Profiling ✓
--------------
Or: How to find a PC with Nvidia GPU which is fast enough for processing all this data.

Idea: Use pool computers for training. 

~~Save images at some online storage and dynamically load them (in e.g. 100 MB batches) for training.~~

Save tar archive of lossy compressed images at some online storage. Then load everything into RAM and shuffle them around as you want.

Inspiration
--------------
* http://mattya.github.io/chainer-DCGAN/ (source of repo image)
* http://waifu2x.udp.jp/
* http://paintschainer.preferred.tech/
* http://illustration2vec.net/

What to do next
----------------
(When there's time; hopefully starting October.)

Train a network on multi-labeling of images like in http://illustration2vec.net/. But use some state-of-the-art networks like ResNet or Inception or Inception-ResNet or whatever. Next use these weights for implementing Deepdreaming.
