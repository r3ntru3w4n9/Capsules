_To be moved to my github.io page in the near future._

# Capsule Network

### Dynamic Routing Between Capsules, paper by Geoffrey Hinton

**This repository is an implementation of the paper [Dynamic Routing Between Capsules][paper]**



#### Intro

More traditional deep learning models define **layers**, and each layer is consisted of an array of **neurons**. However, because of the nature of a neuron, which is defined as a mapping $$f : R^n \rightarrow R$$, the **scalar** output does not preserve **spacial** information. This is the reason Geoffrey Hinton proposed a new architecture called `Capsule Networks`. Now instead of an array of **neurons**, each layer is made by an array of **capsules**. Simply put, a capsule is a neuron that has a **vector** output compared to a normal neuron's **scalar** output. A capsule is formally defined as a mapping $$f : (R^k)^m \rightarrow R^n$$. `Capsule Networks` are believed to be more similar to how the brain works, and achieved the state-of-the-art result on **MNIST** dataset.

#### Why another `CapsNet` implementation?

This implementation is a **generalization** of the `CapsNet` implementation. The original `CapsNet` is best described as the image below. The image is from the original paper.

![img](contents/1*uItEGzY1I9NK6hl1u4hPYg.png)

Most on-line implementation does not make a generalized version of the `CapsNet`, that is, the hyper-parameters are all hard coded into the architecture of the model, with no easy way to modify them. They typically has three parts:

1. 2 convolution layers that has `channels == (256, 32)`, `kernel_size == (9, 9)`, `strides == (1, 2)`, `paddings == (0, 0)`
2. 2 capsule layers named `PrimaryCaps` and `DigitsCaps`. `PrimaryCaps` transform the output of the convolution layers to shape `(8, 16)` with convolution, and `DigitCaps` uses **dynamic routing** to transform the previous output to shape `(16, 10)`.
3. A linear `Decoder` module. Maps the output of shape` (16, 10)` back to `(28 * 28,)`.

 This implementation provides an easy way to extend the `CapsNet` structure. For the three parts above, we fit them into a slightly structure containing also three parts.

\* please note that `==` denotes equal, `=` denotes modifiable value with default.

\** also note that this implementation **is** the model proposed in the original paper with default hyper parameters.

1. N convolution layers that has `channels: tuple = (256, 32)`, `kernels: tuple = (9, 9)`, `strides: tuple = (1, 2)`, `paddings == (0, 0)`. This part includes the constitutional layer part and `PrimaryCaps` layers.
2. N capsule layers that has `channels: tuple = (16,)` and `num_capsules: tuple = (10,)`. This is called `DigitCaps` in the original implementation.
3. A linear `Decoder` module. Same as part 3 above.

The reason for this restructure is that it is better for generalizing the entire structure. _Careful observation leads me to believe that `PrimaryCaps` layer is nothing but another convolution layer wrapped as a capsule layer, without implementing the **dynamic routing algorithms** that makes capsules capsules._ The original implementation uses a `list` of convolution layer output, concatenated along the channel axis, which is simplified here as a larger convolution layer. In doing so, I achieved some speedup due to matrix multiplication optimization. Also it looks better this way.

#### Usage

usage: main.py [-h] [-Cc CONV_CHANNELS [CONV_CHANNELS ...]]

               [-Ck CONV_KERNELS [CONV_KERNELS ...]]
               
               [-Cs CONV_STRIDES [CONV_STRIDES ...]]
               
               [-ch CAPS_CHANNELS [CAPS_CHANNELS ...]]
               
               [-cc CAPS_CAPSULES [CAPS_CAPSULES ...]] [-ci CAPS_ITERS]
               
               [-dl DECODER_LAYERS [DECODER_LAYERS ...]] [-am AUG_MEAN]
               
               [-as AUG_STD] [-ar AUG_ROTATION] [-ab AUG_BRIGHTNESS]
               
               [-ac AUG_CONTRAST] [-at AUG_SATURATION] [-lt TC] [-ll LMBDA]
               
               [-lb BOUND BOUND] [-lw RECON_WEIGHT] [-e EPOCHS] [-b BATCH]
               
               [-lr LEARNING_RATE] [-mg MAX_GRAD] [-d DEVICE] [-P PROCESSES]
               
               [-r DATASET_ROOT] [-s SAVE] [-o OUT] [-p] [-tnan] [-pm]

optional arguments:

  -h, --help            

​		show this help message and exit

  -e EPOCHS, --epochs EPOCHS

​		epochs to train the capsule network

  -b BATCH, --batch BATCH

​		batch size to use in training

  -lr LEARNING_RATE, --learning-rate LEARNING_RATE

​		learning rate to use in training

  -mg MAX_GRAD, --max-grad MAX_GRAD

​		maximum gradient norm

  -d DEVICE, --device DEVICE

​		device to train model on ('cpu' or 'cuda' or 'cuda:N', N is a non negative integer)

  -P PROCESSES, --processes PROCESSES

​		number of processes to use for the dataloader

  -r DATASET_ROOT, --dataset-root DATASET_ROOT

​		location to save your dataset

  -s SAVE, --save SAVE

​		number of epoch on which to save the model state

  -o OUT, --out OUT

​		folder to save training history

  -p, --plot

​		plot training history if used

  -tnan, --terminate-on-nan

​		terminate the training process if `math.nan` is found in back propagation. slows down training a bit

  -pm, --print-module

​		include to print the module on terminal



convolution:

  -Cc CONV_CHANNELS [CONV_CHANNELS ...], --conv-channels CONV_CHANNELS [CONV_CHANNELS ...]

​		nargs = '+'. convolution layer channels in forward propagation order

  -Ck CONV_KERNELS [CONV_KERNELS ...], --conv-kernels CONV_KERNELS [CONV_KERNELS ...]

​		nargs = '+'. convolution layer kernels in forward propagation order

  -Cs CONV_STRIDES [CONV_STRIDES ...], --conv-strides CONV_STRIDES [CONV_STRIDES ...]

​		nargs = '+'. convolution layer strides in forward propagation order



capsule:

  -ch CAPS_CHANNELS [CAPS_CHANNELS ...], --caps-channels CAPS_CHANNELS [CAPS_CHANNELS ...]

​		nargs = '+'. capsule layer channels in forward propagation order

  -cc CAPS_CAPSULES [CAPS_CAPSULES ...], --caps-capsules CAPS_CAPSULES [CAPS_CAPSULES ...]

​		nargs = '+'. capsule layer capsules (as opposed to neurons in linear layers) in forward propagation order

  -ci CAPS_ITERS, --caps-iters CAPS_ITERS

​		number of iterations for each dynamic routing. higher iterations is better theoretically, but also slower in practice



decoder:

  -dl DECODER_LAYERS [DECODER_LAYERS ...], --decoder-layers DECODER_LAYERS [DECODER_LAYERS ...]

​		nargs = '+'. decoder layer neurons in forward propagation order



augmentation:

  -am AUG_MEAN, --aug-mean AUG_MEAN

​		if specified, shift input data to have mean == aug_mean

  -as AUG_STD, --aug-std AUG_STD

​		if specified, scale input data to have std == aug_std

  -ar AUG_ROTATION, --aug-rotation AUG_ROTATION

​		rotate the input image by aug_rotation degrees

  -ab AUG_BRIGHTNESS, --aug-brightness AUG_BRIGHTNESS

​		adjust the input image brightness to $$ brightness \sim (1-aug_brightness, 1+aug_brightness) $$

  -ac AUG_CONTRAST, --aug-contrast AUG_CONTRAST

​		adjust the input image contrast to $$ contrast \sim (1-aug_brightness, 1+aug_brightness) $$

  -at AUG_SATURATION, --aug-saturation AUG_SATURATION

​		adjust the input image saturation to $$ saturation \sim (1-aug_brightness, 1+aug_brightness) $$



losses:

​	the first three under this category are variables in the function $$Loss = T_c max(0, m^+-|v_c|)^2 + \lambda (1- T_c) max(0, |v_c|-m^-)^2$$

  -lt TC, --Tc TC

​		corresponds to $$ T_c $$

  -ll LMBDA, --lmbda LMBDA

​		corresponds to $$ \lambda $$

  -lb BOUND BOUND, --bound BOUND BOUND

​		`(int, int)`. corresponds to $$ (m^+, m^-) $$

  -lw RECON_WEIGHT, --recon-weight RECON_WEIGHT

​		the weight of reconstruction loss, which is not included in the above formula



##### Details

:class: `FeatureExtractor`

\* Stacked convolution layers

`__init__(self, conv_channels, kernel_sizes, strides, out_features) -> None`

​    \* conv_channels: number of channels for each convolution layer in module

​    \* kernel_sizes: size of kernels for each convolution layer in module

​    \* strides: size of strides for each kernel for each convolution layer in module

​    \* out_features: number of output `capsules`

`__call__(self, x) -> output`

​    \* x: input images in batches

​    \* output: output features (in capsule compatible form)

\* Note: ReLU here matters!! (without one I couldn't manage to get past 90% accuracy)

\* Note: output is `squash`ed. 

\* The reason being crazy _gradient vanishing_ without it



:class: `Transform`

\* Transformation of previous

`__init__(self, prev_capsules, num_capsules, in_channels, out_channels)`

​    \* prev_capsules: number of capsules of the previous `CapsuleLayer`

​    \* num_capsules: number of capsules of this `CapsuleLayer`

​    \* in_channels: dimension of the vector for each previous capsule

​    \* out_channels: dimension of the vector for each capsule of this layer

`__call__(self, U) -> U_hat`

​    \* U: output of the previous `CapsuleLayer`

​    \* U_hat: transformed output. 

​    \* every vector is transformed into a set of vectors,

​    \* corresponding to every capsule in this layer



:class: `Route`

\* implements the _dynamic routing algorithm by agreement_.

`__init__(self, num_iters, prev_capsules, num_capsules)`

​    \* num_iters: number of iterations in dynamic routing algorithm

​    \* prev_capsules: number of capsules of the previous `CapsuleLayer`

​    \* num_capsules: number of capsules of this `CapsuleLayer`

`__call__(self, U_hat) -> V`

​    \* U_hat: transformed output as in `Transform` module

​    \* V: a set of combined vectors

​    \* Note: for every previous capsule, the sum of the weights is 1

​    \* the vector input `U_hat` is then weighted by those weights

​    \* and summed together for each capsule in this layer, and we call it V



:class: `CapsuleLayer`

\* a convenience module, wrapper for :class: `Transform` and `Route`

\* the reason for the class'es existence is that its existence makes sense

`__init__(self, in_channels, out_channels, in_capsules, out_capsules, num_iters)`

​    \* in_channels: as in `Transform`

​    \* out_channels: as in `Transform`

​    \* in_capsules: as in `Route`

​    \* out_capsules: as in `Route`

​    \* num_iters: as in `Route`

`__call__(self, U_prev) -> U_self`

​    \* U_prev: output of the previous `CapsuleLayer`

​    \* U_self: output of this `CapsuleLayer`



:class: `Cequential`

\* a convenience module much like pytorch's `nn.Sequential`

\* except only number of channels and of capsules and of iterations need be passes

`__init__(self, channels, capsules, num_iters)`

​    \* channels: iterable of number of channles

​    \* capsules: iterable of number of capsules

​    \* num_iters: as in `Route`

`__call__(self, x) -> y`

​    \* x: batchs of sets of input vectors

​    \* y: batchs of sets of output vectors



:class: `CapsDecoder`

\* reconstructs the original image, with a densly connected model,

\* also acts as a regulator to ensure the preservation of spacial feature

`__init__(self, layers, categories)`

​    \* layer: number of neurons for each layer in the model

​    \* categories: number of categories of outputs (as in MNIST, 10)

`__call__(self, x) -> y`

​    \* x: batchs of sets of input vectors

​    \* y: batchs of reconstructed images



:class: `CapsNet`

\* a generalization of the `CapsNet` implemented by original

\* leaving the default unchanged as in `main.py` uses the same configuration

\* as the original paper, however, you can use different parameters as you like

\* also, this module is only a wrapper

`__init__(self, feature_net, capsule_net, decoder_net)`

​    \* feature_net: an instance of `FeatureExtractor`

​    \* capsule_net: an instance of `Cequential`

​    \* decoder_net: an instance of `CapsDecoder`

`__call__(self, input, label) -> (predicted, reconstructed)`

​    \* input: batches of images

​    \* label: given label. derived from `predicted` if not provided

​    \* predicted: predicted vectors encoding probability (length) and state (direction)

​    \* reconstructed: reconstructed images



[paper]:https://arxiv.org/abs/1710.09829
