Torch Starter
=============

This is a simple Torch7 starter package. It can be used  as a simplified kickoff point for a Torch project.

<img src='http://i.imgur.com/3a5fAAy.png' width='500'>

I pieced together this package largely from [Torch7 resources online](https://github.com/soumith/imagenet-multiGPU.torch). I mostly just copied the code, and stripped a lot of extra functionality out, to make it easier to hack on. 

If something is not clear, or could be made more simple, please let me know. The goal is to be simple.

Installation
------------

If you are at CSAIL, you can use my Torch installation:
```bash
. /data/vision/torralba/commonsense/torch/install/bin/torch-activate
export LD_LIBRARY_PATH=/data/vision/torralba/commonsense/cudnnv5/cuda/lib64:$LD_LIBRARY_PATH
```

Otherwise, installation is fairly simple. You need to install:
- [Torch7](http://torch.ch/docs/getting-started.html#_)
- [cunn](https://github.com/torch/cunn) for training on GPU
- [cudnn](https://github.com/soumith/cudnn.torch) for faster training on GPU
- [tds](https://github.com/torch/tds) for some data structures
- [display](https://github.com/szym/display) for graphs 

You can install all of these with the commands:
```bash
# install torch first
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh

# install libraries
luarocks install cunn
luarocks install cudnn
luarocks install tds
luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
```

### Learning Resources
- [Torch Cheat Sheet](https://github.com/torch/torch7/wiki/Cheatsheet)
- [60 minute blitz](https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb)

Model
-----
I trained an AlexNet-esque network on [Places365](http://places2.csail.mit.edu/)
with this code, which you can [download here](http://mit.edu/vondrick/torch-starter/pretrained.zip). This model obtains
50% top-1 accuracy on the validation set. This is slightly worse than the published result because we didn't do averaging over 10 crops.

If you use this model, please cite the Places2 paper (of which I am not
affiliated).  Note this model is slightly different from the AlexNet in Caffe.
Notable differences: no groups in the convolutions, no
spatial normalization, batch normalizaiton, trained with Adam instead of SGD,
and sampling with replacement. It is unclear to me whether these changes have a
significant impact on performance. 

Data Setup 
----------
By default, we assume you have a text file that lists your dataset. This text does not store your dataset; it just lists filepaths to it, and any meta data. Each line in this text file represents one training example, and its associated category ID. The syntax of the line should be: 
```
<filename><tab><number>
```
For example:
```
bedroom/IMG_050283.jpg    5
bedroom/IMG_237761.jpg    5
office/IMG_838222.jpg     10
```
The `<number>` should start counting at 1. 

After you create this file, open `main.lua` and change `data_list` to point to this file. You can specify a `data_root` too, which will be prepended to each filename. 

Training
--------
Define your model in the `net` variable. By default, it is AlexNet. To learn more about the modules you can use, see [nn](https://github.com/torch/nn/blob/master/README.md). You can also adjust your loss with the `criterion` variable. 

Remember to also adjust any options in `opt`, such as the learning rate and the number of classes. Setting these hyperparameters is a bit of an art, but generally it is recommended to use a learning rate of `0.001` and batch size of at least `64`, but `128` or `256` may be better if you have the memory. For a systematic study, see [this paper](https://arxiv.org/pdf/1606.02228v2.pdf).

Finally, to start training, just do:

```bash
$ CUDA_VISIBLE_DEVICES=0 th main.lua
```
where you replace the number after `CUDA_VISIBLE_DEVICES` with the GPU you want to run on. 
You can find which GPU to use with `$ nvidia-smi` on our GPU cluster. Note: this number is 0-indexed, unlike the rest of Torch!

During training, it will dump snapshots to the `checkpoints/` directory every epoch. Each time you start a new experiment, you should change the `name` (in `opt`), to avoid overwriting previous experiments. The code will not warn you about this (to keep things simple).

Evaluation
----------
To evaluate your model, you can use the `eval.lua` script. It mostly follows the same format as `main.lua`. It reads your validation/testing dataset from a file similar to before, and sequentially runs through it, calculating both the top-1 and top-5 accuracy. 

Graphics, Logs
--------------
If you want to see graphics and the loss over time, in a different shell on the same machine, run this command:
```bash
$ th -ldisplay.start 8000 0.0.0.0
```
then navigate to ```http://HOST.csail.mit.edu:8000``` in your browser. Every 10th iteration it will
push graphs. 

On the CSAIL vision cluster, you can run this code out-of-the-box, and it will start to train
AlexNet on the Places2 database. 
