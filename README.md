This is a simple Torch7 starter package. It can be used either as a educational
tool or simplified kickoff point for a Torch project.

This is pieced together with many awesome Torch7 resources online. I mostly
just copied the code, and stripped a lot of extra functionality out, to make it
easier to hack on.  

To use it, you just have to create a text file that lists images and a corresponding category.
Each line in this text file represents one training example. The syntax of the line should be: 

  <filename><tab><number>
  
Then, open main.lua and change data_list to point to this file. You can specify a data_root too, which will
be prepended to each filename. 

Define your model in the `net' variable. By default, it is AlexNet.

Then, to start training, just do:

  $ CUDA_VISIBLE_DEVICES=0 th main.lua

where you replace the number after CUDA_VISIBLE_DEVICES with the GPU you want to run on. 
You can find which GPU to use with `$ nvidia-smi' on our GPU cluster.

Note: If you want to see graphics and the loss over time, run this command on the same
machine: 
  $ th -ldisplay.start 8000 0.0.0.0
then navigate to http://HOST.csail.mit.edu:8000 in your browser. Every 10th iteration it will
push graphs. 

Note: If you are at CSAIL, you can run this code out-of-the-box, and it will start to train
AlexNet on the Places2 database. 
