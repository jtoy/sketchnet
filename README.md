# sketchnet

sketchnet - processing code generator

can we teach a computer to draw pictures with code. We will use processing and java/jruby code paired with the pictures it makes to train the system.  The model will then be able to take unseen images generate working code for it.

As far as we know, this should be the first working model to generate working usable code.


# problem areas

* The model will most likely fail to generate realistic looking images. future models will need to work with more realistic looking images

* how do we measure performance of the model?  The model should generate valid code. We can use a javascript parser to know if we are generating valid code.  How can we know if the pictures are valid though? We can test if the images produce any color, which they should besides just white.  

* what resolution should image be in? Should we expect images to be at the same scale?

* do we care about care angles/fov ? or stick to one "angle"?



# data collection



General rule of thumb is 5000 images per category and 10 million examples to surpass human performance.

We will try to write code to generate training data. Can we use genetic programming.  For the objective function, we can use 2 things, does it generate color? And is the image "interesting"? based off a neural network model.

we want to avoid the random function as much as possible as we hypothesis that the model will not learn the random function.  


# functions we are interested in modeling

To help us generate data, we will need to think about what kinds of functions are interesting for us to teach the model

* sin/cos
* shapes: ellipse/rect/etc
* line
* bezier



# references
* https://github.com/karpathy/neuraltalk   andrew karpathy neuraltalk first implementation
* https://github.com/karpathy/neuraltalk2  rewrite #2
* https://github.com/tensorflow/models/tree/master/im2txt/im2txt im2txt tensorflow version
* http://blog.otoro.net/2015/12/28/recurrent-net-dreams-up-fake-chinese-characters-in-vector-format-with-tensorflow/
* https://arxiv.org/abs/1609.06647 Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge
* https://cl.ly/2w1d3e1F1P3S NeuralTalk on Embedded System and GPU-accelerated RNN
* https://cl.ly/2P2Z2y2e2N0j Deep Visual-Semantic Alignments for Generating Image Descriptions
* https://arxiv.org/pdf/1611.01989.pdf   deepcoder paper
* https://arxiv.org/abs/1510.07211  On End-to-End Program Generation from User Intention by Deep Neural Networks
* http://memorability.csail.mit.edu/index.html lamem site
* http://people.csail.mit.edu/khosla/papers/iccv2015_khosla.pdf lamem paper
* https://arxiv.org/abs/1506.06726 skip thought vectors
* https://arxiv.org/pdf/1405.0312.pdf mscoco paper
* https://arxiv.org/pdf/1602.02410.pdf Exploring the Limits of Language Modeling
* https://people.csail.mit.edu/rinard/paper/popl16.pdf Automatic Patch Generation by Learning Correct Code
* https://arxiv.org/abs/1705.07962 pix2code: Generating Code from a Graphical User Interface Screenshot
* https://arxiv.org/pdf/1609.04938.pdf What You Get Is What You See: A Visual Markup Decompiler
* https://arxiv.org/pdf/1511.07275.pdf LEARNING SIMPLE ALGORITHMS FROM EXAMPLES



# notes on how to generate 10 million files

each group of code should generate 10,000 variations, that is still 1k different types of files I need to come up with.
Can I find people's scripts online?  if I do 100,000 variations, I can do 100 files by hand. We can try to use lamem net to choose the "good" looking images
variants:
* square of different colors covering the screen
* shapes of different sizes and colors covering the screen
* lines across the screen causing different gradients
* different grid systems


# TODOs

* setup schedule and milestones
* draw architect
* setup research log
* be able to regenerate test with code revision/data revision/hyper parameters
* each training requires a unique path/model to store to, dont allow a default, we want to put the name of the revision in general

# command to run

## Prepare data

## Training

