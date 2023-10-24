# SQUEEZENET

Squeeze Layer
====

Conv2D(1x1, relu)

Expand Layer
===
Conv2D(1x1, relu) 
Conv2D(1x1, relu)
Conv2D(1x1, relu)
Conv2D(1x1, relu)

Conv2D(3x3, relu)
Conv2D(3x3, relu)
Conv2D(3x3, relu)
Conv2D(3x3, relu)

Fire Module
===
- Squeeze Layer
- Expand Layer


SQUEEZENET base Architecture
====

Conv2D(kernel_size=7x7,stride=2,filter=96)

Fire_module(filter=128) 

Fire_module(filter=128) 

Fire_module(filter=256) 

Maxpool2D(stride=2,kernel_size=3x3) 

Fire_module(filter=256) 

Fire_module(filter=384) 

Fire_module(filter=384) 

Fire_module(filter=512) 

Maxpool2D(s=2,kernel_size=3x3) 

Fire_moduel(filter=512) 

dropout(0,5) 

Conv2D(kernel_size=1x1,stride=1,filter=10) 

softmax()

# Paper
[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model
size](https://arxiv.org/abs/1602.07360)
