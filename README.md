
# Gan
### Introduction
Can a computer learn how to draw?  
  
Perhaps "draw" it is not the more appropriate term, but try to watch the result. It seems impossible, right? These astonishing images have been made thanks to the development of a Deep Convolutional Generative Adversarial Networks (DCGAN), following - partially - the work done by [Alec Radford et al.](https://lnkd.in/e-rKFft).

### The model
Generative models can be used to realize realistic samples for colourization, super-resolution, artwork but also for simulate times series data. In particular, GANs are a type of implicit density estimation. This means that this NN sample from a simple distribution, like Gaussian noise, and learn a transformation from these distributions directly to the training distribution that we want.

So, a Gan is constituted by two different networks (you can check every detail [here](https://github.com/MarcoChain/Deep-Convolutional-Generative-Adversarial-Networks/blob/master/Gan.py)):

-   A Discriminator network that should be able to distinguish between fake and real images.
-   A generator network that tries to fool the discriminator by generating real-looking images.

The algorithm used to train this network is clearly explained in the following pseudo-code of Ian Goodfellow et al. :

![](https://cyclegans.github.io/img/Harshad/GAN/algo.png)


> Written by [MarcoChain](https://www.linkedin.com/in/marcogullotto/).
