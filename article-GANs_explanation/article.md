

### Clear explanation of GANs: how noise is turned into images?

Let's first answer one of the questions which comes to mind when hearing the acronym “GAN” which stands for **Generative Adversarial Networks**, what are they used for and what is the purpose of an adversarial or a competitive model architecture?

The name which first comes to mind when thinking about AI image generation has to be DALL-E, an image-generating model developed by OpenAI.

![](https://cdn-images-1.medium.com/max/1000/1*viFmsPJkg6axGp5CRDz74A.png)

> Image generated from the DALL-E website, created by OpenAI

![](https://cdn-images-1.medium.com/max/1000/1*nRKs3pm3sxLWNa-PcNRWGg.png)

Although DALL-E and GANs have the same purpose, the model architectures used differ. DALL-E and the infamous ChatGPT use the same **generative pre-trained transformer** architecture to produce novel images based on input text prompts, while StyleGANs, an extension of GANs, use an **adversarial architecture** to do the same thing.

If you are already an expert in the realm of GANs and want to find out more about **StyleGANs** the following [link](https://machinelearningmastery.com/introduction-to-style-generative-adversarial-network-stylegan/) has an in-depth explanation of the changes made to better suit GANs for image generation.

#### Now let's deep dive into the explanation of GANs

One of the most common analogies used to describe the mechanism behind GANs is that of the **Art forger (the thief) and the policeman**, the policeman has the job of detecting the authenticity of a painting, while the thief is trying to fool him. The thief is constantly trying to make fake paintings while the policeman is constantly trying to detect the counterfeit paintings. The competition or adversarial relationship between them leads to the improvement of both. As the thief gets _incrementally better_ so does the policeman.

![](https://cdn-images-1.medium.com/max/1000/1*zDhPHNfiAnXDJVRv4YGHcg.png)

> An example of a genuine painting vs a forgery

In the beginning, the paintings made by the thief are of poor quality and are easily detected by the policeman. Still, with feedback from the policeman, the thief keeps improving his paintings until the policeman can no longer distinguish the fakes from the genuine paintings.

In GANs, the thief is the **generator model**, while the policeman is the **discriminator model.** Thus we realize that we only need the discriminator model (the policeman) to help improve the generator model (the thief) since our goal is to **generate images**.

However, we notice that while the generator model improves, so does the discriminator. This means that we can apply the discriminator model to tasks such as **deep fake detection**.

![](https://cdn-images-1.medium.com/max/1000/1*nu79D-SNE1yF3gClose9OA.jpeg)

Ok then,

Now that we understand how adversarial relationships work towards achieving mutual benefit for both the generator and the discriminator model we can move on to the next point.

#### What is the input of the model and how does it improve?

Since we are working with images let's first understand how they are processed:

To explain how our images are processed let's assume that the task we want to achieve is detecting images which contain **circles**, for simplicity's sake we will use a 3x3 black and white pixel grid to avoid worrying about RGB values.

  

![](https://cdn-images-1.medium.com/max/1000/1*ldywl6xP0TcoYBLLeeyAjw.png)

> This pixel grid represents the perfect circle, with all pixels having
> high values except for the corners and the middle pixels

As you've figured out the circle above is our desired output, thus, the purpose of our generator model will be to produce the image above.

There is a catch, however, as with other things in life, not all things are exact copies of each other, the same goes for circles :)

This means that the images below will also be considered circles within the context of our explanation.

  

![](https://cdn-images-1.medium.com/max/500/1*hIIPxgrzeNkoFvCtChRAEw.png)

![](https://cdn-images-1.medium.com/max/500/1*pZP5aZ3LxqeBCc-LCrBKcQ.png)

![](https://cdn-images-1.medium.com/max/500/1*--Q1Z9UrZUeYnrq08h6YzQ.png)

#### Now that we know how circles look, let's see some forgeries or random data

![](https://cdn-images-1.medium.com/max/750/1*y9Ic0d9-xRLWAudEN-KrAQ.png)

![](https://cdn-images-1.medium.com/max/750/1*IdSGzPsFbgfYz6Ovw4yIGA.png)

The images above do not resemble circles, which leads us to believe that these aren't images created by our generated model.

#### Let's now talk about our discriminator model and how it processes data

The input from our images will be a list containing the brightness of each pixel, for the following image it will look like this.

    [[0.1 , 0.82, 0.01],  
     [0.75, 0.04, 0.87],  
     [0.15, 0.74, 0.15]]

![](https://cdn-images-1.medium.com/max/1000/1*B_jkZg9X3qDZvDglIXfSfQ.png)

Our discriminator model works like a simple **binary classification model**, which outputs **Yes** or **No** based on the image. The machine learning model used for making these classifications is a **neural network**, it takes in the array as input and after processing it and passing it through the sigmoid activation function outputs a number ranging from 0 to 1. This number represents **the probability of an image being real**.

If you are unfamiliar with the concept of **neural networks** the following video series has a great [explanation](https://youtu.be/CqOfi41LfDw) :)

The discriminator is trained on labelled data consisting of real and fake images of circles, the error of the model is then calculated. The error formula used is log loss which is the most popular error function for binary classification tasks, the formula is the following

![](https://cdn-images-1.medium.com/max/1000/1*SrnHWd-WGCooP3G2zpl5eA.png)

> y is the actual value (0 or 1), p is the predicted value (0 or 1)

The formula above can be broken up into:

-   -ln(prediction) if the actual value is 1
-   -ln(1-prediction) if the actual value of 0

You can visualize the logic behind this formula by looking at the graphs below.

  

![](https://cdn-images-1.medium.com/max/1000/1*mUWNRoG_ABkWc7oFEwN7tw.png)

> When the actual value is 1, Error= -ln(prediction), if the prediction
> isn't close to 1, the error will be large.

  

![](https://cdn-images-1.medium.com/max/1000/1*F-A5IY79oMMbVfRgrLUdZA.png)

> When the actual value is 0, Error= -ln(1-prediction), if the
> prediction isn’t close to 0, the error will be large.

After the error is calculated the weights and bias of the discriminator model are adjusted through the process of **backpropagation**.

#### The generator

Now to generate our images we will need a random input, let's call it **X**. In our case, **X** will be an integer, however, in general, **X** is a vector which comes from a fixed distribution. For example, **X** might be a vector from a latent space representing a word. If you aren't familiar with the concept of latent space here is an [explanation](https://towardsdatascience.com/understanding-latent-space-in-machine-learning-de5a7c687d8d) :)

It's important to realize that this random input, **X**, is what allows us to generate an infinite of possible images.

Now that we have our input, **X,** we can feed it into a **neural network**, which will output an array, in our case a 3x3 matrix that represents an image. This image is then fed into the discriminator network, after receiving feedback from the discriminator network, we know by how much to adjust the weights and bias of our generator model.

#### How does the generator improve

The discriminator model will calculate the Error of the image generated knowing that the image is fake, which means that the loss function will be -ln(1-prediction), since actual = 0, while the generator model will calculate the error based on the formula -ln(prediction), since it tries to create real images.

For example, the discriminator calculates the following prediction: 0.74, here is what our error will look like.

-   discriminator error: -ln(1–0.74),
-   generator error: -ln(0.74).

![](https://cdn-images-1.medium.com/max/1000/1*QAdINL8ZP2kKGcu3OdeV_A.png)

> The relationship between the generator and the discriminator is summed
> up within this image.

If you made it to the end of the article congratulations 👏🎉🎆

You now understand how Generative Adversarial Networks work.

If this article was useful to you and you found out something new, please support me by leaving a like. For more articles related to machine learning, you can check out SIGMOID’s medium page.