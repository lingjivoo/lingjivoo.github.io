---
title: Softmax Temperature
author: Cheng Luo
date: 2024-01-12 19:30:00 +1100
categories: [Study]
tags: [Technique]
pin: true

---
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

Copied from [Softmax Temperature](https://medium.com/mlearning-ai/softmax-temperature-5492e4007f71)
---
Temperature is a hyperparameter of LSTMs (and neural networks generally) used to control the randomness of predictions by scaling the logits before applying softmax. Temperature scaling has been widely used to improve performance for NLP tasks that utilize the Softmax decision layer.
For explaining its utility, we will consider the case of Natural Language Generation, wherein we need to generate text by sampling out novel sequences from the language model (using the decoder part of the seq-to-seq architecture). At each time step in the decoding phase, we need to predict a token, which is done by sampling from a softmax distribution (over the vocabulary) using one of the sampling techniques. In short, once the logits are obtained, the quality and the diversity of the predictions is controlled by the softmax distribution and the sampling technique applied thereupon. This article is about tweaking the softmax distribution to control how diverse and novel the predictions are. The latter will be covered in a future article.

Fig 1 is a snapshot of how the prediction is made at one of the intermediate timesteps in the decoding phase.

![](https://raw.githubusercontent.com/lingjivoo/lingjivoo.github.io/master/_posts/2024-1-12-Softmax-Temperature-vis.assets/Logits-transformation-by-Softmax.webp)


## But what is the issue here?
---

The generated sequence will have a **predictable** and generic structure. And the reason is **less entropy** or **randomness** in the softmax distribution, in the sense that the likelihood of a particular word (corresponding to index 9 in the above example) getting chosen is way higher than the other words. A sequence being predictable is not problematic as long as the aim is to get realistic sequences. But if the goal is to generate a novel text or an image which has never been seen before, randomness is the holy grail.

## The Solution?
---
**Increase the randomness.**
And that’s precisely what Temperature scaling does. It characterizes the entropy of the probability distribution used for sampling, in other words, it controls how surprising or predictable the next word will be. The scaling is done by dividing the logit vector by a value T, which denotes the temperature, followed by the application of softmax.

![](https://raw.githubusercontent.com/lingjivoo/lingjivoo.github.io/master/_posts/2024-1-12-Softmax-Temperature-vis.assets/Temperature-Scaling-Equation.webp)


The effect of this scaling can be visualized in Fig 3:
![](https://raw.githubusercontent.com/lingjivoo/lingjivoo.github.io/master/_posts/2024-1-12-Softmax-Temperature-vis.assets/Vis-outputs.webp)

![](https://raw.githubusercontent.com/lingjivoo/lingjivoo.github.io/master/_posts/2024-1-12-Softmax-Temperature-vis.assets/dynamic-vis.gif)

The distribution above approaches uniform distribution giving each word an equal probability of getting sampled out, thereby rendering a more creative look to the generated sequence. Too much creativity isn’t good either. In the extreme case, the generated text might not make sense at all. Hence, like all other hyperparameters, this needs to be tuned as well.

## Conclusion
The scale of temperature controls the smoothness of the output distribution. It, therefore, increases the sensitivity to low-probability candidates. As $T \rightarrow \infty$, the distribution becomes more uniform, thus increasing the uncertainty. Contrarily, when  $T \rightarrow 0$, the distribution collapses to a point mass.
As mentioned earlier, the scope of Temperature Scaling is not limited to NLG. It is also used to calibrate deep learning models while training and in Reinforcement Learning as well. Another broader concept that it is a part of is Knowledge Distillation. Below are the links on these topics for further exploration.

---
Author: Harshit Sharma [Source](https://medium.com/mlearning-ai/softmax-temperature-5492e4007f71)
