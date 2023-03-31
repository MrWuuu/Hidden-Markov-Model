# Hidden-Markov-Models

The HMM is based on augmenting the Markov chain, A **Markov chain** is a model that tells us something about the probabilities of sequences of random variables, *states*, each of which can take on values from some set.

A Markov chain makes a very strong assumption that if we want to predict the future in the sequence, all that matters is the current state.

<img src="https://zywu-blog-image.oss-cn-nanjing.aliyuncs.com/images/image-20230331100016906.png" alt="image-20230331100016906" style="zoom: 5%;" />

<center><b>Figure 1.</b> A Markov chain for weather (a) and one for words (b), showing states and transitions.</center>

A Markov chain is useful when we need to compute a probability for a sequence of observable events. In many cases, however, the events we are interested in are **hidden**: we don't observe them directly.

<img src="https://zywu-blog-image.oss-cn-nanjing.aliyuncs.com/images/image-20230331100411701.png" alt="image-20230331100411701" style="zoom: 5%;" />

<center><b>Figure 2.</b> A hidden Markov model for relating numbers of ice creams eaten by Jason (the observations) to the weather.</center>

## Likelihood Computation

### The Forward Algorithm

<img src="https://zywu-blog-image.oss-cn-nanjing.aliyuncs.com/images/image-20230331100706993.png" alt="image-20230331100706993" style="zoom: 5%;" />

<center><b>Figure 3.</b> Visualizing the computation of single element by summing all the previous value.</center>

<img src="https://zywu-blog-image.oss-cn-nanjing.aliyuncs.com/images/image-20230331100808709.png" alt="image-20230331100808709" style="zoom: 5%;" />

<center><b>Figure 4.</b> The Forward Algorithm.</center>

We can implement the script `hmm.py` to create a class `HMM` , then we can use function `forward_prob` to solve the problem of likelihood computation.

```python
hmm = HMM(V, Q, Pi, A, B, O)	# create a HMM class vairable
hmm.forward_prob()				# likelihood computation
```

### The Backward Algorithm

<img src="https://zywu-blog-image.oss-cn-nanjing.aliyuncs.com/images/image-20230331104046396.png" alt="image-20230331104046396" style="zoom: 5%;" />

<center><b>Figure 5.</b> Visualizing the computation of single element by summing all the behind value.</center>

To understand the algorithm, we need to define a useful probability related to the forward probability and called the **backward probability**. The backward probability $\beta$ is the probability of seeing the observations from time $t + 1$ to the end, given that we are in state $i$ at time $t$ (and given the automaton $\lambda$)
$$\beta_{t}(i) = P(o_{t+1}, o_{t+2}, \cdots, o_{T}|q_{t} = i,\lambda)$$
​	It is computed inductively in a similar manner to the forward algorithm.

1. Initialization:
   $$\beta_{T}(i) = 1, \ 1 \leq i \leq N$$

2. Recursion:
   $$\beta_{t}(i) = \sum_{j = 1}^{N} a_{ij} b_{j}(o_{t+1})\beta_{t+1}(j), \ 1 \leq i \leq N, \ 1 \leq t \leq T$$

3. Termination:
   $$P(O|\lambda) = \sum_{j=1}^{N}\pi_{j} b_{j}(o_{1})\beta_{1}(j)$$

We can implement the script `hmm.py` to create a class `HMM` , then we can use function `backward_prob` to solve the problem of likelihood computation.

```python
hmm = HMM(V, Q, Pi, A, B, O)	# create a HMM class vairable
hmm.forward_prob()				# likelihood computation
```

## Decoding: The Viterbi Algorithm

For any model, such as  HMM, that contains hidden variables, the task of determining which sequence of variables is the underlying source of some sequence of observations is called the **decoding** task.

<img src="https://zywu-blog-image.oss-cn-nanjing.aliyuncs.com/images/image-20230331105919076.png" alt="image-20230331105919076" style="zoom: 5%;" />

<center><b>Figure 6.</b> The Viterbi trellis for computing the best path through the hidden state.</center>

<img src="https://zywu-blog-image.oss-cn-nanjing.aliyuncs.com/images/image-20230331110047779.png" alt="image-20230331110047779" style="zoom: 5%;" />

<center><b>Figure 7.</b> Viterbi algorithm for finding optimal sequence of hidden states.</center>

We can implement the script `hmm.py` to create a class `HMM` , then we can use function `decoding` to solve the problem of decoding.

```python
hmm = HMM(V, Q, Pi, A, B, O)	# create a HMM class vairable
hmm.decoding()					# hmm coding
```

## HMM Training: The Forward-Backward Algorithm

Learning: Given an observation sequence $O$ and the set of possible states in the HMM, learn the HMM parameters $A$ and $B$.

<img src="https://zywu-blog-image.oss-cn-nanjing.aliyuncs.com/images/image-20230331110423790.png" alt="image-20230331110423790" style="zoom: 5%;" />

<center><b>Figure 8.</b> Computation of the joint probability.</center>

<img src="https://zywu-blog-image.oss-cn-nanjing.aliyuncs.com/images/image-20230331110520475.png" alt="image-20230331110520475" style="zoom: 5%;" />

<center><b>Figure 9.</b> The forward-backward algorithm.</center>

We can implement the script `hmm.py` to create a class `HMM` , then we can use function `forward_backward` to solve the problem of learning.

```python
hmm = HMM(V, Q, Pi, A, B, O)	# create a HMM class vairable
hmm.forward_backward()			# hmm learning
```

## Reference

Thanks to Stanford University for the HMM Markov Model chapter! 

