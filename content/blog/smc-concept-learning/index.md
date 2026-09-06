---
title: "[Reading-Notes] A Sensorimotor-Contingency-Based Representation for Concept Learning"
date: 2026-09-06
tags: ["paper-reading", "Representation Learning", "Reinforcement Learning"]
description: "Reading notes on representing concepts via sensorimotor contingencies (Hay et al., AAAI 2018)."
hideSummary: true
cover:
    image: "smc-cover.png"
    alt: "Cover image for the reading note on sensorimotor-contingency-based concept learning"
    relative: true
draft: false
---

> Hay, N. J., Stark, M., Schlegel, A., Wendelken, C., Park, D., Purdy, E., Silver, T., Phoenix, D., & George, D. (2018). **Behavior Is Everything: Towards Representing Concepts with Sensorimotor Contingencies.** *Proceedings of the AAAI Conference on Artificial Intelligence, 32(1).* [DOI: 10.1609/aaai.v32i1.11547](https://doi.org/10.1609/aaai.v32i1.11547)

## Motivation

It has long been recognized that current AI systems understand complex concepts in a fundamentally different way than the human brain does. However, a thought-provoking paper at AAAI'18 offers a promising direction toward human-like concept learning, motivated by *sensorimotor contingency theory* from cognitive science.

The authors argued that this gap might come from a fundamental misalignment between human and typical AI representations:

> While the former are grounded in rich sensorimotor experience, the latter are typically passive and limited to a few modalities such as vision and text.

In contrast to traditional passive vector representations that merely refer to common objects, the SMC-based approach represents concepts through two basic behaviors: *classification* and *bring-about*. The intuition is that if an agent truly understands a concept, it should be able to (1) **discriminate** whether the concept holds, and (2) **bring about** a state where it does.

Importantly, a concept is not merely a label for an entity—it can also be a relational property, such as "whether an object is *pushable*", which requires the agent to first attempt to push it (bring-about) and then signal the result (classification).

## Overall Framework

### Overview of Task Setting

Let us begin with a directed acyclic graph (DAG). Each node in this graph corresponds to a concept (implemented as an SMC), and each directed edge indicates that the higher-level node can invoke the lower-level one.

Inside every node, the concept (behavior) is of the contrast form, looks like `cl_self-left-of-two-objects_vs_one` or `ba_self-in-container_from_between-potential-containers`. And one concept corresponds to one specific SMC, which can be seen as a policy $\pi_{\theta}(a|o)$.

The goal is to train n policies, one per concept, each sharing the same network architecture but with distinct parameters $\theta_{i}$, $i \in \{1, 2, \ldots, n\}$.

### How things operate in a single node

#### 1. A set of environments

Authors use CSP to generate a set of reasonable PixelWorld environments for the training of the SMC (training of an optimal policy $\pi_{\theta}$ in substance).

#### 2. A glance at model training

We adopt a GRU architecture, whose weights $\theta$ is stochastically initialized and serve as an optimization goal.

Then in each epoch (totally 200 epochs for every node), we randomly generate some environments (corresponding to this concept) and run current policy to gain a trajectory per environment. Note that a few of steps consist of a full trajectory, and total number of steps are confined within 2000 in our experiments. Finally we collect all rewards from these steps for one-time update of $\theta$:

$$\theta \Leftarrow \theta + \nabla J(\theta)$$

while $J(\theta)$ is the cumulated reward. In this paper, authors considered NPO for policy gradient method.

#### 3. What happens in a single step?

For a single step, the pipeline is as follows:

1. Input **current observation** $o_{t}$, **result vector** $v_{t}$ (explained later), and **hidden state (history information)** $h_{t-1}$;
2. Output a **prob. distribution of actions** (in the form of a prob. vector) and an **updated hidden state** $h_{t}$;
3. Then sample from it to choose an action to conduct, and record the signal returned by lower-level SMC into result vector if a lower-level SMC is invoked;
4. Finally use **new observation** $o_{t+1}$, **updated result vector** $v_{t+1}$, and **updated hidden state** $h_{t}$ as input of the next step.

#### 4. Role of Result vector

- If the sampled action is a primitive behavior (e.g., `UP/DOWN` or a `signaling` action), the result vector remains unchanged and carries no new information.
- Nevertheless, if a lower-level SMC is invoked, without the result vector, the agent would not receive the invocation result from the lower-level SMC in the previous step. Consequently, the GRU's input would remain identical to that of the previous step, causing the agent to "stagnate" (i.e., go in circles).
- This vector persists across steps and is only updated when the corresponding SMC is invoked again.

## Advantages

1. For some concept tasks such as `whether an object is pushable or not`, it's difficult for a CNN-based agent but easy for SMC-based agent, since SMC-based agent learns concepts by interacting with environment, therefore gains richer experience about the environment;
2. Empirically, the hierarchical SMC approach achieves ~85% accuracy on classification tasks vs. ~75% for flat policies, and ~75% success on bring-about tasks vs. ~53% for flat policies;
3. Incorporate the cognitive science hypothesis and so that be more interpretable.

## Limitations

1. A large number of concepts must be hand-designed via curricula, which limits the method's applicability to real-world conceptual learning;
2. The generalization ability is bad, since it trains a policy for each predefined concept, and has no idea what to do upon encountering an unfamiliar concept, which the authors themselves suggest could be addressed by incorporating *intrinsic motivation* (Barto & Singh, 2004).
