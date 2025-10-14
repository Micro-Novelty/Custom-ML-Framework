# Custom-Transformer-For-ML-Framework
This Repository Is Private For Now, Since This Framework Was Purely Custom Made by me And I Have conducted some test of It, although, This repository contains the Whole Block of FolderNet, Epsitron Transformer, And EpsilonPolicy
Creater = X.11 (AstroVont)

# Mathematical Foundation And Expression

The Mathematical Principle used to built FolderNet, Epsitron Transformer and Epsilon Policy used the Nonlinear Numpy syntax Such as:

```math
~ Numpy.log()  ~ Numpy.log1p()
~ Numpy.exp() 
```

The Mathematical Formula That Acts as A Foundation for Nonlinear dynamic equilibrium On All Of those Modules Were:
~ Kullback-Leibler (KL) Divergence
~ Curvature Geometry
~ Nested Exponentials
~ Efficient KL comparatives
~ sigmoid 

.- Explanation About Why I Used That Mathematical Principles is How They Can Calculate Logits Or Probabilities Sensitivity, Meta Simulations Or Planner
, and compare them directly with each Divergence formulas from the Meta Simulations and the Raw Logits.
- Below is A Compact Explanation:

1. KL Divergence:
KL Divergence is Used To Calculate sensitivity on How much the Logit is shifting from uniform logit.
- Code Formula:
```math
Kl divergence = np.sum(logit * np.log(logit)) - np.log(uniform_logit))
```
From the code, This Version Of KL divergence was much more sensitive to How much it has diverged, using ``` log(uniform)``` tells the model how much it has diverged, This Formula was Proved to be more Numerically stable and Efficient At Calculating Divergence from uniformity.

2. Curvature:
Curvature Is Used to calculate the geometry curve of the logits and the curvature of each Nested logit or Probabilities Simulation.

- Code Formula:
```math 
np.mean(np.abs(np.diff(np.diff(logit))))
```

• From The Code formula, ```
numpy.mean()``` is used to calculate mean on logits itself. 



# FolderNet Class

FolderNet is A MLP (Multi Layer perceptron) Neural Network That Used Chain_algorithm, forward_algorithm, and tune_algorithm, to dynamically shift and refine logits before activations, this feedback loop creates a shifting nonlinear process that were constantly self correcting and self adjusting. 


# EpsitronTransformer Class

Epsitron is A Custom Transformer meant to Make Agents Adaptive To All environments, both Noisy And Silent, Epsitrons Modules or functions acts as both Linear attention For Agent and Non Linear attention with Meta convergence and multi Matrix Calculation to Achieve convergence faster than regular Transformer.

