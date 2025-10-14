# Custom-Transformer-For-ML-Framework
This Repository Is Private For Now, Since This Framework Was Purely Custom Made by me And I Have conducted some test of It, although, This repository contains the Whole Block of FolderNet, Epsitron Transformer, And EpsilonPolicy
Creater = X.11 (AstroVont)

# Mathematical Foundation

The Mathematical Principle used to built FolderNet, Epsitron Transformer and Epsilon Policy used the Nonlinear Numpy syntax Such as:
~ Numpy.log()  ~ Numpy.log1p()
~ Numpy.exp() 

The Mathematical Formula That Acts as A Foundation for Nonlinear dynamic equilibrium On All Of those Modules Were:
~ Kullback-Leibler (KL) Divergence
~ Curvature Geometry
~ Nested Exponentials
~ Efficient KL comparatives
~ sigmoid 

.- Explanation About Why I Used That Mathematical Principles is How They Can Calculate Logits Or Probabilities Sensitivity, Meta Simulations Or Planner
, and compare them directly with each Divergence formulas from the Meta Simulations and the Raw Logits.
- Below is A Compact Explanation:
1. KL Divergence is Used To Calculate sensitivity on How much the Logit is shifting from uniform logit.
Formula:

D'_{KL}(x) = \sum (x \log x - \log \text{uniform})




# FolderNet Class

FolderNet is A MLP (Multi Layer perceptron) Neural Network That Used Chain_algorithm, forward_algorithm, and tune_algorithm, to dynamically shift and refine logits before activations, this feedback loop creates a shifting nonlinear process that were constantly self correcting and self adjusting. 

