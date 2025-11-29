# Custom-ML-Framework
This Framework Was Purely Custom Made by us And we Have conducted some analytical test of It (lyapunov stability test, exploration vs exploitation), although, This repository contains the Whole Block of cellularAutomataNet, Epsitron Transformer, And EpsilonPolicy, Hope you understand the rigorous math and its Functions. :)
Creator = Anonimity X.11 / Indonesia.

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

●. KL Divergence:
KL Divergence is Used To Calculate sensitivity on How much the Logit is shifting from uniform logit.
- Code Formula:
```math
Kl_divergence = np.sum(logit * np.log(logit)) - np.log(uniform_logit))
```
From the code, This Version Of KL divergence was much more sensitive to How much it has diverged, using ``` log(uniform)``` tells the model how much it has diverged, This Formula was Proved to be more Numerically stable and Efficient At Calculating Divergence from uniformity.

●. Logistic Growth equation:
CellularAutomataNet Used A Logistic equation to maximize Logistic Growth and Potentially Maximize Growth efficiency from Superlinear Manifold with Dynamic Constraint From Logistic equation:

``` Logistic satisfiability = 1.0 + Logistic_stability / sigmoid```
Where sigmoid creates a bounded Superlinear growth and stability ensures Logistic constraint.

●. Curvature:
Curvature Is Used to calculate the geometry curve of the logits and the curvature of each Nested logit or Probabilities Simulation.

- Code Formula:
```math 
curvature = np.mean(np.abs(np.diff(np.diff(logit))))
```

• From The Code formula, ```
numpy.mean()``` is used to calculate mean on logits itself directly. While For ```numpy.abs()``` itself is to turn each scalar inside the list to be an absolute value, so double differential scaling will be much easier and precisely accurate, the double ```numpy.diff()``` is used to calculate the differential value of each scalar inside that logit, double usage here is used to achieve precision higher.

●. Nested Meta Simulations:
Nested meta Simulations here Are A bunch of ```numpy.exp()``` scaling that exponentiate logits to see how the model can simulate logit by scaling it to predict how future logits looked like.

●. Hitchins Moduli Space:
Hitchins moduli space is a mathematical equation to calculate the theoretical geometry of each matrix in this case, the equation codes:

```
(example from CellularImbrium class)
				alpha1 = np.dot(x, multipliers)
				alpha2 = np.dot(x, mutation)
				alpha3 = np.dot(x, equilibrium)	
							
				trA1 = np.linalg.norm(alpha1)
				trA2 = np.linalg.norm(alpha2)
				trA3 = np.linalg.norm(alpha3)	
				s1 = trA1**2 - trA3 / equilibrium
				s2 = (1/2) * (trA2**3 + trA3** 2  / equilibrium)
				s3 = (1/6) * (trA3**3 - (3 * trA1 * trA2**2) + (2 * trA3**3) / (3 * trA3**3) / equilibrium)
				all_sample_scores = (1.0 + s1 + s3 / s3 + s2 - s1)
```

where trAa1 used to calculate th the magnitude of a vector or matrix, and seasons (the s1, s2, s3 part) will calculate the theoretical geometrical position based on the magnitude of that matrix or a vector by dividing it with equilibrium, it ensures each seasons can theoretically return a stable equilibrium scalar.



# CellularAutomataNet Class

CellularAutomataNet is A MLP (Multi Layer perceptron) Neural Network That Used Chain_algorithm, forward_algorithm, and tune_algorithm, to dynamically shift and refine logits before activations, this feedback loop creates a shifting nonlinear process that were constantly self correcting and self adjusting. 


# EpsitronTransformer Class

Epsitron is A Custom Transformer meant to Make Agents Adaptive To All environments, both Noisy And Silent, Epsitrons Modules or functions acts as both Linear attention For Agent and Non Linear attention with Meta convergence and multi Matrix Calculation to Achieve convergence faster than regular Transformer.


# LaFoldBot Class

LaFoldBot is A Custom Meta-Helper that helps to Automate And Refined Logits, It Acts As A Coordinator to Supervised Logits or Probabilities, its presence acts like a backbone helper for FolderNet To Help FolderNet Refines its logits.

# geometricalSeeker Class

geometricalseeker is a custom helper seekers that seeks stable divergence, conv, entropy, of scalars inside a matrix by exploring the probabilities of equilibrium in a theoretical dynamic  matrix interaction per training iteration. The mathematical Calculus derivatives used was riemannian equation.

# cellularImbrium Class

CellularImbrium Is a Hierarchical Tree that consist of two Networking Layer perceptron and training functions used to determine if a function should return a value, or scalars or not, by using dynamic lyapunov stability, it ensures that the equation remains stable per iteration and score >= stable_equilibrium, the other Functions contains a stable reprogramming function used to reprogram scalars based on the iteration scores, there is also special Nodes that contains A function that calculates all three special nodes that was acquired through hitcins moduli space equation.

# Solo Development History By me (2025- Now):
- 3 months Ago Foldernet Was successfully Made by me, for 2 months ive refined it to be a purposeful agent for my small game.
- After FolderNet was created, i realized that conventional Transformer doesnt directly Work and fit with my FolderNet, since it was using Standard Linear Q, K, V. So FolderNet requires attention that works by using geometric learning, FolderNet was then renamed to CellularNet.
- EpsilonPolicy was made for the Agents Policy that enables the Agent to Change its behavior by injecting a Stable exploration and exploitation with geometric optimization.
- LafoldBot And geometrical seeker was later made to Help EpsitronTransformer and EpsilonPolicy, It used as a meta helper for Development.
- Days Ago cellularImbrium Was made to execute Many of that Modules to cooperate and form a stable dynamic hieraichal tree.

