# A. Custom-ML/Neuromorphic- Circuit-Framework
This Framework Was Purely Custom Made by us And we Have conducted some analytical test of It (lyapunov stability test, spectral radius, exploration vs exploitation evaluation), although, This repository contains the Whole Block of cellularAutomataNet, Epsitron Transformer, And EpsilonPolicy, etc to provide originality of development. Hope you understand the math and its Functions. Have Fun Checking and testing. :)
Creator = Anonimity X.11 / Indonesia 


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
A. CellularAutomataNet Used A Logistic equation to maximize Logistic Growth and Potentially Maximize Growth efficiency from Superlinear Manifold with Dynamic Constraint From Logistic equation:

``` Logistic satisfiability = 1.0 + Logistic_stability / sigmoid```

Where sigmoid creates a bounded Superlinear growth and stability ensures Logistic constraint.

B. 3 fundamental superlinear with bounded logistic growth constraint equation inspired from riemannian geometry	with differential calculus that forms a logarithmic coupling.
This equation created a stable superlinear with logistic cnstraint that was ensured from the 1.0 as statistical coupling to ensure finite distances given any logits, thus creating a differential calculus inspired from riemannian optimization and logarithmic coupling where each equations complete each others weaknessess.  

```
fundamental_logistic_geodesic= optimum / 1.0 + logistic_stability
    fundamental_logistic_stability = logistic_equilibrium / 1.0 + optimum
    fundamental_logistic_sequencing = optimum / 1.0 + sigmoid
```


C. 3 fundamental logistic equation derived to acquire the thorough geodesic info per using calculus variations that used to acquire dimensionless number of probabilities to acquire a stable modelling and a high efficiency of geodesic information in any dimensionless geometric space of moduli space properties.		   
1. (1/2) was used to calculate the moduli space of the phase projection of geometric properties of dimensionless matrix that will thoroughly acquire a stable geodesic efficiency of an information transport or data where trA2 > 0 given positive logits  to ensure geodesic stability of each logistic growth.
2. (1/6) was used to calculate the theoretical geodesic space of information efficiency through euclidean range in moduli space in respect to geometric properties where trA3 > 0 to ensure efficient search through superlinear growth with logistic constraint modelling of phase projection of any given valid value.
			    
3. simplified moduli space equations combined with geodesic mapping efficiency to ensure both logistic and superlinear growth to maximize information gather efficiency and stability ensuring both appear in geometric efficiency through moduli space search with trA3 > 0 and range 0 -> finite numbers with any given positive logits, this equation will provide implicit eigenvalues to the model (as shown in the geodesic_optimum) that can map any geodesic topological space where it will ensure the models stability and convergence.

		    
```
trA1 = projection / (1.0 - slope)
   trA2 = (1/2) + stability_modelling / 1.0 + trA1**2
   trA3 = (1/6) + logistic1 / (trA2**2) - 1.0
geodesic_optimum = np.dot(x, trA3)
```
			    


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

# GeodeTransformer Class

GeodeTransformer is A transformer that implements Geometric Attention that keeps both superlinear growth with logistic growth implicitly, the module inside contains the primary Equations of trAs, that was inspiredly derived from taylor expansion, thus it encodes implicit eigenvalues and improving stability while preserving memory and exploration.

# geometricalSeeker Class

geometricalseeker is a custom helper seekers that seeks stable divergence, conv, entropy, of scalars inside a matrix by exploring the probabilities of equilibrium in a theoretical dynamic  matrix interaction per training iteration. The mathematical Calculus derivatives used was riemannian equation.

# cellularImbrium Class

CellularImbrium Is a Hierarchical Tree that consist of two Networking Layer perceptron and training functions used to determine if a function should return a value, or scalars or not, by using dynamic lyapunov stability, it ensures that the equation remains stable per iteration and score >= stable_equilibrium, the other Functions contains a stable reprogramming function used to reprogram scalars based on the iteration scores, there is also special Nodes that contains A function that calculates all three special nodes that was acquired through hitcins moduli space equation.


# VectorizedTangent:
VectorizedTangent is a Custom Complex Reasoning AI, that was able to perform on both complex noisy environments, including linear, and non linear data, with built in Certainty measurement.
This AI has Many types of Reasoning, the most dominant one includes epistemic reasoning and causality reasoning and certainty reasoning, with some other capabilities like internal belief reasoning, VectorizedTangent used advanced nested programming logic including geometric calculation and principles based calculation like anisotropy, and entropy measurement, so performance is a bit of a drawback for this AI.


# Solo Development History By me (2025- Now):
- 3 months Ago Foldernet Was successfully Made by me, for 2 months ive refined it to be a purposeful agent for my small game.
- After FolderNet was created, i realized that conventional Transformer doesnt directly Work and fit with my FolderNet, since it was using Standard Linear Q, K, V. So FolderNet requires attention that works by using geometric learning, FolderNet was then renamed to CellularNet.
- EpsilonPolicy was made for the Agents Policy that enables the Agent to Change its behavior by injecting a Stable exploration and exploitation with geometric optimization.
- LafoldBot And geometrical seeker was later made to Help EpsitronTransformer and EpsilonPolicy, It used as a meta helper for Development.
- later cellularImbrium Was made to execute Many of that Modules to cooperate and form a stable dynamic hieraichal tree.
- Logistic equations ensure the model has implicit stability and self converging toward a stable dynamic attractor.

# Empirical Validation of My Model:

A. Empirical Result:

1. Spectral Radius:
●. - Spectral Radius = 0.0 -> 0.0000000001
   - Model contraction = True
   - Convergence probabilities Guarantee from first training, given the fixed attractor.

2. Pyshics based Extension evaluation:
  1. Mean Motion of the Model can reach up to 0.4 -> 0.6
  2. Std Motion of the model can reach
Up to 0.4 -> 0.7
  3. Initial entropy gain = 1.0008e-12
  4. Transport metric of the NN = -1.3 -> -0.6
  5. Fisher distance in geometric topological space = 0.7

B. explanation:
The Empirical Results implies that the model Cant be evaluated by simple ML heuristics, it exhibits a stable dynamic between convergence and exploration with ensured stability from the mathematical equations. Despite a High convergence probabilities, the AI can Still Automatically Explores probabilities because of the topological noise given from the curvature sensitivity and initial basin projection.

# B. Custom Neuromorphic Analog Circuit Framework

1. The Neuromorphic Circuit Is A Circuit that used and manipulate the geometric properties of Noise to use it to form attractors and stable basin of exploration, thus forming the so called intelligence. 

2. My Neuromorphic Circuit design used the same Equations:
``` 
		trA1 = projection_value / (1.0 - slope)
		trA2 = (1/2) + stability_value / 1.0 + trA1**2
		trA3 = (1/6) + logistic_growth_value / (trA2**2) - 1.0	
```

Where Each Equations is Shown As V1, V2, and V3. The Reason i used this Equations because this Equations Works universally because it used The Taylor Expansion series of mathematics which provides the second and third order terms of constants that was a results of possible geometric calculations from taylor expansions series.

3. The Optimal Design choice:
-. V1 or trA1 encodes projection so the maximum Voltage of it can increase between 5 V to -5 Voltage, this is the exploration dynamics of the whole circuit, providing a greater value for trA2 and trA3.
-. V2 or trA2 Encodes stability metrics for the circuit, the Maximum Voltage of V2 is around 0.48 -> 0.52 Voltage, V2 Acts as a stable basin of attractors that models stability or memory for the circuit.
-. V3 or trA3 Encodes Growth, Which modelled how the Circuit can behave under growth of trA2, which provides another stable basin of attractors thus forming as a secondary memory, The maximum Voltage of trA3 resides around -0.83 -> -0.9 Voltage, this constant value provides the circuit a stable memory in which trA2 was able to provide as a stable projection for trA3 to form another stable basin of attractors.





  


