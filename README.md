# Bayesian Jointpoint Regression

## Introduction
Here, I implement the Bayesian jointpoint regression of proposed by Miguel A. Martinez-Beneito, Gonzalo Garcia-Donato, and Diego Salmeron the the paper [A Bayesian Jointpoint Regression Model with an Unknown Number of Break Points](https://www.jstor.org/stable/23069368) using R and Stan.

The jointpoint regression program developed and distributed by the National Cancer Institute has several drawbacks:

1. It is biased towards more simplistic models (fewer break points), i.e. more conservative, when the average number of observed deaths is low, hence it may miss changes in trends when case numbers are low or when the population size is small (e.g., ethnic minorities). 

2. It is hard to quantify to what extent one selected model is more likely than others.

3. It assumes that conditional mortality rates follow a normal distribution which may be inappropriate. Ideally, mortality rates should be modeled using a Poisson or Negative Binomial population model on account of the properties of count data.

The primary reason which I believe Bayesian approaches to jointpoint modelling have not been taken up within the medical/public health community to model the changing patterns of disease mortality or incidence is due to the fact that the mathematics is involved, but even more because software which abstracts out the mathematical complications and allow for easy use by researchers outside the domain of statistics is unavailable. This implementation will try to overcome this by providing a simple interface to the model through R with full Bayesian inference supported by Stan.

## The Mathematics
Let $Y_i$ for $i \in \{1,\ldots,N\}$ denote the number of deaths at time point $t_i$. The standard formulation for a jointpoint regression with $J$ change-points is 
$$
g\left(\mathbb{E}\left[Y_i | t_i \right]\right) = \alpha + \beta_0 t_i + \sum_{j = 1}^{J} \beta_j (t_i - \tau_j)^+.
$$
where $g$ is a link function. This can be reparameterised as,
$$
\mathbb{E}\left[Y_i | t_i \right] = \alpha + \beta_0 (t_i - \bar{t}) + \sum_{j=1}^{J}\beta_j \mathcal{B}_{\tau_j}(t_i)
$$
where $\mathcal{B}_{\tau_j}(t)$ is called a *break-point* function and defined as,
$$
\mathcal{B}_{\tau_j}(t) = 
\begin{cases}
    a_{0j} + b_{0j}t, & \text{if } t \leq \tau_j\\
    a_{1j} + b_{1j}t, & t > \tau_j
\end{cases}
$$
Four conditions are imposed upon $\mathcal{B}_{\tau_j}(t)$, such that the value of the function is unambiguously determined for every $t$:

1. $\lim_{t \rightarrow \tau_j^-}\mathcal{B}_{\tau_j}(t) = \lim_{t \rightarrow \tau_j^+}\mathcal{B}_{\tau_j}(t)$ which in other words means that the function is continuous even around the breakpoints. This is a more realistic assumption compared to the frequentist jointpoint approach as drastic changes in mortality trends are unlikely.

2. $\sum_{i=1}^{N}\mathcal{B}_{\tau_j}(t_i) = 0$: New breakpoints are imposed to be geometrically orthogonal to the intercept term.

3. $\sum_{i=1}^{N}\mathcal{B}_{\tau_j}(t_i)t_i = 0$: New breakpoints are imposed to be geometrically orthogonal to the slope.

4. $\mathcal{B}_{\tau_j}(t_i) = 1$: So that $\beta_j$ has the role of measuring the magnitude of the breakpoint in the location where the change of tendency takes place.












