# Useful formula
- CE $H(P, Q)=-\sum p \ln q$
- KL: $D_{\mathrm{KL}}(P \| Q)=\sum p \ln (p/q)$
- Entropy: $H(X) = - \mathbb{E} \ln p$
- Gaussian $\mathcal{N}(\mu, \sigma)=e^{-(x-\mu)^2/2/\sigma^2}/(\sqrt{2\pi}\sigma)$
    - Entropy $H=[\ln(2\pi\sigma^2)+1]/2$
    - $D_{\mathrm{KL}}(1\|2)=[\sigma_1^2/\sigma_2^2 - 1 - 2\ln \sigma_1/\sigma_2 + (\mu_1-\mu_2)^2/2\sigma_2^2]/2$
- multi gaussian $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})=(2 \pi)^{-k / 2} \operatorname{det}(\boldsymbol{\Sigma})^{-1 / 2} \exp (-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\top} \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}))$
    - Entropy $H = [\ln \det (2\pi e \boldsymbol{\Sigma})]/2$
    - $D_{\mathrm{KL}}(1\|2)=[\operatorname{tr}\left(\boldsymbol{\Sigma}_2^{-1} \boldsymbol{\Sigma}_1\right)+\left(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1\right)^{\mathrm{T}} \boldsymbol{\Sigma}_2^{-1}\left(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1\right)-k + \ln |\boldsymbol{\Sigma}_2| / |\boldsymbol{\Sigma}_1|]/2$
- conditional gaussian $\mu_{a|b} =\mu_a+\Sigma_{ab}\Sigma_{bb}^{-1}(x_b-\mu_b)$, $\Sigma_{a|b} = \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$
- Woodbury $(A + U C^{-1} V)^{-1} = A^{-1} - A^{-1}U(C+VA^{-1}U)^{-1}V A^{-1}$
    - $(A + ww^{\top})^{-1} = A^{-1} -  (1+w^{\top}A^{-1}w)^{-1} A^{-1}ww^{\top} A^{-1}$
- Matrix derivative: 
    - $\partial_x A^{-1} =-A^{-1} (\partial_x A) A^{-1}$
    - $\partial_x\ln\det A=\mathrm{Tr}\left(A^{-1}\partial_x A\right)$
    - define $(\partial_A f)_{ij}=\partial_{a_{ji}} f$, then $\partial_{A^{\top}} f = (\partial_A f)^{\top}$
        - $\partial_A \mathrm{tr}(BA)=B$, $\partial_A \ln\det A=A^{-1}$
        - $\partial_A \mathrm{tr}(ABA^{\top})=(B+B^{\top})A^{\top}$

# Bayesian Linear Regression
- Model: $\beta^{\top}\phi(x)+E$, $E{\small \sim}\mathcal{N}(0,\sigma^2)$ i.i.d., Prior $\beta{\small \sim} \mathcal{N}(0,\Lambda^{-1})$, data $\Phi=[\phi(x_{1:n})]^{\top} \in \mathbb{R}^{n\times m}$
- Posterior $\beta{\small \sim}\mathcal{N}(\mu_{\beta},\Sigma_{\beta})$, $\Sigma_{\beta} = (\Lambda + \sigma^{-2}\Phi^{\top}\Phi)^{-1}, \mu_{\beta} = \Sigma_{\beta}\Phi^{\top} Y$
- Bayesian Prediction $\mathbb{E}_{\beta_{\mathrm{posterior}},E} Y^* = \Phi^{*}\mu_{\beta}$, $\mathrm{Cov}[Y^*] = \Phi^{*}\Sigma_{\beta} \Phi^{*\top} + \sigma^{2} I_{p\times p}$, first term *epistemic*, uncertainty on $f^*$, due to lack of data, second *aleatoric*, uncertainty about $y^* | f^*$, irreducible noise.
- MAP, Ridge regression: use $\mu_\beta$ as prediction.

# Gaussian Process
- kenrel $k(x, x')$, symmetric, positive semi-definite
- $k_1+k_2$, $k_1 k_2$, $ck_1$, $f(k_1)$ are also kernels.
- *Stationary* if $k\left(x, x^{\prime}\right)=k\left(x-x^{\prime}\right)$
- *isotropic* if $k\left(x, x^{\prime}\right)=k\left(\left\|x-x^{\prime}\right\|_2\right)$
- Rational Quadratic Kernel $(1+\alpha (x-x')^2/2h^{2})^{-\alpha}$
- RBF (Square exponential ) $\exp \left(-\left\|x-x^{\prime}\right\|_2^2 / h^2\right)$
- Periodic $\exp(-2\sin^2(\pi|x-x'|/p)/h^2)$
- exponential $\exp \left(-\left\|x-x^{\prime}\right\|_1 / h\right)$, no where diff'able
- linear $\sigma_b^2+\sigma_v^2(x-c)\left(x^{\prime}-c\right)$, or $\phi(x)^{\top}\Lambda^{-1}\phi(x')$, $O(n^3)$ to $O(n)$ acceleration.
- Matérn, $k(x,x';\nu, h)$, $\nu=1/2$ exponential, $3/2$ once diff, $5/2$ twice diff, $\infty$ gaussian, 
- Kernel regression, prior $(f(x^{*}_{1:m}))^{\top} {\small \sim} \mathcal{N}(0, \sigma^{2} I_{m\times m} + K_{m\times m})$, posterior $\mathbb{E} [Y^*] = K_{m\times n}(\sigma^{2} I_{n\times n}+K_{n\times n})^{-1}Y$, $\mathrm{Cov}[Y^*] = K_{m\times m} - K_{m\times n} (\sigma^2  I_{n\times n} + K_{n\times n})^{-1} K_{n\times m}+\sigma^{2}I_{m\times m}$
- optimizing hyperparameter: cross-validation on predictive performance
- Bayesian approach: maximize marginal likelihood of data $\hat{\theta}=\arg\max_{\theta} p(Y|X,\theta)=\int p(Y|X,f)p(f|\theta)\mathrm{d}f$, likelihood$\times$prior, big when both terms big.
- Objective: $\arg\min_\theta[\ln |K_Y(\theta)|+y^{\top}K_Y(\theta)^{-1}y]/2$ , where $K_Y(\theta)=K(X|\theta)+\sigma^2I$(NLL), volume of hypothesis space + goodness of fit.
- training $\frac{\partial}{\partial \theta_j} \log p(\mathbf{y} | X, \theta)=\frac{1}{2} \operatorname{tr}\left(\left(\alpha \alpha^T-\mathbf{K}^{-1}\right) \frac{\partial \mathbf{K}}{\partial \theta_j}\right)$, non-convex, local optima.
- Sampling
    1. $f=L\varepsilon$, where $K=LL^{T}$ using cholesky decompo
    2. *forward samp* $p(f_{1:N})=p(f_1)p(f_2|f_1)\ldots$.

## Fast GP
- By default needs $O(n^3)$ for solving linear system, linear kernel $O(nd^2)$ (also can solve recursively)
- Solution: GPU parallel
- Local methods: For decaying kernel, ignore points when $\left|k\left(x, x^{\prime}\right)\right| < \mathbf{\tau}$, still expensive if many points closed by.
- Kernel approximation $k\left(x, x^{\prime}\right) \approx \phi(x)^T \phi\left(x^{\prime}\right), \phi(x) \in \mathbb{R}^m$, $O\left(n m^2+m^3\right)$
    - similar idea: Random Fourier features, Nystrom Features.
- RFF: stationary kernel has fourier form $k\left(x-x^{\prime}\right)=\int_{\mathbb{R}^d} p(\omega) e^{j \omega^T\left(x-x^{\prime}\right)} d \omega$, (is kernel iff $p(\omega)$ nonegative.
    - Idea: interpret as expectation $k\left(x-x^{\prime}\right)=\mathbb{E}_{\omega, b}\left[z_{\omega, b}(\boldsymbol{x}) \cdot z_{\omega, b}\left(\boldsymbol{x}^{\prime}\right)\right]$, $\omega {\small \sim} p(\omega),b {\small \sim} U([0,2 \pi])$
    - Acceleration by replacing with sampling, then use $z(\boldsymbol{x}):=\sqrt{\frac{2}{D}}\left[\cos \left(\boldsymbol{\omega}_1^T \boldsymbol{x}+b_1\right), \ldots, \cos \left(\boldsymbol{\omega}_D^T \boldsymbol{x}+b_D\right)\right]$, and $k\left(x, x^{\prime}\right) \approx z(x)^T z\left(x^{\prime}\right)$.
    - approximate uniformly well $\sup _{\boldsymbol{x}, \boldsymbol{x}^{\prime} \in M}\left|z(\boldsymbol{x})^T z\left(\boldsymbol{x}^{\prime}\right)-k\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)\right| \leq \epsilon$
    - No need of data, (good and bad)
- Inducing Points, joint is $p\left(\mathbf{f}^*, \mathbf{f}\right)=\int p\left(\mathbf{f}^*, \mathbf{f} | \mathbf{u}\right) p(\mathbf{u}) d \mathbf{u}$, approximate by $\int q\left(\mathbf{f}^* | \mathbf{u}\right) q(\mathbf{f} | \mathbf{u}) p(\mathbf{u}) d \mathbf{u}$, while $q\left(\mathbf{f}^* | \mathbf{u}\right)$ and $q(\mathbf{f} | \mathbf{u})$ approximate $p\left(\mathbf{f}^* | \mathbf{u}\right)$ and $p(\mathbf{f} | \mathbf{u})$.
    - Subset of Regressor(SoR): $q_{S O R}(\boldsymbol{f} | \boldsymbol{u})=N\left(\boldsymbol{K}_{\boldsymbol{f}, \boldsymbol{u}} \boldsymbol{K}_{\boldsymbol{u}, \boldsymbol{u}}^{\boldsymbol{- 1}} \boldsymbol{u}, 0\right)$, can show $k_{S o R}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=k(\mathbf{x}, \mathbf{u}) \mathbf{K}_{\mathbf{u}, \mathbf{u}}^{-1} k\left(\mathbf{u}, \mathbf{x}^{\prime}\right)$
    - Fully independent training conditional (FITC) $q_{F I T C}(\boldsymbol{f} | \boldsymbol{u})=\prod_{i=1}^n p\left(f_i | u\right)$
    - complexity mainly on inverting $K_{\boldsymbol{u}, \boldsymbol{u}}$ (cubic), linear in $n$.

# Laplace approximation
- When $p(\theta|\mathcal{D})$ intractable, approximate with $q(\theta)=\mathcal{N}\left(\hat{\theta}, \Lambda^{-1}\right)$, where $\hat{\theta}=\arg \max _\theta p(\theta |\mathcal{D})$, $\Lambda=-\nabla \nabla \log p(\hat{\theta} | y)$
- may lead to poor overconfident approximation
## Bayesian Logistic Regression
- mode $\hat{\mathbf{W}}=\arg \min _{\mathbf{w}} \sum_{i=1}^n \log \left(1+\exp \left(-y_i \mathbf{w}^T \mathbf{x}_i\right)\right)+\lambda\|\mathbf{w}\|_2^2$, optimized by SGD.
- var $\Lambda = \mathbf{X}^T \operatorname{diag}\left(\left[\pi_i\left(1-\pi_i\right)\right]_i\right) \mathbf{X}$
- prediction is 1D-integration $p\left(y^* | \mathcal{D} \right) = \int \sigma\left(y^* f\right) \mathcal{N}\left(f ; \hat{\mathbf{w}}^T \mathbf{x}^*, \mathbf{x}^{* T} \Lambda^{-1} \mathbf{x}^*\right) d f$

# Variational Inference
- tractable $q$ to approximate intractable $p$, $q^* \in \arg \min _{q \in \mathcal{Q}} K L(q \| p)$ reverse KL, mode selection but over confident
    - equiv to $\arg \max _q \mathbb{E}_{\theta {\small \sim} q(\theta)}[\log p(\theta, y)]+H(q)$ $=\arg \max _q \mathbb{E}_{\theta {\small \sim} q(\theta)}[\log p(y | \theta)]-K L(q \| p(\theta))$
    - $q^* \in \arg \min _{q \in \mathcal{Q}} K L(p \| q)$ forward KL, more spreaded distribution
- equivalent to evidence lower bound (ELBO) $\log p(y)=\log \mathbb{E}_{\theta {\small \sim} q}\left[p(y | \theta) \frac{p(\theta)}{q(\theta)}\right] d \theta$ $\geq \mathbb{E}_{\theta {\small \sim} q}\left[\log \left(p(y | \theta) \frac{p(\theta)}{q(\theta)}\right)\right] d \theta$ $=\mathbb{E}_{\theta {\small \sim} q}[\log p(y | \theta)]-K L(q \| p(\theta))$
- reparam trick $\mathbb{E}_{\theta {\small \sim} q_\lambda}[f(\theta)]=\mathbb{E}_{\epsilon {\small \sim} \phi}[f(g(\epsilon ; \lambda))]$, so $\nabla_\lambda \mathbb{E}_{\theta {\small \sim} q_\lambda}[f(\theta)]=\mathbb{E}_{\epsilon {\small \sim} \phi}\left[\nabla_\lambda f(g(\epsilon ; \lambda))\right]$
- score gradient $\nabla_\lambda L(\lambda)=\mathbb{E}_{\theta {\small \sim} q_\lambda}\left[\nabla_\lambda \log q(\theta | \lambda)(\log p(y, \theta)-\log q(\theta | \lambda))\right]$
    - for diagonal $q$, twice expensive as MAP (additional variance param)
    - both unbiased estimator

# MCMC
- approximate unnormalized $p$ by sampling from it. 
- Hoeffding ineq $P\left(\left|\mathbb{E}_P[f(X)]-\frac{1}{N} \sum_{i=1}^N f\left(x_i\right)\right|>\varepsilon\right) \leq 2 \exp \left(-2 N \varepsilon^2 / C^2\right)$ for bounded $f\in[0,C]$, exponential acc.
- Markov Chain, stationary distribution, detailed balance (DB) $Q(\mathbf{x}) P\left(\mathbf{x}^{\prime} | \mathbf{x}\right)=Q\left(\mathbf{x}^{\prime}\right) P\left(\mathbf{x} | \mathbf{x}^{\prime}\right)$
    - Ergodic theorem: for ergodic Markov chain, finite state $\lim _{N \rightarrow \infty} \frac{1}{N} \sum_{i=1}^N f\left(x_i\right)=\sum_{x \in D} \pi(x) f(x)$
- Metropolis Hastings MC, need proposal distribution samplable $R\left(X^{\prime} | X\right)$.
    - sample $x'$ given $x$ and accept with prob $\alpha=\min \left\{1, \frac{Q\left(x^{\prime}\right) R\left(x | x^{\prime}\right)}{Q(x) R\left(x^{\prime} | x\right)}\right\}$
- Gibbs sampling: sample per coordinate by $P\left(X_i | \boldsymbol{x}_{-i}\right)$, can show $\alpha \equiv 1$, random order of coordinates retains DB, fixed order no DB. 
## continuous space: 
- Metropolis adjusted Langevin Algorithm (MALA, LMC) $R\left(x^{\prime} | x\right)=\mathcal{N}\left(x^{\prime} ; x-\tau \nabla f(x) ; 2 \tau I\right)$, to prefer high density region, efficient, mixing time polynomial in dimension.
- Stochastic Gradient Langevin Dynamics (SGLD), minibatch on posterior, $p(\theta|\mathcal{D})\approx \exp(\log p(\theta) + \frac{n}{m} \sum_{j=1}^m \log p\left(y_{i_j} | \theta_t, x_{i_j}\right))$.
    - converge if $\eta_t=O(t^{-1/3})$, but usually constant.
- Hamiltonian MC: adding momentem term.

# Bayesian DL
- noise depends on input, *heteroscedastic* $p(y | \mathbf{x}, \theta)=\mathcal{N}\left(y ; f_1(\mathbf{x}, \theta), \exp \left(f_2(\mathbf{x}, \theta)\right)\right)$
- MAP: $\arg \min _\theta \lambda\|\theta\|_2^2+\frac{1}{2} \sum_{i=1}^n\left[\frac{1}{\sigma\left(x_i ; \theta\right)^2}\left\|y_i-\mu\left(x_i ; \theta\right)\right\|^2+\log \sigma\left(x_i ; \theta\right)^2\right]$, can attenuate loss by attributing error to large var.
- Bayesian: $p\left(y^* | x^*, x_{1: n}, y_{1: n}\right)=\int p\left(y^* | x^*, \theta\right) p\left(\theta | x_{1: n}, y_{1: n}\right) d \theta$
    - using $q$ and maximizing ELBO
    - in prediciton time, approx with sampling $p\left(y^* | \mathbf{x}^*, \mathbf{x}_{1: n}, y_{1: n}\right) \approx \mathbb{E}_{\theta {\small \sim} q(\cdot | \lambda)}\left[p\left(y^* | \mathbf{x}^*, \theta\right)\right]$
    - $\operatorname{Var}\left[y^* | \theta, \mathbf{y}_{1: n}, \mathbf{x}^*\right]=\mathbb{E}\left[\operatorname{Var}\left[y^* | \mathbf{x}^*, \theta\right]\right]+\operatorname{Var}\left[\mathbb{E}\left[y^* | \mathbf{x}^*, \theta\right]\right]\approx \frac{1}{m} \sum_{j=1}^m \sigma^2\left(x^*, \theta^{(j)}\right)+\frac{1}{m} \sum_{j=1}^m\left(\mu\left(x^*, \theta^{(j)}\right)-\bar{\mu}\left(x^*\right)\right)^2$, *Aleatoric* + *Epistemic*
    - MCMC methods like SGLD SGHMC, and sample $\theta$
        - challenge: expensive to store all $\theta$
        - need to drop first samples to avoid burn-in.
        - sol1: keep a subset of m snapshots
        - sol2: approximate distribution with gaussian, works well with SGD with no noise (SWAG)
- specialized approach: 
    - Monte-carlo Dropout: $q_j\left(\theta_j | \lambda_j\right)=p \delta_0\left(\theta_j\right)+(1-p) \delta_{\lambda_j}\left(\theta_j\right)$, dropout in prediction.
    - Probablistic Ensembles, train with several subset and treat as sampling.
- Aleatoric uncertainty of classification, by injecting learnable gaussian noise $\mathbf{p}=\operatorname{softmax}(\mathbf{f}+\varepsilon)$

# Calibration
- Want predicted prob coincide with empirical freq, $P(\hat{y}=y|\hat{p}=p)=p$, if model output $p$ as the true prediction of ground truth label $y$, we want this to be close to the empirical frequency, amoung all validaiton input that the model predicts $\hat{p}=p$.
- Evaluate, partition $\mathcal{D}_{\mathrm{val}}=B_1\cup \ldots \cup B_M$, s.t. $\hat{p}_i\in((m-1)/M, m/M], \forall i\in B_m$, then $\mathrm{Acc}_m:=\sum_{i}|\hat{y}_i = y_i|/|B_m|$, $\mathrm{Conf}_m:=\sum_i \hat{p}_i / |B_m|$, Calibration wants $\mathrm{Acc}_m = \mathrm{Conf}_m$
- Expected/Maximum calibration error (ECE/MCM) is the weighted average/maximum discrepancy across bins.
- Reliability diagrams, Confidence Accuracy Plot, calibrated model should give diagonal.
- improve calibration: 
    - histogram binning, assign calibrated score to each bin $\hat{q}_i=\operatorname{freq}\left(B_m\right)$
    - Isotonic regression, find piecewise constant funciont $\hat{q}_i=f\left(\hat{p}_i\right)$ minimize bin-wise square loss $\min _{M, \theta_1, \ldots, \theta_M, a_1, \ldots, a_{M+1}} \sum_{m=1}^M \sum_{i=1}^n \mathbf{1}\left(a_m \leq \hat{p}_i<a_{m+1}\right)\left(\theta_m-y_i\right)^2$
    - Platt (temperature) scaling, binary $\hat{q}_i=\sigma\left(a z_i+b\right)$, multi-class $\hat{q}_i=\max _k \sigma_{\text {softmax }}\left(\boldsymbol{z}_i / T\right)^{(k)}$, param learnable.

# Active Learning
- Goal: decide which data to collect, provides most useful information
- *Mutual Info* loss of uncertainty after observing $F(S):=H(f)-H\left(f | y_S\right)=I\left(f ; y_S\right)=\frac{1}{2} \log \left|I+\sigma^{-2} K_S\right|$, find $S$ that maximize this rate, combinatorial problem, NP hard.
    - greedy algorithm $x_{t+1}=\arg \max _{x \in D} F\left(S_t \cup\{x\}\right)$ $=H(f|y_{S_{t}}) - H(f|y_{S_{t+1}}) = I(f;y_{x} | y_{S_{t}})$ $=\ln(1+\sigma_{x|S_{t}}^2/\sigma_n^2)/2 =\arg \max _x \sigma_t^2(x)$
    - PS: you can actually not define entropy for GP, but entropy rate, this metric taught in class comes out of nowhere.
    - MI is *monotone submodular*, $\forall x \in D \quad \forall A \subseteq B \subseteq D: F(A \cup\{x\})-F(A) \geq F(B \cup\{x\})-F(B)$, equiv to prove $H(y_x|y_A) \geq H(y_x|y_B)$ information never hurts.
    - provides constant-factor approximation, near optimal $F\left(S_T\right) \geq\left(1-\frac{1}{e}\right) \max _{S \subseteq D,|S| \leq T} F(S)$
    - fail in heteroscedastic case $x_{t+1} \in \arg \max _x {\sigma_f^2(x)}/{\sigma_n^2(x)}$, unable to distinguish epistemic and aleatoric uncertainty, where uncertain outcomes are not necessarily most informative
- Informative Sampling for Classification (BALD) $x=\arg\max_x I(\theta; y_x | \mathcal{D})$ $=H(y|x, \mathcal{D}) - \mathbb{E}_{\theta{\small \sim} p(\theta|\mathcal{D})}H(y|x, \theta)$, second term by Bayesian DL.
    - first term approx by the predictive distri of approx posterior, e.g. VI
    - second by sampling. (actually, both term is approximated by sampling...)
    - original paper's formula $\mathbb{I}\left[y, \boldsymbol{\omega} | \mathbf{x}, \mathcal{D}_{\text {train }}\right]:=\mathbb{H}\left[y | \mathbf{x}, \mathcal{D}_{\text {train }}\right]-\mathbb{E}_{p\left(\boldsymbol{\omega} | \mathcal{D}_{\text {train }}\right)}[\mathbb{H}[y | \mathbf{x}, \boldsymbol{\omega}]]$ $=-\sum_c p\left(y=c | \mathbf{x}, \mathcal{D}_{\text {train }}\right) \log p\left(y=c | \mathbf{x}, \mathcal{D}_{\text {train }}\right)$ $+\mathbb{E}_{p\left(\boldsymbol{\omega} | \mathcal{D}_{\text {train }}\right)}\left[\sum_c p(y=c | \mathbf{x}, \boldsymbol{\omega}) \log p(y=c | \mathbf{x}, \boldsymbol{\omega})\right]$ $\approx-\sum_c\left(\frac{1}{T} \sum_t \hat{p}_c^t\right) \log \left(\frac{1}{T} \sum_t \widehat{p}_c^t\right)+\frac{1}{T} \sum_{c, t} \widehat{p}_c^t \log \widehat{p}_c^t$

# Bayesian Optimization
- Goal: sequentially pick $x_1,\ldots, x_T$ to find $\max_x f(x)$ with minimal samples, with some model $y_t=f\left(x_t\right)+\varepsilon_t$
- Cumulative regret $R_T=\sum_{t=1}^T(\underbrace{\max } f(x)-f\left(x_t\right))$, sublinear $R_T / T \rightarrow 0$, implies $\max _t f\left(x_t\right) \rightarrow f\left(x^*\right)$
- Upper confidence sampling, GP-UCB, accusition funciton: $x_t=\arg \max _{x \in D} \mu_{t-1}(x)+\beta_t \sigma_{t-1}(x)$, trade-off exploration and exploitation.
- Thm: maximum information gain determines regret, if choose $\beta_t$ correctly $\frac{1}{T} \sum_{t=1}^T\left[f\left(x^*\right)-f\left(x_t\right)\right]=\mathcal{O}^*\left(\sqrt{\frac{\gamma_T}{T}}\right)$, where $\gamma_T=\max _{|S| \leq T} I\left(f ; y_S\right)$
- linear kernel $\gamma_T=O(d \log T)$, RBF $\gamma_T=O\left((\log T)^{d+1}\right)$, Matérn with $\nu>1/2$, $\gamma_T=O\left(\frac{d}{T^{\frac{d}{2 v+d}}}(\log T)^{\frac{2 v}{2 v+d}}\right)$, UCB guarantees sublinear regret convergence
- non-convex, low-dim Lipschitz optim, high-D gradient ascent
- Other option
    - expected improvement, noise free setting  $u(f(x)):=\max \left(0, f(x)-f^*\right)$, $f^*:=\max _{i=1, \ldots, t} y_i$, $a_{\mathrm{EI}}(x)=\left(\mu_t(x)-f^*\right) \Phi\left(\frac{\mu_t(x)-f^*}{\sigma_t(x)}\right)+\sigma_t(x) \phi\left(\frac{\mu_t(x)-f^*}{\sigma_t(x)}\right)$
    - probability of improvement (PI) $P(f(x) \geq f^*)=a_{\mathrm{PI}}(x)=\Phi\left(\frac{\mu_t(x)-f^*}{\sigma_t(x)}\right)$
    - Information directed sampling
    - Thomas Sampling, every time, drops samples $\tilde{f} {\small \sim} P\left(f | x_{1: t}, y_{1: t}\right)$, and select $x_{t+1} \in \arg \max _{x \in D} \tilde{f}(x)$
        - randomness in sampling $\tilde{f}$ enough for EE-tradeoff
        - similar regret bounds to UCB

# MDP
- expected value $J\left(\pi\right)=\mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r\left(X_t, \pi\left(X_t\right)\right)\right]$
- value func $V^\pi(x):=J\left(\pi | X_0=x\right)=r(x, \pi(x))+\gamma \sum_{x^{\prime}} P\left(x^{\prime} | x, \pi(x)\right) V^\pi\left(x^{\prime}\right)$, can be obtained by solving linear system.
- Thm Bellman: $V^*(x)=\max _a\left[r(x, a)+\gamma \sum_{x^{\prime}} P\left(x^{\prime} | x, a\right) V^*\left(x^{\prime}\right)\right]$, policy optimal <=> greedy w.r.t. its value func
- *Linear Programing* of Bellman ineq
- *Policy Iteration*
    - 1. compute value func $V^\pi(x)$
    - 2. get greedy policy $\pi_G$ w.r.t. $V^\pi(x)$
    - 3. $\pi \leftarrow \pi_G$
    - monotonically improve $V^{\pi_{t+1}}(x) >= V^{\pi_{t}}(x)$
    - converge to optimal in $O\left(n^2 m /(1-\gamma)\right)$ iteration
    - Every iteration requires computing a value function
    - complexity per iteration: $\mathcal{O}\left(|\mathcal{S}|^3+|\mathcal{S}|^2|\mathcal{A}|\right)$
- *Value iteration* (DP, fixed point)
    - 1. $Q_t(x, a)=r(x, a)+\gamma \sum_{x^{\prime}} P\left(x^{\prime} | x, a\right) V_{t-1}\left(x^{\prime}\right)$
    - 2. $V_t(x)=\max _a Q_t(x, a)$
    - 3. stop when $\left\|V_t-V_{t-1}\right\|_{\infty}=\max _x\left|V_t(x)-V_{t-1}(x)\right| \leq \varepsilon$
    - 4. choose greedy policy w.r.t $V_t$
    - guarantee converge to $\varepsilon$-optimal policy, proof $\|B^{*} V - B^{*}V'\|_{\infty}\leq \gamma \|V - V'\|_{\infty}$
    - per iter cost $\mathcal{O}\left(|\mathcal{S}|^2|\mathcal{A}|\right)$
    - local updates
- Which works better depends on application, can combine ideas of both algorithms
- partially observed MDP (POMDP), only obtain noisy observations $Y_t$ of hidden state $X_t$
    - powerful model, extremely intractable
    - POMDP = Belief-state MDP, can be seen as enlarged state space, $P\left(X_t | y_{1: t}\right)$, beliefs.
    - $\mathcal{B}=\Delta(\{1, \ldots, n\})=\left\{b:\{1, \ldots, n\} \rightarrow[0,1], \sum_x b(x)=1\right\}$
    - stochastic observation $P\left(Y_{t+1}=y | b_t, a_t\right)=\sum_{x, x^{\prime}} b_t(x) P\left(x^{\prime} | x, a_t\right) P\left(y | x^{\prime}\right)$
    - state update (bayesian filter) $b_{t+1}\left(x^{\prime}\right)=\frac{1}{Z} \sum_x b_t(x) P\left(X_{t+1}=x^{\prime} | X_t=x, a_t\right) P\left(y_{t+1} | x^{\prime}\right)$
    - reward $r\left(b_t, a_t\right)=\sum_x b_t(x) r\left(x, a_t\right)$
    - For finite horizon T, set of reachable belief states is finite (but exponential in T)
    - Can calculate optimal action using dynamic programming

    
# RL
- trajectories $\tau=\left(x_0, a_0, r_0, x_1, a_1, r_1, x_2, \ldots\right)$
- *episodic setting* each episodes results in a trajectory, after that environment resets
- *non-episodic / continuous* single trajectory
- *on/off-policy* agent have full control / can only observe data (from a different policy)
- *model-based*
    - estimate $P\left(x^{\prime} | x, a\right)$ and $r(x, a)$, and optimize upon it.
- *model-free*
    - estimate value func, policy-gradient, actor-critic

## Model based
- Learn MDP, MLE yields $\hat{P}\left(X_{t+1} | X_t, A\right) \approx \frac{\#\left(X_{t+1}, X_t, A\right)}{\#\left(X_t, A\right)}$, $\hat{r}(x, a)=\frac{1}{N_{x, a}} \sum_{t: X_t=x, A_t=a} R_t$
- exploration-explitation dilemma
- If sequence $\varepsilon_t$ satisfies Robbins-Monro condition $\sum_t \varepsilon_t=\infty, \sum_t \varepsilon_t^2<\infty$, converge to optim policy with $p=1$.
    - often performs fiarly well, but does't quickly eliminate clearly suboptimal acitons.
- $R_{\mathrm{max}}$ algo
    - if don't know $r(x, a)$, set to $R_{\mathrm{max}}$
    - if don't know $P\left(x^{\prime} | x, a\right)$, set $P\left(x^* | x, a\right)=1$, where $x^{*}$ fairy tale state.
    - no need to explicit choose exploration, exploitation, rule out suboptimal action quickly
    - estimate $P$ and $r$ using hoeffiding bound $P\left(\left|\mu-\frac{1}{n} \sum_{i=1}^n Z_i\right|>\varepsilon\right) \leq 2 \exp \left(-2 n \varepsilon^2 / C^2\right)$, $n_{x,a}=O(R_{\mathrm{max}}^2log(\delta^{-1})\varepsilon^{-2})$
    - thm: every $T$ times, with high prob, $R_{\mathrm{max}}$ either obtains near-optimal reward, or visit at least one unknown $(s,a)$ pair, $T$ related to mixing time of MDP wiht optimal policy.
    - thm: with prob $1-\delta$ $R_{\mathrm{max}}$ reach $\epsilon$-optimal policy with steps poly in $|X|,|A|, T, 1 / \varepsilon$, $\log (1 / \delta), R_{\max }$
- model-based:
    - memory required: store every $r(x, a)$ and $P\left(x^{\prime} | x, a\right)$, totally $O(|X|^2|A|)$
    - computation time: value/policy iteration.

## Model free
- TD-learing $\hat{V}^\pi(x) \leftarrow\left(1-\alpha_t\right) \hat{V}^\pi(x)+\alpha_t\left(r+\gamma \hat{V}^\pi\left(x^{\prime}\right)\right)$, converge to $V^{\pi}$ with $p=1$ when $\sum_t \alpha_t=\infty$ and $\sum_t \alpha_t^2<\infty$
- Q-learning $\hat{Q}^*(x, a) \leftarrow\left(1-\alpha_t\right) \hat{Q}^*(x, a)+\alpha_t\left(r+\gamma \max _{a^{\prime}} \hat{Q}^*\left(x^{\prime}, a^{\prime}\right)\right)$
    - mem: $O(nm)$
    - computation: per transition: $O(m)$
    - convergence of optimistic Q-learning, if init $\hat{Q}^*(x, a)=\frac{R_{\max }}{1-\gamma} \prod_{t=1}^{T_{\text {init }}}\left(1-\alpha_t\right)^{-1}$, and at time $t$, pick $a_t \in \arg \max _a \hat{Q}^*\left(x_t, a\right)$, then with $p=1-\delta$, algo obtains $\varepsilon$-optimal policy with steps poly in $|X|,|A|, 1 / \varepsilon$, $\log (1 / \delta)$.
    - challenge: 
        - scaling up, $|X|, |A|$ exponential in #agents
        - learn/approx value func
- TD as SGD, bootstrapping (use old value as targets)
    - $\ell_2\left(\theta ; x, x^{\prime}, r\right)=\frac{1}{2}\left(V(x ; \theta)-r-\gamma V\left(x^{\prime} ; \theta_{o l d}\right)^2\right.$
- paramterization $V(x ; \theta)$ or $Q(x, a ; \theta)$, e.g. $Q(x, a ; \theta)=\theta^T \phi(x, a)$
- DQN: $L(\theta)=\sum_{\left(x, a, r, x^{\prime}\right) \in D}\left(r+\gamma \max _{a^{\prime}} Q\left(x^{\prime}, a^{\prime} ; \theta^{\text {old }}\right)-Q(x, a ; \theta)\right)^2$
    - suffer from maximization bias, double DQN: $L(\theta)=\sum_{\left(x, a, r, x^{\prime}\right) \in D}\left(r+\gamma  Q\left(x^{\prime}, a^{*}(\theta) ; \theta^{\text {old }}\right)-Q(x, a ; \theta)\right)^2$, where $a^*(\theta)=\underset{a^{\prime}}{\operatorname{argmax}} Q\left(x^{\prime}, a^{\prime} ; \theta\right)$

## Policy gradient
- parameterize $\pi(x)=\pi(x ; \theta)$, and optimized globally $\theta^*=\arg \max _\theta J(\theta)$,
    - $\nabla J(\theta)=\nabla \mathbb{E}_{\tau {\small \sim} \pi_\theta} r(\tau)=\mathbb{E}_{\tau {\small \sim} \pi_\theta}\left[\sum_{\tau=0}^{\infty} r(\tau) \nabla \log \pi_\theta(\tau)\right]$, where can be shown $r(\tau)=\sum_{t=0}^T \gamma^t r\left(x_t, a_t\right)$ , unbiased but large variance
    - solve by adding bias $\mathbb{E}_{\tau {\small \sim} \pi_\theta}\left[(r(\tau)-b) \nabla \log \pi_\theta(\tau)\right]$, state dependent bias
    - can also show $=\mathbb{E}_{\tau {\small \sim} \pi_\theta}\left[\sum_{t=0}^T\left(r(\tau)-b\left(\tau_{0: t-1}\right)\right) \nabla \log \pi\left(a_t | x_t ; \theta\right)\right]$, choosing $b\left(\tau_{0: t-1}\right)=\sum_{t^{\prime}=0}^{t-1} \gamma^{t^{\prime}} r_{t^{\prime}}$, we get $\nabla J(\theta)=\mathbb{E}_{\tau {\small \sim} \pi_\theta}\left[\sum_{t=0}^T \gamma^t G_t \nabla \log \pi\left(a_t | x_t ; \theta\right)\right]$, where $G_t=\sum_{t^{\prime}=t}^T \gamma^{t^{\prime}-t} r_t$ (*REINFORCE*)
    - further variance reduction $\nabla J(\theta)=\mathbb{E}_{\tau {\small \sim} \pi_\theta}\left[\sum_{t=0}^T \gamma^t\left(G_t-b_t\left(x_t\right)\right) \nabla \log \pi\left(a_t | x_t ; \theta\right)\right]$, where $b_t\left(x_t\right):=b_t=\frac{1}{T} \sum_{t=0}^{T-1} G_t$
    - improvements: 
        - value-function estimates->AC
        - regularization
        - off-policy 
        - natual gradient

## Actor-Critic
- advantage function $A^\pi(x, a)=Q^\pi(x, a)-V^\pi(x)$
- Action-Value expression $\mathbb{E}_{\tau {\small \sim} \pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t Q\left(x_t, a_t\right) \nabla \log \pi\left(a_t | x_t ; \theta\right)\right]$
- On-line AC: use approximation $Q(x,a;\theta)$ instead, and train $Q$ with TD like $\theta_Q \leftarrow \theta_Q-\eta_t\left(Q\left(x, a ; \theta_Q\right)-r-\gamma Q\left(x^{\prime}, \pi\left(x^{\prime}, \theta_\pi\right) ; \theta_Q\right)\right) \nabla Q\left(x, a ; \theta_Q\right)$
    - can replace $Q$ in PG with $Q\left(x, a ; \theta_Q\right)-V\left(x ; \theta_V\right)$, *A2C*
- (trust-region) TRPO PPO $\theta_{k+1}=\arg \max _\theta \hat{J}\left(\theta_k, \theta\right) \text { s.t. } K L\left(\theta \| \theta_k\right) \leq \delta$, $\hat{J}\left(\theta_k, \theta\right)=\mathbb{E}_{x, a {\small \sim} \pi_{\theta_k}}\left[\frac{\pi(a | x ; \theta)}{\pi\left(a | x ; \theta_k\right)} A^{\pi_{\theta_k}}(x, a)\right]$, guarantees to monotonic improvement in $J(\theta)$
- Off-line:
    - use Q-learning to train $Q$, replace maximization with $L\left(\theta_Q\right)=\sum_{\left(x, a, r, x^{\prime}\right) \in D}\left(r+\gamma Q\left(x^{\prime}, \pi\left(x^{\prime} ; \theta_\pi\right) ; \theta_Q^{o l d}\right)-Q\left(x, a ; \theta_Q\right)\right)^2$, where $\theta_\pi$ close to greedy