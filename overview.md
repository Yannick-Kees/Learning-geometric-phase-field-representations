# Nonlinear Spectral Analysis

### Examples:

|Functional $J$| p-homogeneous| $H$| $\partial J (u)$| $K_J$ | minimal norm subgradient | remark
| --- | --- | --- |--- |--- |--- | --- | 
| $\|\|u\|\|_{\ell^q}$ | $p= 1$ | $\mathbb{R}^d $ | | $\{\|\|u\|\|_{q'}\leq 1\} $ for $q=\frac{q}{q-1}$ || $K_J=\{J_*\leq 1\},\ J_*=\|\cdot\|_{q'}$
| $\sup_{\xi \in K}\langle \xi, u \rangle,\ K=-K$ | $p= 1$ | $H$ | | $\overline{conv(K)} $  ||
| $TV(u)$ | $p= 1$ | $L^2(\Omega)$ | | $\overline{ \{div(\phi) \| \phi\in W_0^{1,2}(\Omega)\leq 1 \} } $  ||
| $\|\|u\|\|_{L^1}$ | $p= 1$ | $L^2(\Omega)$ | $\{ \xi \in K_J: \xi=\pm 1, u\overset{>}{<}0; \xi\in[-1,1], u=0 \} $ | $\{\eta \in L^2:\ \|\eta\|_\infty\leq 1 \}$  |$\{ \xi \in K_J: \xi=\pm 1, u\overset{>}{<}0; \xi=0, u=0 \} $| Let $\eta\in K_J$ since $\|\eta\|=\|\eta\|^2: \langle \xi,\xi-\eta\rangle\geq 0 $
| $\|\|u\|\|_{L^\infty}$ | $p= 1$ | $L^2(\Omega)$ | $\{ \xi \in K_J: \Omega_{Max}: sgn(u)=sgn(\xi); \Omega_{Max}^C: \xi=0 \} $ | $\{\eta \in L^2:\ \|\eta\|_{L^1}\leq 1 \}$  |$\{ \xi \in K_J:  \xi=\frac{sgn(u)}{\|\Omega_{Max}:\|} \} $| $\Omega_{MAX}=\{x\| u(x)=\|\|u\|\|_{L^\infty}\} $
| $\int \|div(u)\|$ | $p= 1$ | $L^2(\Omega)$ | $\{ -\nabla v \in K_J: v=\pm1 (div(u)\overset{>}{<}0), div(v)\in[-1,1] div(u)=0 \} $ | $\{-\nabla v: v\in H_0^1\}$  |$-\nabla argmin\{\|u\|_{L^2}^2 v\in H_0^1, v=\pm 1 \|div u\| a.e. \}$| $J(u)=\sup_{C_c^\infty(R^d,[-1,1])} $