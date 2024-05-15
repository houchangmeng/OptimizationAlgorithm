# OptimizationAlgorithm

The most common optimization algorithm, implemented by Eigen.
___

## Requirement

python: matplotlib  
c++: Eigen
___

### Clone and bulid

```{bash}
git clone git@github.com:houchangmeng/OptimizationAlgorithm.git
cd OptimizationAlgorithm/
build.bash
```

### Run examples

```{bash}
./bin/example 
./bin/example1
./bin/example2
```

___

## Algorithm

### Unconstraint

* GradientDescend
* GaussNewton
* QuasiNewton_BFGS
* QuasiNewton_DFP
* ConjugateGradient

### Constraint

* AugmentedLagrangian
* InteriorPoint

___

## Examples

### Remark
>
> Equality constraints can be written as inequality constraints.

$$
\begin{aligned}
h(\mathbf{x}) = \mathbf{g} \\
\rightarrow \qquad \qquad \quad \mathbf{g} \leq h(\mathbf{x}) \leq \mathbf{g} \\
\rightarrow h(\mathbf{x})  \leq \mathbf{g}, \quad -h(\mathbf{x}) \leq -\mathbf{g}
\end{aligned}
$$
___

#### example.cc

$$
\begin{aligned}
    \textnormal{min} \qquad f(\mathbf{x})= (\mathbf{x-v}^{T})\mathbf{Q} (\mathbf{x-v}) \\
    \textnormal{s.t.} \qquad \mathbf{Gx} \leq \mathbf{b}
\end{aligned}
$$

![example](example.gif)
___

#### example1.cc

$$
\begin{aligned}
    \textnormal{min} \qquad f(\mathbf{x})= \mathbf{x}[0] * \mathbf{x}[1] \\
    \textnormal{s.t.} \qquad \mathbf{Gx} \leq \mathbf{b} \\
\end{aligned}
$$

![example](example1.gif)
___

#### example2.cc

$$
\begin{aligned}
    \textnormal{min} \qquad f(\mathbf{x})= \sin{(\mathbf{x}[0])} * \cos{((\mathbf{x}[1]))} \\
    \textnormal{s.t.} \qquad \mathbf{Gx} \leq \mathbf{b} \\
\end{aligned}
$$

![example](example2.gif)

## Reference

[bilibili](https://www.bilibili.com/video/BV1m7411u72b/?spm_id_from=333.337.search-card.all.click&vd_source=19f665b7702c6f2ce3b93bfe2d3cbcb2)

[Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

[CMU 16 745](https://github.com/Optimal-Control-16-745/)

[autodiff](https://github.com/autodiff/autodiff)
