# Enhancing portfolio optimization solutions
## Wisely encoding constrained combinatorial optimization problems on quantum devices
# Objective
Portfolio optimization problems usually come with a set of constraints, for example, in [1] two inequality constraints are applied to ensure the assets bought do not surpass a budget and to guarantee a minimum profit (see Sec. IV A of [1]). Usually, to encode these inequality constraints in quantum computers we require extra qubits (for the slack variables). This has two disadvantages, first the probability of finding optimal solutions decrease and second the depth of the circuits increases. To solve this problem, I use unbalanced penalization [2], a new encoding method that does not require those extra qubits to encode inequality constraints. The method has been used in the traveling salesman problem, the bin packing problem, and the knapsack problem. I extended the solution to work with portfolio optimization to see if this enhances the probability of finding optimal solutions reducing the circuit depth.

I solve the problem using QAOA, and VQE and compare it with the results from [1]. Additionally, results on real hardware are presented, and a visualization technique for the optimal solution and the circuit requirements.

[1] Herman, D., Shaydulin, R., Sun, Y., Chakrabarti, S., Hu, S., Minssen, P., Rattew, A., Yalovetzky, R., & Pistoia, M. (2022). Portfolio Optimization via Quantum Zeno Dynamics on a Quantum Processor. http://arxiv.org/abs/2209.15024

[2] Montanez-Barrera, A., Willsch, D., Maldonado-Romo, A. & Michielsen, K. (2022). Unbalanced penalization: A new approach to encode inequality constraints of combinatorial problems for quantum optimization algorithms. 23–25. http://arxiv.org/abs/2211.13914

# Introduction
Solving combinatorial optimization problems (COP) using quantum devices is a promising area of research with significant potential for practical applications, and it is expected to be one of the main use cases for early quantum computers. One of the main representatives of COP is portfolio optimization owes to its applicability to financial services. Modern portfolio optimization is modeled by a function that tries to increase the profit as possible while keeping the risk associated to buy a set of assets as low as possible, it is described by

$$f(x) = -\mu x + x^T \sigma x$$

where $x=\\{0,1\\}^n$ is a vector that represents the set of assets with value 1 if the asset is selected, $\mu$ is the expected return, and $\sigma$ is the covariance matrix associated with the risk. Different constraints can be imposed on this model depending on the requirements, between them the most important is the one for not exceeding the maximum budget.

$$ \sum_i^n c_i x_i \le Budget$$  
