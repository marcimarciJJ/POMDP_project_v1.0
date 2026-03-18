# Theoretical Background

## Partially Observable Markov Decision Processes (POMDPs)

### Definition

A POMDP is defined by the 7-tuple $(S, A, T, R, \Omega, O, \gamma)$:

- **S**: finite state space
- **A**: finite action space
- **T**: transition function $T(s, a, s') = P(s'|s, a)$
- **R**: reward function $R(a, s) \in \mathbb{R}$
- **$\Omega$**: finite observation space
- **O**: observation function $O(o, s', a) = P(o|s', a)$
- **$\gamma$**: discount factor $\in (0, 1)$

### Belief State

Since states are not directly observable, the agent maintains a **belief state** $b$: a probability distribution over $S$.

**Belief update** (Bayes' rule):
$$b'(s') = \eta \cdot O(o, s', a) \sum_{s \in S} T(s, a, s') \cdot b(s)$$

where $\eta$ is a normalizing constant.

### Optimal Value Function

The optimal value function over beliefs:
$$V^*(b) = \max_{\pi} \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(a_t, s_t) \Big| b_0 = b, \pi\right]$$

For finite-horizon POMDPs, $V^*(b)$ is piecewise-linear and convex (PWLC) in $b$.

---

## QMDP Algorithm

**Idea**: Ignore the partial observability and solve the underlying fully-observable MDP.

**Value Iteration** (standard MDP):
$$V(s) \leftarrow \max_{a} \left[ R(a, s) + \gamma \sum_{s'} T(s, a, s') V(s') \right]$$

**Policy under belief** $b$:
$$\pi(b) = \arg\max_a \sum_s b(s) Q(s, a)$$

**Properties**:
- Fast to compute (same as MDP value iteration)
- Optimistic: ignores information gain
- Works well when states become fully observable quickly

---

## PBVI Algorithm (Point-Based Value Iteration)

**Idea**: Represent $V^*(b)$ as a set of $\alpha$-vectors. Each $\alpha$-vector defines a hyperplane $V(b) = \alpha \cdot b$.

**Alpha vector backup** for action $a$ at belief $b$:
$$\alpha^{a,*}_s = R(a, s) + \gamma \sum_{o \in \Omega} \arg\max_{\alpha' \in \Gamma} \sum_{s'} O(o, s', a) T(s, a, s') \alpha'_{s'}$$

**Algorithm**:
1. Sample belief points $B = \{b_1, \ldots, b_K\}$
2. Initialize $\Gamma = \{\mathbf{0}\}$
3. For each $b \in B$: compute best $\alpha$-vector backup
4. Repeat until convergence

**Properties**:
- Better belief-space coverage than QMDP
- Polynomial complexity per iteration
- Approximates true optimal POMDP value function

---

## Convergence

Both algorithms track **Bellman error**:
$$\delta = \max_s |V_{k+1}(s) - V_k(s)|$$

Convergence is declared when $\delta < \epsilon$ (typically $\epsilon = 10^{-6}$).

The **J(b0)** metric tracks expected value under the initial belief:
$$J(b_0) = \sum_s b_0(s) V(s)$$
