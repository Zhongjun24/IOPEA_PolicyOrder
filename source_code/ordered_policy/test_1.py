import math
from decimal import Decimal, getcontext
from itertools import product
import numpy as np
import pandas

def multinomial_coeff(indices):
    """
    Compute the multinomial coefficient (sum(indices))! / (i1! i2! ... ik!)
    """
    total = sum(indices)
    coeff = math.factorial(total)
    for idx in indices:
        coeff //= math.factorial(idx)
    return Decimal(coeff)

def generate_indices(k, target):
    """
    Generate all non-negative integer tuples (i1, i2, ..., ik)
    such that i1 + 2*i2 + ... + k*ik == target
    """
    if k == 1:
        if target % 1 == 0 and target >= 0:
            yield (target,)
        return

    for i in range(target // k + 1):
        for rest in generate_indices(k-1, target - i*k):
            yield rest + (i,)

def consecutive_success_pmf(k, p, x, precision=50):
    """
    Compute f_k(x) using the new multinomial sum formula, with high precision
    """
    if x < k:
        return 0.0

    getcontext().prec = precision
    p = Decimal(p)
    one_minus_p = Decimal(1) - p
    result = Decimal(0.0)

    target = x - k

    for indices in generate_indices(k, target):
        total_indices = sum(indices)
        coeff = multinomial_coeff(indices)
        term = coeff * (p**x) * ((one_minus_p/p)**total_indices)
        result += term

    return float(result)

import numpy as np

def true_consecutive_success_pmf(k, p, max_steps=1000):
    """
    Correctly compute the PMF f_k(x) where:
    - k is the number of consecutive successes,
    - p is the success probability,
    - max_steps is the maximum number of trials we simulate.

    Returns:
    - f: numpy array of length (max_steps),
         where f[i-1] = probability first hitting at step i
    """
    dp = np.zeros((max_steps + 1, k))  # dp[i][s]: probability at step i in state s
    f = np.zeros(max_steps)            # f[i-1]: probability hitting at step i

    dp[0][0] = 1.0  # Start with 0 consecutive successes

    for i in range(1, max_steps + 1):
        for s in range(k):
            if s == k-1:
                # From state k-1, success reaches target
                f[i-1] += dp[i-1][k-1] * p
            else:
                dp[i][s+1] += dp[i-1][s] * p
            dp[i][0] += dp[i-1][s] * (1 - p)

    return f  # now f[0] = hitting at step 1, f[1] at step 2, etc

def generate_z_sequence(k_prime, length):
    """
    Generate z_i sequence up to given length.
    Every k' elements, value multiplied by 3.
    """
    z = []
    value = 3
    while len(z) < length:
        z.extend([value] * k_prime)
        value *= 3
    return np.array(z[:length])

def compute_weighted_sum(k, k_prime, p, max_x):
    """
    Compute sum_{i=1}^max_x z_i * f_k(i)
    """
    f = true_consecutive_success_pmf(k, p, max_steps=max_x)
    z_seq = generate_z_sequence(k_prime, max_x)
    total = np.sum(z_seq * f)
    return total

# ===== Example usage =====
k = 3       # number of consecutive successes
k_prime = 2 # how many times each number repeats
p = 0.95    # success probability
max_x_list = [20,30,40,50,60,70,80,90,100,110]  # list of truncation points

# Compute results for each max_x
result = [compute_weighted_sum(k, k_prime, p, v) for v in max_x_list]

# Output results
print(f"Weighted sums = {result}")

# Optionally, compute growth ratios between consecutive results
growth_ratios = [(result[i+1]-result[i])/(result[i]-result[i-1]) for i in range(len(result)-2)]
print(f"Growth ratios = {growth_ratios}")
