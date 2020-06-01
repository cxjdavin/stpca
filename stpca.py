import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import sys

from itertools import permutations, combinations
from matplotlib import cm

# Returns the value Y(u, ... , u)
def test(Y, u, anchor=None):
    if anchor is None:
        dimensions = Y.shape + u.shape
    else:
        dimensions = Y.shape + u.shape + anchor.shape
    assert dimensions.count(dimensions[0]) == len(dimensions)

    if anchor is None:
        val = Y
        for i in range(len(Y.shape)):
            val = np.dot(val, u)
        return val
    else:
        val = np.dot(Y, u)
        for i in range(len(Y.shape)-1):
            val = np.dot(val, anchor)
        return val

# Inefficient computation of alphas. We actually compute ALL n coordiantes
# Returns a vector of length n
def get_alpha(Y, u):
    dimensions = Y.shape + u.shape
    assert dimensions.count(dimensions[0]) == len(dimensions)

    output = Y
    for i in range(len(Y.shape)-1):
        output = np.dot(output, u)
    return output

# Returns sign of non-zero number x
def sign(x):
    assert x != 0
    if x > 0:
        return 1
    else:
        return -1

# Returns 1-based indexing (if start from 0, cannot differentiate sign)
def signed_supp(x):
    supp = []
    for i in range(len(x)):
        if x[i] != 0:
            supp.append(sign(x[i]) * (i+1))
    return supp

# Returns 0-based indexing without signs
def unsigned_supp(x):
    return [abs(i)-1 for i in signed_supp(x)]

# Returns next maximizer conditioned on disjoint support from E
def get_maximizer(scores, E, idx):
    output = None
    while True:
        if idx == len(scores):
            output = [0]*len(scores[-1][1])
            break
        output = scores[idx][1]
        idx += 1
        if len(E.intersection(set(unsigned_supp(output)))) == 0:
            break
    assert output is not None
    return output, idx

# Generates random instance
# If no strength is given, we randomly generate it
# Note: Random strengths may not fulfill lambda_r >> beta * lambda_1 since we do not take in epsilon into account...
def generate(p, k, n, r, strengths=None):
    print("Generating with p={0}, n={1}, k={2}, r={3}".format(p,n,k,r))
    assert k <= np.sqrt(n*p)
    assert 1 <= k and k <= n
    assert p >= 2
    assert r >= 1
    assert k * r <= n
    tensor_dims = tuple([n]*p)

    if strengths is None:
        # Generate signal strengths where minimum strength is 1
        unscaled_strengths = sorted([np.random.random() for _ in range(r)], reverse=True)
        strengths = [x / unscaled_strengths[-1] for x in unscaled_strengths]
    else:
        assert(len(strengths) == r)

    # Generate planted signals
    signals = []
    X = np.zeros(tensor_dims)
    indices = set(range(n))
    for i in range(r):
        supp = np.random.choice(list(indices), k, replace=False)
        x = [0]*n
        for j in supp:
            indices.remove(j)
            if np.random.random() > 0.5:
                x[j] = 1 / np.sqrt(k)
            else:
                x[j] = -1 / np.sqrt(k)
        # Remark: np.linalg.norm(x,2) == 1 fails for some cases due to precision (e.g. norm is 0.99999...)
        assert len(x) == n and np.isclose(np.linalg.norm(x,2), 1) and len(signed_supp(x)) == k
        x = np.array(x)
        signals.append(x)

        # Add to signal tensor
        x_tensor = x
        for _ in range(p-1):
            x_tensor = np.multiply.outer(x_tensor,x)
        X += strengths[i] * x_tensor

    # Generate noise matrix
    W = np.random.normal(0,1, tensor_dims)

    for i in range(r):
        print("lambda_{0} = {1}, support = {2}".format(i+1, strengths[i], signed_supp(signals[i])))
    print()

    return strengths, signals, X, W

'''
Works for arbitrary p, t, k, n, r
Assumes that smallest signal strength is L
Notes:
    - Can only recover signals with sufficient large signal strengths
    - Maximization may fail if signal strength not strong enough. In that case, estimate with the 0 vector
'''
def solve(Y, L, p, t, k, n, r, epsilon=0.1, delta=0.1):
    print("Dimensions of observation Y: {0} -> {1} numbers".format(Y.shape, pow(n,p)))
    print("Solving instance with epsilon = {0} and delta = {1}".format(epsilon, delta))
    print("Signal strengths rescaled such that lambda_{0} = L = {1}".format(r,L))
    assert 1 <= t and t <= k and k <= n
    assert k <= np.sqrt(n * p)
    # Rough requirement on smallest lambda (may be off by constant factor)
    req = t / pow(epsilon,2) / pow(1-epsilon,2) * pow(k / (1-epsilon) / t,p) * np.log(k/(1-epsilon)/delta)
    if L*L < req:
        print("(Recommend L >> const * {0})".format(np.sqrt(req)))

    # Generate U_t
    Ut = []
    indices = list(range(n))
    for t_supp in combinations(indices, t):
        for i in range(pow(2,t)):
            test_vec = [0]*n
            for j in range(t):
                if (i >> j) & 1:
                    test_vec[t_supp[j]] = 1
                else:
                    test_vec[t_supp[j]] = -1
            Ut.append(np.array(test_vec) / np.sqrt(t))
    assert len(Ut) == scipy.special.binom(n,t) * pow(2,t)
    print("Generated U_{0} of size {1}".format(t, len(Ut)))

    # Sort U_t in descending Y(u,...,u) order
    scores = [(test(Y,u), u) for u in Ut]
    scores.sort(key=lambda x: x[0], reverse=True)
    idx = 0

    # Recover r signals
    alpha_tau = L / (2 * np.sqrt(t)) * pow(1-epsilon, p) * pow(t/k, p/2)
    E = set()
    recovered = []
    maximizer_trace = [] # Used for making slides
    for i in range(r):
        # Print to track progress
        if i == 0:
            print("Extracting {0}st signal: ".format(i+1), end="")
        elif i == 1:
            print("Extracting {0}nd signal: ".format(i+1), end="")
        elif i == 2:
            print("Extracting {0}rd signal: ".format(i+1), end="")
        else:
            print("Extracting {0}th signal: ".format(i+1), end="")

        # Initialize
        ok = True
        estimate = [0]*n
        estimate_trace = []

        # Find anchor u
        u, idx = get_maximizer(scores, E, idx)
        maximizer_trace.append(u)
        if np.count_nonzero(u) == 0: # Maximization fails
            ok = False

        if ok:
            # In thesis, I wrote as computing only the relevant coordinates of alpha
            # For ease of implementation, we compute the entire alpha vector once here
            alpha_u = get_alpha(Y,u)

            # Extract signal coordinates to store in E and update estimate
            for z in unsigned_supp(u):
                a_z = alpha_u[z]
                if abs(a_z) > alpha_tau and len(signed_supp(estimate)) < k - t:
                    estimate[z] = sign(a_z) / np.sqrt(k)
                    E.add(z)

            # Sort U_t in descending Y(u,...,u,v) order
            scores_u = [(test(Y,v,anchor=u), v) for v in Ut]
            scores_u.sort(key=lambda x: x[0], reverse=True)
            idx_u = 0

            # Loop until k-t signal coordinates recovered
            while len(signed_supp(estimate)) < k-t and ok:
                v, idx_u = get_maximizer(scores_u, E, idx_u)
                maximizer_trace.append(v)
                if np.count_nonzero(v) == 0: # Maximization fails
                    ok = False
                    print(v)
                    break

                # Extract signal coordinates to store in E and update estimate
                for z in unsigned_supp(v):
                    a_z = alpha_u[z]
                    if abs(a_z) > alpha_tau and len(signed_supp(estimate)) < k - t:
                        estimate[z] = sign(a_z) / np.sqrt(k)
                        E.add(z)

            # Final maximization for current signal
            v, idx_u = get_maximizer(scores_u, E, idx_u)
            maximizer_trace.append(v)
            if np.count_nonzero(v) == 0: # Maximization fails
                ok = False

            if ok:
                # Extract signal coordinates to store in E
                for z in unsigned_supp(v):
                    a_z = alpha_u[z]
                    # Always update estimate for final maximization
                    # Note: This may result in overlaps with other signals
                    estimate[z] = sign(a_z) / np.sqrt(k)
                    if abs(a_z) > alpha_tau:
                        E.add(z)
        assert len(signed_supp(estimate)) == k
        print("{0}".format(signed_supp(estimate)))
        recovered.append(np.array(estimate))
        maximizer_trace.append(None)
    return recovered, maximizer_trace

def evaluate_recovery(signals, recovered):
    assert len(signals) == len(recovered)
    r = len(signals)

    best_ordering = None
    best_score = -float('inf')
    for ordering in permutations(list(range(r))):
        score = 0
        for q in range(r):
            score += abs(np.dot(signals[q], recovered[ordering[q]]))
        if score > best_score:
            best_score = score
            best_ordering = ordering
    return best_ordering, best_score

if __name__ == "__main__":
    np.random.seed(42)

    # Read parameters
    p = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    r = int(sys.argv[4])
    L = float(sys.argv[5])
    t = int(sys.argv[6])
    print("Parameters: p = {0}, k = {1}, n = {2}, r = {3}, L = {4}, t = {5}".format(p,k,n,r,L,t))
    print()

    strengths, signals, X, W = generate(p,k,n,r) 

    Y = W + L*X
    recovered, maximizer_trace = solve(Y, L, p, t, k, n, r)
    print("Solved")
    print()

    print("Pairing up signals and recovered...")
    best_ordering, best_score = evaluate_recovery(signals, recovered)
    for q in range(r):
        planted = signals[q]
        recover = recovered[best_ordering[q]]
        print("Planted signed support  : {0}".format(signed_supp(planted)))
        print("Recovered signed support: {0}".format(signed_supp(recover)))
        print("Inner product           : {0}".format(np.dot(planted, recover)))
        #print()

