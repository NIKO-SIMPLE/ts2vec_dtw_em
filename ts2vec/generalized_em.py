"""Generalized EM_old.py trajectory fusion, implemented from Equations (8) and (9).

Input file layout (12345.txt): each row stores N trajectories consecutively,
and each trajectory contains T time steps.  With N=10 and T=50, each row has
500 values.  This program uses file row 2 as x (zero-based index 1) and row 3
as the one-dimensional observations f (zero-based index 2).
"""

from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


EPS = 1e-8


def row_to_trajectories(row, n_trajectories, n_steps):
    """Convert a flattened row to (N, T): one trajectory per row."""
    row = np.asarray(row, dtype=np.float64).ravel()
    expected_size = n_trajectories * n_steps
    if row.size != expected_size:
        raise ValueError(f"Expected {expected_size} values, got {row.size}.")
    return row.reshape(n_trajectories, n_steps)


def log_gaussian_pdf_by_time(y, mu, sigma):
    """Return log N(y_t | mu, sigma) for all t.

    y:     (D, T)
    mu:    (D,)
    sigma: (D, D)
    return: (T,)
    """
    dimension = y.shape[0]
    sigma = sigma + EPS * np.eye(dimension)

    sign, log_determinant = np.linalg.slogdet(sigma)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")

    difference = y - mu[:, None]                         # (D, T)
    inverse_sigma = np.linalg.inv(sigma)
    mahalanobis = np.einsum("dt,de,et->t", difference, inverse_sigma, difference)

    return -0.5 * (
        dimension * np.log(2.0 * np.pi)
        + log_determinant
        + mahalanobis
    )


def e_step(x, z, pi, mu, sigma):
    """Equation (8): compute gamma_{k,t}.

    x:     (Dx, T)
    z:     (Dz, T)
    pi:    (K,)
    mu:    (K, Dx + Dz)
    sigma: (K, Dx + Dz, Dx + Dz)

    returns gamma with shape (K, T).
    """
    y = np.vstack((x, z))                                 # (D, T)
    component_count = pi.size

    log_numerator = np.empty((component_count, y.shape[1]))
    for k in range(component_count):
        log_numerator[k] = (
            np.log(pi[k] + EPS)
            + log_gaussian_pdf_by_time(y, mu[k], sigma[k])
        )

    # This is the normalized form of the fraction in Equation (8).
    gamma = np.exp(log_numerator - logsumexp(log_numerator, axis=0, keepdims=True))
    return gamma


def m_step_parameters(x, z, gamma, f, s):
    """Update pi, mu, Sigma, and sigma_f^2 in Equation (9).

    x:     (Dx, T)
    z:     (Dz, T)
    gamma: (K, T)
    f:     (Dz, N, T)
    s:     (N, T)
    """
    y = np.vstack((x, z))                                 # (D, T)
    component_count, _ = gamma.shape
    dimension = y.shape[0]

    # N_k and pi_k
    nk = np.sum(gamma, axis=1) + EPS                       # (K,)
    pi = nk / np.sum(nk)                                   # (K,)

    # mu_k = sum_t gamma_{k,t} y_t / N_k
    mu = np.einsum("kt,dt->kd", gamma, y) / nk[:, None]   # (K, D)

    # Sigma_k = sum_t gamma_{k,t}(y_t-mu_k)(y_t-mu_k)^T / N_k
    difference = y[None, :, :] - mu[:, :, None]            # (K, D, T)
    sigma = np.einsum("kt,kdt,ket->kde", gamma, difference, difference)
    sigma /= nk[:, None, None]
    sigma += 1e-6 * np.eye(dimension)[None, :, :]

    # sigma_f^2 = sum_{t,n} s_{n,t} ||f_{n,t}-z_t||^2 / sum_{t,n} s_{n,t}
    residual = f - z[:, None, :]                            # (Dz, N, T)
    squared_error = np.sum(residual ** 2, axis=0)            # (N, T)
    sigma_f2 = np.sum(s * squared_error) / (np.sum(s) + EPS)
    sigma_f2 = max(float(sigma_f2), EPS)

    return nk, pi, mu, sigma, sigma_f2


def estimate_z_map(x, z_initial, gamma, mu, sigma, f, s, sigma_f2, lambda_s):
    """Equation (9): MAP estimate of the complete trajectory z_{1:T}.

    This function maximizes the exact three-term objective in the paper:
    GMM log probability - weighted observation error - temporal smoothness.
    """
    z_dimension, n_steps = z_initial.shape
    component_count = gamma.shape[0]

    def negative_map_objective(z_flat):
        z = z_flat.reshape(z_dimension, n_steps)
        y = np.vstack((x, z))

        # sum_t sum_k gamma_{k,t} log N((x_t,z_t)^T | mu_k, Sigma_k)
        gmm_term = 0.0
        for k in range(component_count):
            gmm_term += np.sum(
                gamma[k] * log_gaussian_pdf_by_time(y, mu[k], sigma[k])
            )

        # sum_t sum_n s_{n,t} ||f_{n,t} - z_t||^2 / (2 sigma_f^2)
        residual = f - z[:, None, :]
        squared_error = np.sum(residual ** 2, axis=0)
        observation_term = np.sum(s * squared_error) / (2.0 * sigma_f2)

        # lambda_s / 2 * sum_{t=2}^T ||z_t-z_{t-1}||^2
        smoothness_term = lambda_s * np.sum((z[:, 1:] - z[:, :-1]) ** 2) / 2.0

        # scipy minimizes, while Equation (9) maximizes.
        return -gmm_term + observation_term + smoothness_term

    solution = minimize(
        negative_map_objective,
        z_initial.ravel(),
        method="L-BFGS-B",
    )
    if not solution.success:
        print("Warning: z optimization did not fully converge:", solution.message)

    return solution.x.reshape(z_dimension, n_steps)


def initialize_gmm(x, z, component_count, random_state=42):
    """Initialize pi, mu and Sigma from the current (x,z) trajectory."""
    y = np.vstack((x, z))
    _, n_steps = y.shape
    if not 1 <= component_count <= n_steps:
        raise ValueError("component_count must be between 1 and T.")

    rng = np.random.default_rng(random_state)
    initial_times = rng.choice(n_steps, size=component_count, replace=False)
    mu = y[:, initial_times].T
    pi = np.full(component_count, 1.0 / component_count)

    base_sigma = np.atleast_2d(np.cov(y)) + 1e-4 * np.eye(y.shape[0])
    sigma = np.repeat(base_sigma[None, :, :], component_count, axis=0)
    return pi, mu, sigma


def generalized_em(x, f, s, component_count=10, lambda_s=20.0,
                   max_iter=100, tolerance=1e-5, random_state=42):
    """Run generalized EM_old.py and return the fused trajectory and model parameters.

    x: (Dx, T), f: (Dz, N, T), s: (N, T)
    """
    x = np.asarray(x, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    if x.ndim != 2 or f.ndim != 3 or s.shape != f.shape[1:]:
        raise ValueError("Expected x=(Dx,T), f=(Dz,N,T), and s=(N,T).")
    if x.shape[1] != f.shape[2] or np.any(s < 0):
        raise ValueError("x/f time dimensions must match and s must be non-negative.")

    # Equation (7): z_t^(0) = sum_i s_{i,t} f_{i,t}.
    # Normalize weights per time step to guarantee sum_i s_{i,t}=1.
    s = s / (np.sum(s, axis=0, keepdims=True) + EPS)
    z = np.sum(f * s[None, :, :], axis=1)                   # (Dz, T)

    pi, mu, sigma = initialize_gmm(x, z, component_count, random_state)
    sigma_f2 = 1.0

    for iteration in range(1, max_iter + 1):
        previous_z = z.copy()

        # a) E-step
        gamma = e_step(x, z, pi, mu, sigma)

        # b) M-step: update Theta and sigma_f
        nk, pi, mu, sigma, sigma_f2 = m_step_parameters(x, z, gamma, f, s)

        # b) M-step: MAP update of z
        z = estimate_z_map(x, z, gamma, mu, sigma, f, s, sigma_f2, lambda_s)

        z_change = np.linalg.norm(z - previous_z)
        print(f"Iteration {iteration:03d}: |Δz|={z_change:.8f}, sigma_f^2={sigma_f2:.8f}")

        if z_change < tolerance:
            print("Converged.")
            break

    # Align returned parameters with the final z.
    gamma = e_step(x, z, pi, mu, sigma)
    nk, pi, mu, sigma, sigma_f2 = m_step_parameters(x, z, gamma, f, s)

    return {
        "z": z,                 # (Dz, T)
        "gamma": gamma,         # (K, T)
        "nk": nk,               # (K,)
        "pi": pi,               # (K,)
        "mu": mu,               # (K, Dx+Dz)
        "sigma": sigma,         # (K, Dx+Dz, Dx+Dz)
        "sigma_f2": sigma_f2,
        "s": s,                 # (N, T)
        "iterations": iteration,
    }


def quality_weights(observation_row, n_trajectories, n_steps):
    """Use the existing DTW + TS2Vec implementation to calculate s_{n,t}."""
    from calculate_s_it import calculate_s_it

    return calculate_s_it(
        np.asarray(observation_row, dtype=np.float64).reshape(1, -1),
        demoNum=n_trajectories,
        demoLen=n_steps,
        alpha_m=0.5,
        alpha_n=0.5,
        alpha_c=0.8,
        lambda_c=0.5,
    )


if __name__ == "__main__":
    N = 10
    T = 50
    K = 10
    LAMBDA_S = 10.0
    USE_QUALITY_WEIGHTS = True

    folder = Path(__file__).resolve().parent
    raw = np.loadtxt(folder / "12345.txt", dtype=np.float64)

    # File row 2: known scalar x.  The 10 copies are identical, so take one.
    x = row_to_trajectories(raw[1], N, T)[0][None, :]       # (1, 50)

    # File row 3: ten scalar observations f_{n,t}.
    f = row_to_trajectories(raw[2], N, T)[None, :, :]       # (1, 10, 50)

    if USE_QUALITY_WEIGHTS:
        s = quality_weights(raw[2], N, T)                  # (10, 50)
    else:
        s = np.ones((N, T), dtype=np.float64) / N

    result = generalized_em(
        x=x,
        f=f,
        s=s,
        component_count=K,
        lambda_s=LAMBDA_S,
    )

    output_path = folder / "gem_z.txt"
    np.savetxt(output_path, result["z"].T, fmt="%.10f")
    print(f"Saved fused z to: {output_path}")
