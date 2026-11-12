import jax.numpy as jnp
import jax
# want psi_ee = D_ee K phi  eq (24)
# D_2, decimation operator
# for 2d, we want every other site in both dimensions


def decimate_2d_paper(psi: jnp.ndarray):
    # psi could be batched (N, L, L) or just (L, L)
    if psi.ndim not in (2, 3):
        raise ValueError("psi must be 2D field or batch of 2D fields")

    # psi needs to be even in both spatial dimensions
    #
    if any(n % 2 != 0 for n in psi.shape[-2:]):
        raise ValueError(
            "psi must have even dimensions in both spatial directions")

    # slice
    # all rows, all colums, step size 2; array[start:stop:step]
    # ... keeps the batch dimension if it exists
    return psi[..., ::2, ::2]

# need smoothing kernel
# psi(vec[x]) = (Kphi)(vec[x]) = sum_r K(vec[r]) phi(vec[x] + vec[r]) eq (25)


# coefficients from table 1
# {(delta_x, delta_y): coefficient}; distance from the central site (0, 0)
K_coeffs_paper = {
    (0, 0): 0.888641822,
    (1, 0): 0.010508374,
    (1, 1): -0.066254410,
    (2, 0): 0.037116968,
    (2, 1): 0.022657096,
    (2, 2): -0.001649881,
    (3, 0): 0.020222068,
    (3, 1): 0.007299892,
    (3, 2): -0.003848342,
    (3, 3): -0.001693935,
}


def make_smoothing_kernel_paper():
    # K is a 7x7 kernel centered at (0, 0)
    # K is like [ [K(-3,-3), ..., K(-3,3)],
    #               .                .
    #               .                .
    #               .                .
    #            [K(0,-3), ..., K(0,3)],
    #               .                .
    #               .                .
    #               .                .
    #            [K(3,-3), ..., K(3,3)] ]
    K = jnp.zeros((7, 7))
    for (dx, dy), coeff in K_coeffs_paper.items():
        # symmetry for swap and sign flips
        for a, b in {(dx, dy), (dy, dx),
                     (-dx, dy), (-dy, dx),
                     (dx, -dy), (dy, -dx),
                     (-dx, -dy), (-dy, -dx)}:
            K = K.at[3 + a, 3 + b].set(coeff)
    return K


def apply_smoothing_kernel_paper(phi: jnp.ndarray,
                                 K: jnp.ndarray,
                                 spatial_axes: tuple):
    # phi could be batched (N, L, L) or just (L, L)
    if phi.ndim not in (2, 3):
        raise ValueError("phi must be 2D field or batch of 2D fields")
    psi = jnp.zeros_like(phi)
    for i in range(K.shape[0]):
        for j in range(K.shape[0]):
            phi_shifted = jnp.roll(phi, shift=(i - 3, j - 3),
                                   axis=spatial_axes)
            psi += K[i, j] * phi_shifted

    return psi


# def initialize_smoothing_kernel(b=2,
#                                 d=2,
#                                 eta=1/4,
#                                 offset_range=3,
#                                 seed = None):
#     # initialize K
#     if seed is None:
#         seed = jax.random.PRNGKey(0)

#     key = jax.random.PRNGKey(seed)
#     K_shape = (2 * offset_range + 1, 2 * offset_range + 1)
#     # ex) offset_range=3 -> K_shape=(7, 7) like paper

#     K = jax.random.randint(key, K_shape, -10*3, 10**3)
#     # enforce normalization
#     K = normalize_smoothing_kernel(K, b=b, d=d, eta=eta)
#     return K

def initialize_smoothing_kernel(b=2,
                                d=2,  # can get from len(spatial_axes)
                                eta=1 / 4,
                                max_r_i=3,
                                seed=None):
    # initialize K
    if seed is None:
        seed = jax.random.PRNGKey(0)

    key = jax.random.PRNGKey(seed)
    K_shape = (2 * max_r_i + 1, 2 * max_r_i + 1)

    init_vals = jax.random.normal(key, max_r_i * d + 1)
    K_indices = jnp.stack(jnp.indices(K_shape), axis=-1)
    d4_map = (abs(K_indices - max_r_i)).sum(axis=-1)
    # K = init_vals[d4_map]
    K = make_smothing_kernel(init_vals, d4_map)

    # ex) offset_range=3 -> K_shape=(7, 7) like paper

    # D4 symmetry gives upper triangle

    # K = jax.random.randint(key, K_shape, -10*3, 10**3)
    # enforce D4 symmetry
    # K_diag = jnp.diag(jnp.diag(K))
    # K_upper = jnp.triu(K, k=1)
    # K_lower = K_upper.transpose()
    # K = K_diag + K_upper + K_lower

    # get upper index if needed

    # enforce normalization
    K = normalize_smoothing_kernel(K, b=b, d=d, eta=eta)
    return K, init_vals, d4_map


def make_smothing_kernel(K_vals, d4_map):
    return K_vals[d4_map]


def normalize_smoothing_kernel(K,
                               b=2,  # blocking factor
                               d=2,  # spatial dimensions
                               eta=1 / 4  # critical scaling exponent
                               ):
    # assumes gaussian kinetic action term
    #    S = 1/2 int d^dx (gradphi)^2
    #    rescale: x'=x/b;
    #             phi'(x') = b^gamma phi(bx')
    #             grad = 1/b grad'
    #    S = 1/2int d^d(b x') (b^(-1-gamma) grad'phi'(x'))^2
    #      = 1/2 b^(d-2-2gamma) int d^dx' (grad'phi(x'))^2
    #    scale invariance requires d-2-2gamma = 0 at the critical point
    #    gamma = (d-2+eta)/2

    # paper uses normalization sum(K)=b^gamma
    gamma = (d - 2 + eta) / 2
    K = b**gamma * K / jnp.sum(K)
    assert jnp.isclose(jnp.sum(K), b**gamma)
    return K


# I think I may be able to use autocorrelation.py instead
def observable_covariance_tensor(obs_dict):
    """
    Compute the covariance tensor of the observables.
    Sample covariance:
        C_{ij}_S = 1/(N-dof) * sum_n (O_{in} - <O_i>)(O_{jn} - <O_j>)
    Covariance of estimated means:
        C_{ij}_M = C_{ij}_S / N_samples
    Assumes samples are independent.
    Args:
        obs_dict: Dictionary of observables.
                  Shape: (N observables, N samples)
    Returns:
        obs_covar_arr: Covariance tensor of the observables.
                       Shape: (N observables, N observables)
    """
    # # compute the mean of each observable in the dictionary
    # obs_means_dict = jax.tree.map(lambda obs: obs.mean(axis=0),
    #                               obs_dict)

    # # convert to jnp.array
    # obs_means_arr = jnp.array(jax.tree.leaves(obs_means_dict))

    # convert to jnp.array
    obs_arr = jnp.array(jax.tree.leaves(obs_dict))

    N_samples = obs_arr.shape[1]

    # compute the covariance of each pair of observables
    # shape: (N observables, N observables)
    obs_covar_arr = jnp.cov(obs_arr, rowvar=False) / N_samples
    return obs_covar_arr


def get_chi_K2(coarse_obs_dict,
               blocked_obs_dict):
    """
    Compute the chi-squared statistic for the difference between coarse and blocked observables.
    Args:
        coarse_obs_dict: Dictionary of coarse observables.
        blocked_obs_dict: Dictionary of blocked observables.
        shape = (N observables, N samples)
    Returns:
    """
    coarse_covar_arr = observable_covariance_tensor(coarse_obs_dict)
    blocked_covar_arr = observable_covariance_tensor(blocked_obs_dict)

    C_ab = coarse_covar_arr + blocked_covar_arr

    # compute the mean of each observable in the dictionary
    coarse_obs_means_dict = jax.tree.map(lambda obs: obs.mean(axis=0),
                                         coarse_obs_dict)
    coarse_obs_means_arr = jnp.array(jax.tree.leaves(coarse_obs_means_dict))

    blocked_obs_means_dict = jax.tree.map(lambda obs: obs.mean(axis=0),
                                          blocked_obs_dict)
    blocked_obs_means_arr = jnp.array(jax.tree.leaves(blocked_obs_means_dict))

    delta_obs_means = blocked_obs_means_arr - coarse_obs_means_arr

    # contraction
    chi_K2 = jnp.einsum("a,ab,b->",
                        delta_obs_means,
                        jnp.linalg.inv(C_ab),
                        delta_obs_means)
    return chi_K2


def apply_smoothing_kernel(phi: jnp.ndarray,
                           K: jnp.ndarray,
                           max_r_i: int,
                           spatial_axes: tuple):
    # phi could be batched (N, (L,)*d) or just (L,)*d
    if phi.ndim not in (len(spatial_axes), len(spatial_axes) + 1):
        raise ValueError(
            "phi must be a d-dimensional field or batch of d-dimensional fields")
    psi = jnp.zeros_like(phi)
    for index in jnp.ndindex(K.shape):
        shift = tuple(i - max_r_i for i in index)
        phi_shifted = jnp.roll(phi, shift=shift, axis=spatial_axes)
        psi += K[index] * phi_shifted

    return psi


def decimate(psi,
             d,
             b=2):
    # if d != psi.ndim:
    #     # batched = True

    # else:
    #     # batched = False

    # (...,) keeps batched dimension if present
    # (slice(None, None, b),) * d drops every b-th site for d dimensions
    phi_blocked = psi[(...,) + (slice(None, None, b),) * d]
    return phi_blocked


def optimize_kernel(lat_coarse,
                    # coarse_obs_dict,
                    lat_fine,
                    # fine_obs_dict,
                    # K,
                    max_r_i,
                    b,
                    # init_vals,
                    d4_map,
                    K_vals,
                    measure_fns_dict,
                    epochs=100
                    ):
    """
    """
    # smooth fine lattice
    K0 = make_smothing_kernel(K_vals, d4_map)
    # phi_fine = lat_fine.phi_x
    # psi = apply_smoothing_kernel(phi_fine,
    #                              K,
    #                              max_r_i,
    #                              lat_fine.spatial_axes)

    # decimate smoothed lattice
    # phi_blocked = decimate(psi,
    #                        d=len(lat_fine.spatial_axes),
    #                        b=b
    #                        )

    def get_obs_dict(phi_x):
        obs_dict = {}
        # compute observables
        for name, fn in measure_fns_dict.items():
            obs_dict[name] = fn(phi_x)
        return obs_dict

    # blocked_obs_dict = get_obs_dict(phi_blocked)
    coarse_obs_dict = get_obs_dict(lat_coarse.phi_x)

    # chi_K2 = get_chi_K2(coarse_obs_dict,
    #                     blocked_obs_dict)
    optimized_K_dict = {}
    optimized_K_dict["d4_map"] = d4_map

    def loss_fn(K_vals):
        optimized_K_dict["K_vals"] = K_vals
        K = make_smothing_kernel(K_vals, d4_map)
        optimized_K_dict["K"] = K

        psi = apply_smoothing_kernel(lat_fine.phi_x,
                                     K,
                                     max_r_i,
                                     lat_fine.spatial_axes)
        optimized_K_dict["psi"] = psi

        phi_blocked = decimate(psi,
                               d=len(lat_fine.spatial_axes),
                               b=b)
        optimized_K_dict["phi_blocked"] = phi_blocked

        blocked_obs_dict = get_obs_dict(phi_blocked)
        optimized_K_dict["blocked_obs_dict"] = blocked_obs_dict

        chi_K2 = get_chi_K2(coarse_obs_dict,
                            blocked_obs_dict)
        optimized_K_dict["chi_K2(loss)"] = chi_K2
        return chi_K2

    def grad_descent(K_vals):
        grad_K_vals = jax.grad(loss_fn)(K_vals)
        K_vals = K_vals - grad_K_vals
        return K_vals

    optimized_K_vals_arr = jax.lax.scan(grad_descent,
                                        K_vals,
                                        length=epochs)

    optimized_K = optimized_K_dict["K"][jnp.argmin(
        optimized_K_dict["chi_K2(loss)"])]
    optimized_psi_blocked = apply_smoothing_kernel(lat_fine.phi_x,
                                                   optimized_K,
                                                   max_r_i,
                                                   lat_fine.spatial_axes)
    optimized_phi_blocked = decimate(optimized_psi_blocked,
                                     d=len(lat_fine.spatial_axes),
                                     b=b)
    optimized_K_dict["optimized_K"] = optimized_K
    optimized_K_dict["optimized_psi_blocked"] = optimized_psi_blocked
    optimized_K_dict["optimized_phi_blocked"] = optimized_phi_blocked

    return optimized_K_dict
