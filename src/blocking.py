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

def make_orbit_map(d, max_r_i):
    K_shape = (2 * max_r_i + 1,)*d
    K_indices = jnp.stack(jnp.indices(K_shape), axis=-1)
    # finds the abs offset from the center of the orbit/K
    offsets = jnp.abs(K_indices - max_r_i)

    # sort offsets so we can find uniques
    sorted_offsets = jnp.sort(offsets, axis=-1)

    # find unique offsets
    unique_offsets, inverse_arr, multiplicities = jnp.unique(
        sorted_offsets.reshape(-1, d),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    d4_map = inverse_arr.reshape(K_shape)
    n_unique_orbits = unique_offsets.shape[0]
    
    return d4_map, n_unique_orbits, multiplicities




def initialize_smoothing_kernel(b=2,
                                d=2,  # can get from len(spatial_axes)
                                eta=1 / 4,
                                max_r_i=3,
                                seed=0):
    # initialize K
    # if seed is None:
    #     key = jax.random.PRNGKey(0)

    key = jax.random.PRNGKey(seed)
    # K_shape = (2 * max_r_i + 1,)*d
    # this made manhattan distance which is more restrictive
    #   init_vals = jax.random.normal(key, max_r_i * d + 1)
    #   K_indices = jnp.stack(jnp.indices(K_shape), axis=-1)
    #   d4_map = (abs(K_indices - max_r_i)).sum(axis=-1)
    #   K = init_vals[d4_map]

    d4_map, n_unique_orbits, multiplicities = make_orbit_map(d, max_r_i)
    # make a random initialization for the non-central orbit values
    init_orbit_vals = jax.random.normal(key,
                                        shape=(n_unique_orbits - 1,))/1000
    # K_0 = 0. # central orbit value
    # init_vals = jnp.concatenate([jnp.array([K_0]), init_theta])
    

    K, K_vals = make_smothing_kernel(init_orbit_vals,
                                     d4_map, multiplicities,
                                     b=b, d=d, eta=eta)

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
    # K = normalize_smoothing_kernel(K, b=b, d=d, eta=eta)
    return K, K_vals, d4_map, multiplicities


def make_smothing_kernel(orbit_vals, d4_map, multiplicities, b=2, d=2, eta=1 / 4):
    # force normalization condition by fixing central coefficient
    #   sum(K) = b^gamma
    #   K_0 = b^gamma - sum_{i != 0} K_i
    K_0 = b ** ((d - 2 + eta) / 2) - jnp.dot(orbit_vals, multiplicities[1:])

    K_vals = jnp.concatenate([jnp.array([K_0]), orbit_vals])
    K = K_vals[d4_map]
    return K, K_vals


# def normalize_smoothing_kernel(K,
#                                b=2,  # blocking factor
#                                d=2,  # spatial dimensions
#                                eta=1 / 4  # critical scaling exponent
#                                ):
#     # assumes gaussian kinetic action term
#     #    S = 1/2 int d^dx (gradphi)^2
#     #    rescale: x'=x/b;
#     #             phi'(x') = b^gamma phi(bx')
#     #             grad = 1/b grad'
#     #    S = 1/2int d^d(b x') (b^(-1-gamma) grad'phi'(x'))^2
#     #      = 1/2 b^(d-2-2gamma) int d^dx' (grad'phi(x'))^2
#     #    scale invariance requires d-2-2gamma = 0 at the critical point
#     #    gamma = (d-2+eta)/2

#     # this approach fails to keep the normalization during grad descent
#     # # paper uses normalization sum(K)=b^gamma
#     # gamma = (d - 2 + eta) / 2
#     # K = b**gamma * K / jnp.sum(K)
#     # # assert jnp.isclose(jnp.sum(K), b**gamma)


#     return K


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
    # obs_covar_arr = jnp.cov(obs_arr, rowvar=False) / N_samples
    obs_covar_arr = jnp.cov(obs_arr, rowvar=True) / N_samples

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
    # psi = jnp.zeros_like(phi)
    # for index in jnp.ndindex(K.shape):
    #     shift = tuple(i - max_r_i for i in index)
    #     phi_shifted = jnp.roll(phi, shift=shift, axis=spatial_axes)
    #     psi += K[index] * phi_shifted
    d = K.ndim
    # can use phi.shape[-d:] to get the spatial shape
    #   ex) phi.shape = (N, L, L, L) -> spatial_shape = (L, L, L) for d = 3
    spatial_shape = phi.shape[-d:]

    # # get padding for K
    # #   conv 2 arr and find diff
    # #   for spat_shap (32,32) & k_shape (7,7) 
    # #     size_diff = (25,25)
    # size_diff = jnp.array(spatial_shape) - jnp.array(K.shape) 
    # # for spat_shap (32,32) & k_shape (7,7) 
    # #   pad_width = array([[0, 25], [0, 25]])
    # pad_width = jnp.stack([jnp.zeros(d, dtype=int),size_diff],
    #                       axis=1)

    # get padding for K
    #    gives pad_width = ((13, 12), (13, 12))
    #       for spatial_shape = (32, 32) and K.shape = (7, 7)
    #    each tuple in pad width gives the padding before and after each spatial axis
    #       ex) add 13 zeros before and 12 zeros after each spatial axis
    pad_width = tuple(((n//2-k//2), n-k-(n//2-k//2)) for n, k in zip(spatial_shape, K.shape))
    K_padded = jnp.pad(K, pad_width)

    # shift the centered kernel so its origin is at index 0 for FFT convolution
    K_padded = jnp.fft.ifftshift(K_padded)

    # transform the padded kernel and field to Fourier space
    K_p = jnp.fft.fftn(K_padded)
    phi_p = jnp.fft.fftn(phi, axes=spatial_axes)

    # convolution becomes elementwise multiplication in Fourier space
    #  K(r) = sum_p K(p) exp(i r.p)  & K(-r) = K(r)
    #  phi(x+r) = sum_p phi(p) exp(i(x+r).p)
    #  psi(x) = sum_r K(r) phi(x+r) 
    #         = sum_r K(r) [sum_p phi(p) exp(i(x+r).p)]
    #         = sum_p phi(p) exp(i x.p) [sum_r K(r) exp(i r.p)]
    #         = sum_p phi(p) exp(i x.p) K(-p)
    #         = sum_p phi(p) exp(i x.p) K(p) from K(-r) = K(r)
    #  psi(q) = sum_x psi(x) exp(-i x.q)
    #         = sum_xsum_p phi(p) exp(i x.p) K(p) exp(-i x.q)
    #         = sum_p phi(p) K(p) sum_x exp(i x.(p-q))
    #         = sum_p phi(p) K(p) kroneckler(p,q)
    #         = phi(q) K(q)

    psi_p = phi_p * K_p

    # transform the convolved field back to real space
    #   should be real but .real() takes real part if complex
    psi = jnp.fft.ifftn(psi_p, axes=spatial_axes).real

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
                    multiplicities,
                    # orbit_vals,
                    measure_fns_dict,
                    epochs=100,
                    lr=0.1
                    ):
    """
    """
    # smooth fine lattice
    # K = make_smothing_kernel(orbit_vals, d4_map, b=b, d=len(lat_fine.spatial_axes))
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

    def loss_fn(orbit_vals):
        # optimized_K_dict["K_vals"] = K_vals
        # orbit_vals = K_vals[1:]
        K, K_vals = make_smothing_kernel(orbit_vals,
                                         d4_map,
                                         multiplicities,
                                         b=b,
                                         d=len(lat_fine.spatial_axes))
        # optimized_K_dict["K"] = K

        psi = apply_smoothing_kernel(lat_fine.phi_x,
                                     K,
                                     max_r_i,
                                     lat_fine.spatial_axes)
        # optimized_K_dict["psi"] = psi

        phi_blocked = decimate(psi,
                               d=len(lat_fine.spatial_axes),
                               b=b)
        # optimized_K_dict["phi_blocked"] = phi_blocked

        blocked_obs_dict = get_obs_dict(phi_blocked)
        # optimized_K_dict["blocked_obs_dict"] = blocked_obs_dict

        chi_K2 = get_chi_K2(coarse_obs_dict,
                            blocked_obs_dict)
        # optimized_K_dict["chi_K2(loss)"] = chi_K2
        return chi_K2

    def grad_descent(orbit_vals, _):
        loss, grad_K = jax.value_and_grad(loss_fn)(orbit_vals)
        new_orbit_vals = orbit_vals - lr * grad_K
        return new_orbit_vals, (orbit_vals, loss)

    final_orbit_vals, (orbit_vals_history, loss_history) = jax.lax.scan(grad_descent,
                                        K_vals[1:],
                                        xs = None,
                                        length=epochs)
    
    # evaluate final updated coefficients
    orbit_vals_history = jnp.concatenate([orbit_vals_history, final_orbit_vals[None, :]],
                                     axis=0)
    K_history, K_vals_history = jax.vmap(
        lambda orbit_vals: make_smothing_kernel(
            orbit_vals,
            d4_map,
            multiplicities,
            b=b,
            d=len(lat_fine.spatial_axes)
            )
            )(orbit_vals_history)

    loss_history = jnp.concatenate([loss_history,
                                    jax.value_and_grad(loss_fn)(final_orbit_vals)[0][None]],
                                    axis=0)

    # find coefficients that minimize the loss (chi_K^2)
    #    first require invertible K
    invertible_mask = jax.vmap(invertible_kernel)(K_history)
    #    then find the minimum loss among invertible kernels
    best_idx = jnp.argmin(jnp.where(invertible_mask, loss_history, jnp.inf))
    if not invertible_mask[best_idx]:
        # invertible_mask[best_idx] gives False if no invertible kernel was found
        raise ValueError("No invertible kernel found")
    
    best_orbit_vals = orbit_vals_history[best_idx]
    optimized_K_dict["best_orbit_vals"] = best_orbit_vals

    # best kernel
    best_K, best_K_vals = make_smothing_kernel(best_orbit_vals,
                                  d4_map,
                                  multiplicities,
                                  b=b,
                                  d=len(lat_fine.spatial_axes))
    
    optimized_K_dict["best_K"] = best_K
    optimized_K_dict["best_K_vals"] = best_K_vals

    
    best_psi_blocked = apply_smoothing_kernel(lat_fine.phi_x,
                                                   best_K,
                                                   max_r_i,
                                                   lat_fine.spatial_axes)
    best_phi_blocked = decimate(best_psi_blocked,
                                     d=len(lat_fine.spatial_axes),
                                     b=b)
    # optimized_K_dict["optimized_K"] = best_K
    optimized_K_dict["optimized_psi_blocked"] = best_psi_blocked
    optimized_K_dict["optimized_phi_blocked"] = best_phi_blocked
    optimized_K_dict["K_vals_history"] = K_vals_history
    optimized_K_dict["K_history"] = K_history
    optimized_K_dict["orbit_vals_history"] = orbit_vals_history
    optimized_K_dict["loss_history"] = loss_history
    optimized_K_dict["epochs"] = epochs
    optimized_K_dict["lr"] = lr
    return optimized_K_dict


def invertible_kernel(K,
                      threshold=1e-8) :
    # I think invertible K has a non-zero determinant
    #    as long as it is square
    return jnp.linalg.det(K) > threshold


# def get_best():
