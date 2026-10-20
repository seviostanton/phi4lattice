import jax.numpy as jnp
# want psi_ee = D_ee K phi  eq (24)
# D_2, decimation operator
# for 2d, we want every other site in both dimensions


def decimate_2d(psi: jnp.ndarray):
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
K_coeffs = {
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


def make_smoothing_kernel():
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
    for (dx, dy), coeff in K_coeffs.items():
        # symmetry for swap and sign flips
        for a, b in {(dx, dy), (dy, dx),
                     (-dx, dy), (-dy, dx),
                     (dx, -dy), (dy, -dx),
                     (-dx, -dy), (-dy, -dx)}:
            K = K.at[3 + a, 3 + b].set(coeff)
    return K


def apply_smoothing_kernel(phi: jnp.ndarray,
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
