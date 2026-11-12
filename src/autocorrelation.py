import jax
import jax.numpy as jnp
from typing import Callable


def autocovariance(obs1: jnp.ndarray,
                   obs2: jnp.ndarray = None,
                   t_max: int = None) -> jnp.ndarray:
    """Compute the autocovariance of the given observables.

    Args:
        obs1 (jax.numpy.ndarray): Array of first set of observables, can be batched.
        obs2 (jax.numpy.ndarray): Array of second set of observables, can be batched.

    Returns:
        jax.numpy.ndarray: Autocovariance of the observables, shape (t_max+1, N_parallel_chains).
    """

    # using trajectory history, observables will
    # have shape like (N_trajectory steps, N parallel trajectories)
    # from lat.trajectory_history["observable"]shape
    N = obs1.shape[0]  # number of trajectory steps
    N = N - 1  # adjust for zero-based indexing

    if t_max is None:
        # if t_max is not provided, set it to N//2
        #  t_max = N//2  # worried this will cut too soon
        t_max = N  # leave as N
    if not isinstance(t_max, int):
        raise ValueError("t_max must be an integer")
    if t_max < 0 or t_max >= N:
        raise ValueError("t_max must be in the range [0, N-1]")
    if N <= 2:
        raise ValueError("Number of trajectory steps N must be greater than 2")
    if obs1.ndim != 2:
        raise ValueError("obs1 must be a 2-dimensional array")
    obs1_mean = obs1.mean(axis=0, keepdims=True)
    obs1_residual = obs1 - obs1_mean
    if obs2 is None:
        obs2 = obs1
        obs2_mean = obs1_mean
        obs2_residual = obs2 - obs2_mean
    else:
        if obs2.ndim != 2 or obs2.shape != obs1.shape:
            raise ValueError(
                "Observables must have matching shapes "
                "(N_trajectories, N_parallel_chains)"
            )
        obs2_mean = obs2.mean(axis=0, keepdims=True)
        obs2_residual = obs2 - obs2_mean

    # gamma_1_2_t = jnp.sum(obs1_residual[:N-t] * obs2_residual[t:],
    #                       axis=0)/(N-t)

    # keep fixed shapes for jax.vmap
    def gamma_a_b_t(t):
        """autocorrelation function
            eq. (3.2) schaefer:
                G_alpha,beta = 1/(N-t) sum_i-1^N-t (
                    (obs_i_alpha - <obs_i_alpha>) * (obs_{i+t}_beta - <obs_{i+t}_beta>)
                    ) + O(1/N)
        """
        # upper limit N-t
        valid_filter = jnp.arange(N) < (N - t)
        # make column vector to broadcast in jnp.where
        valid_filter = valid_filter[:, None]

        obs2_shifted = jnp.roll(obs2_residual,
                                -t,
                                axis=0)
        # y=0 inserts zeros for the invalid positions
        # keeping size static
        gamma = jnp.sum(jnp.where(valid_filter,
                                  obs1_residual * obs2_shifted,
                                  0),
                        axis=0) / (N - t)
        return gamma

    # need 0 for luscher window selection later
    t_arr = jnp.arange(t_max + 1)

    return jax.vmap(gamma_a_b_t)(t_arr)


def autocovariance_tensor(observables_arr: jnp.ndarray,
                          t_max: int = None):
    """
    Uses vectorized computation to calculate the autocovariance tensor of the observables array.

    Args:
        observables_arr (jax.numpy.ndarray):
            Array of observables, (M, N_trajectories, N_parallel_chains).
        t_max (int, optional): Maximum time lag for autocorrelation. Defaults to None.

    Returns:
        jax.numpy.ndarray: Autocovariance tensor of the observables array,
        with shape (M, M, t_max+1, N_parallel_chains).
    """
    obs1 = observables_arr[0]
    N = obs1.shape[0]  # number of trajectory steps
    N = N - 1  # adjust for zero-based indexing
    if t_max is None:
        # t_max = N//2  # worried this will cut too soon
        t_max = N

    # vectorized like [[obs[1],obs[1]], [obs[2],obs[2]], ...] for all pairs of
    # observables
    def gamma_for_obs_a(obs_a):
        return jax.vmap(lambda obs_b:  # f(a, g(a, b))
                        autocovariance(obs_a,
                                       obs_b,
                                       t_max=t_max)
                        )(observables_arr)

    return jax.vmap(gamma_for_obs_a)(observables_arr)


def obs_derivative(  # n_observables,
        # shape: (N_trajectory steps, N_parallel_chains)
        primary_observables_dict: dict,
        secondary_fns: tuple[Callable, ...]):
    # n_observables = observables.shape[0]
    # n_primary_observables = primary_observables.shape[0]

    # p_obs_means = primary_observables_dict["primary_observables"].mean(axis=(0, 1))

    # # initialize f (jacobian)
    # f = jnp.zeros(shape = (n_observables, n_primary_observables))
    # a = n_observables
    # b = len(primary_observables_dict)
    # # get derivatives
    # if secondary_fn is None:
    #     # primary fns have derivative 1: d(m**4)/d(m**4)
    #     return 1
    # f = jnp.where(secondary_fn is None,
    #               jnp.zeros((a, b)),
    #            #   jnp.array(
    #            #        [*(jax.grad(secondary_fn)(primary_observables_dict)
    #            #                ).values()
    #            #                ]
    #            #                ),
    #               )

    primary_means_dict = jax.tree.map(lambda obs: obs.mean(),
                                      primary_observables_dict)

    # identity for primary observables
    a = len(primary_means_dict)
    eye = jnp.eye(a)

    if len(secondary_fns) == 0:
        return eye

    def grad_for_secondary_fn(A_dict):
        # A is the dictionary of primary observable means
        return jnp.stack(jax.tree.map(lambda fn: fn(A_dict),
                         secondary_fns)
                         )

    # differentiation for secondary functions
    j_dict = jax.jacrev(grad_for_secondary_fn)(primary_means_dict)

    j_secondary = jnp.stack(jax.tree.leaves(j_dict),
                            axis=-1)
    return jnp.vstack((eye, j_secondary))


def autocorrelation_time(gamma: jnp.ndarray,
                         primary_observables_dict: dict[str, jnp.ndarray],
                         secondary_fns: tuple[Callable, ...],
                         c=4
                         ):
    """
     schaefer eq (3.4),(3.5)
     tau_int_F = 0.5 + 1.0/v_f * jnp.sum(gamma_f_t) = 0.5 + jnp.sum(rho_f_t)
     truncated at 0<W<T

    """

    #  a = len(primary_observables_dict)  # number of primary observables
    #  b = len(secondary_fns)

    f = obs_derivative(primary_observables_dict,
                       secondary_fns)
    # shape f is (M, a)

    #  f = jax.vmap(lambda secondary_fn:
    #                jnp.where(secondary_fn is None,
    #                          jnp.vstack([jnp.eye(a)],jnp.zeros((b,a))),
    #                          obs_derivative(
    #                   N_observables=M,
    #                   primary_observables_dict=primary_observables_dict,
    #                   secondary_fn=secondary_fn))
    #              )(secondary_fns)

    #  f = jax.vmap(lambda secondary_fn:
    #          jnp.where(secondary_fn is None,
    #                      jnp.vstack([jnp.eye(a)],jnp.zeros((b,a))),
    #                      obs_derivative(
    #              N_observables=M,
    #              primary_observables_dict=primary_observables_dict,
    #              secondary_fn=secondary_fn))
    #          )(secondary_fns)

    # shape of gamma is (M,M,T,C)
    # where M is the number of observables
    # T is the number of time lags
    # and C is the number of parallel chains.

    # contract over a,b the observable indices
    gamma_f_M_tc = jnp.einsum("ma,mb,abtc->mtc", f, f, gamma)

    # gamma_f_M_tc has shape (M, T, C)
    # we average over the parallel chains (axis=-1) to get shape (M, T)
    gamma_f_M_t = gamma_f_M_tc.mean(axis=-1)

    v_f_M = gamma_f_M_t[:, 0]

    rho_f_M_t = gamma_f_M_t / v_f_M[:, None]
    M, T = rho_f_M_t.shape

    # cumulative sum of the autocorrelation function excluding the zero lag
    rho_cumulative = jnp.cumsum(rho_f_M_t[:, 1:], axis=1)
    # shape of rho_cumulative is (M, T-1)

    # prepend 0.5 to the cumulative sum to account for the zero lag
    tau_cumulative = jnp.concatenate((jnp.full((M, 1), 0.5),
                                     0.5 + rho_cumulative),
                                     axis=1)  # shape (M, T)

    W_sokal_arr = sokal_window(tau_cumulative, c)

    # compute the integrated autocorrelation time for each observable using
    # the specified window
    def integrate_tau(W_arr):
        valid = W_arr > 0
        W_arr_safe = jnp.where(valid, W_arr, 0)

        tau_int = tau_cumulative[jnp.arange(tau_cumulative.shape[0]),
                                 W_arr_safe]
        # tau_int = jnp.where(W_arr >0,
        #                     jnp.take_along_axis(tau_cumulative, W_arr[:,None], axis=1),
        #                     jnp.nan)

        # if any a
        return jnp.where(valid, tau_int, jnp.nan)

    tau_int_f_sokal = integrate_tau(W_sokal_arr)

    # shape of tau_int_f is (M,)

    #  # axis 1 corresponds to the time lags
    #  tau_int = lambda W: 0.5 + jnp.sum(rho_f_M_t[:,1:W+1],
    #                          axis=1)

    results_dict = {
        "rho_f_M_t": rho_f_M_t,
        "W_sokal_arr": W_sokal_arr,
        "tau_int_f_sokal": tau_int_f_sokal
    }
    return results_dict


def sokal_window(tau_cumulative, c=4):
    """
    Parameters
    ----------
    tau_cumulative : jnp.ndarray
        Array of cumulative autocorrelation times for each observable.
        Shape (M, T), where M is the number of observables and T is the number of time lags.
    c : float, optional
        Sokal criterion constant, by default 4.

    Returns
    -------
    W_first : int or None
        The first window size in W_arr that satisfies the Sokal criterion,
        or -1 if no such window exists.
    """
    # def sokal_variance(W, tau_int_f):
    #     return (4*W+2)/2 * tau_int_f**2

    t_arr = jnp.arange(tau_cumulative.shape[1])

    # shape (1, T) to broadcast with tau_cumulative
    t_arr = t_arr[None, :]

    mask_arr = (t_arr >= 1) & (t_arr >= c * tau_cumulative)
    # shape of mask_arr is (M, T)

    # first index along time lags where the mask is True
    first = jnp.argmax(mask_arr, axis=1)  # returns index of first 1
    # returns True if any window satisfies the Sokal criterion for each
    # observable
    found_mask = mask_arr.any(axis=1)

    # if no window satisfies the Sokal criterion,
    # we return -1 for that observable
    W_first = jnp.where(found_mask, t_arr[first], -1)
    # shape of W_first is (M,), one w per observable

    return W_first


def luscher_window(rho_f_M_t, N, lam=100):
    M, T = rho_f_M_t.shape
    # M is all observables
    # T is the number of measuredtime lags
    T = T - 1  # adjust for zero-based indexing

    # k+t must be in range of available time lags
    # sum runs from k=1 to k=t+lam, so T>= t+lam for the sum to be valid
    # but we also have rho(k+t) in the sum, so k+t<= T-1 for the indices to be valid
    # -> (t+lam)+t <= T
    # -> t <= (T-lam)/2
    t_max = (T - lam) // 2  # //2 because we need integer

    if t_max < 1:
        raise ValueError("insufficient lags for lambda")

    # exclude 0 time
    t_arr = jnp.arange(1, t_max + 1)

    # itterated time index k = 1 to t-lam
    k_arr = jnp.arange(1, T)

    def rho_variance(t):
        # get valid indices for k

        # indices for k+t
        kp = jnp.minimum(k_arr + t, T)
        # ex) T=7, t=2, pick the lesser of k+t and T
        #  k_array = [1, 2, 3, 4, 5, 6, 7]
        #  kp = [3, 4, 5, 6, 7, 7, 7]

        # indices for k-t
        km = jnp.abs(k_arr - t)
        # ex) T=8, t=2, pick the absolute difference of k-t
        #  k_array = [1, 2, 3, 4, 5, 6, 7]
        #  km = [1, 0, 1, 2, 3, 4, 5]
        # negative values are allowed because rho(t)=rho(-t)

        # for all observables,
        # collect only the rho values for the valid indices
        # this will produce duplicates that we cut later
        rho_kpt = rho_f_M_t[:, kp]  # (M, len(k_array))
        rho_kmt = rho_f_M_t[:, km]  # (M, len(k_array))
        rho_k = rho_f_M_t[:, k_arr]  # (M, len(k_array))
        rho_t = rho_f_M_t[:, t, None]  # (M, 1) to broadcast with rho_k

        s = (rho_kpt + rho_kmt - 2 * rho_k * rho_t)**2
        # shape of s is (M, len(k_array))

        #  last term in sum is k=t+lam
        # but we only have T measurements
        # so we disallow any k+t above T
        valid_k = (
            k_arr <= t +
            lam) & (
            k_arr +
            t < T +
            1)  # shape (len(k_array),)
        # shape (len(k_array), 1) for broadcasting with s
        valid_k = valid_k[:, None]

        # sum over valid k indices only
        rho_var = jnp.where(valid_k, s.sum(axis=1), -1) / N
        # shape of rho_var is (M,)

        return rho_var

    rho_var_arr = jax.vmap(rho_variance)(t_arr).T
    # shape of rho_var_arr is (M, n_valid_t)
    rho_std_arr = jnp.sqrt(rho_var_arr)

    # compare at each t in t_arr
    mask = ((rho_std_arr > 0)
            & jnp.isfinite(rho_std_arr)
            & (rho_f_M_t[:, t_arr] <= rho_std_arr)
            )

    first = jnp.argmax(mask, axis=1)  # first index where the condition is met
    # True if any t satisfies the condition for each observable
    found = mask.any(axis=1)

    # t_array[first] gives the first t where the condition is met for each observable
    # get the first t where the condition is met for each observable
    # but if no t satisfies the condition, we return -1 for that observable
    W_luscher = jnp.where(found, t_arr[first], -1)
    return W_luscher
