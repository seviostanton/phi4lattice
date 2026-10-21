import jax
import jax.numpy as jnp


def autocovariance(obs1: jnp.ndarray,
                    obs2: jnp.ndarray=None,
                    t_max: int=None) -> jnp.ndarray:
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

    # if t_max is not provided, set it to N//2
    if t_max is None:
         t_max = N//2
    if not isinstance(t_max, int):
        raise ValueError("t_max must be an integer")
    if t_max < 0 or t_max >= N:
        raise ValueError("t_max must be in the range [0, N-1]")
    if N <=2:
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
        valid_filter = jnp.arange(N)<(N-t)
        # make column vector to broadcast in jnp.where
        valid_filter = valid_filter[:,None]
        
        obs2_shifted = jnp.roll(obs2_residual,
                                  -t,
                                  axis=0)
        # y=0 inserts zeros for the invalid positions
        # keeping size static
        gamma = jnp.sum(jnp.where(valid_filter,
                                  obs1_residual*obs2_shifted,
                                  0),
                                  axis=0)/(N-t)
        return gamma

    # need 0 for luscher window selection later
    t_arr = jnp.arange(t_max+1)

    return jax.vmap(gamma_a_b_t)(t_arr)


def autocovariance_tensor(observables_arr: jnp.ndarray,
                           t_max: int = None):
    """
    Uses vectorized computation to calculate the autocovariance tensor of the observables array.

    Args:
        observables_arr (jax.numpy.ndarray): Array of observables, (M, N_trajectories, N_parallel_chains).
        t_max (int, optional): Maximum time lag for autocorrelation. Defaults to None.

    Returns:
        jax.numpy.ndarray: Autocovariance tensor of the observables array, with shape (M, M, t_max+1, N_parallel_chains).
    """

    # vectorized like [[obs[1],obs[1]], [obs[2],obs[2]], ...] for all pairs of observables
    def gamma_for_obs_a(obs_a):
         return jax.vmap(lambda obs_b:  # f(a, g(a, b))
                         autocovariance(obs_a,
                                        obs_b,
                                        t_max=t_max)
                                        )(observables_arr)
    
    return jax.vmap(gamma_for_obs_a)(observables_arr)

def obs_derivative(primary_observables, secondary_fn):
        obs_means = primary_observables.mean(axis=(0, 1))

        # get derivatives
        f = jax.grad(secondary_fn)(obs_means)
        return f


def autocorrelation_time(gamma,
                         primary_observables,
                         secondary_fn,
                         W):
     """
     schaefer eq (3.4),(3.5)
     tau_int_F = 0.5 + 1.0/v_f * jnp.sum(gamma_f_t) = 0.5 + jnp.sum(rho_f_t)
     truncated at 0<W<T
     """
     f = obs_derivative(primary_observables, secondary_fn)
     # shape of gamma is (M,M,T,C)
     # where M is the number of observables
     # T is the number of time lags
     # and C is the number of parallel chains.

     #contract over a,b the observable indices
     gamma_f_tc = jnp.einsum("a,b,abtc->tc", f,f,gamma)
     # average over the parallel chains
     gamma_f_t = gamma_f_tc.mean(axis=1)

     v_f = gamma_f_t[0]

     tau_int = 0.5 + 1.0/v_f * jnp.sum(gamma_f_t[1:W+1])
     return tau_int
     