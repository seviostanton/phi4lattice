import jax
import jax.numpy as jnp
from typing import Callable


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

def obs_derivative(#n_observables,
                   primary_observables_dict: dict,
                   secondary_fns: tuple[Callable, ...]):
        # n_observables = observables.shape[0]
        # n_primary_observables = primary_observables.shape[0]
        
        # p_obs_means = primary_observables_dict["primary_observables"].mean(axis=(0, 1))

        # # initialize f (jacobian)
        # f = jnp.zeros(shape = (n_observables, n_primary_observables))
        #a = n_observables
        #b = len(primary_observables_dict)
        # # get derivatives
        # if secondary_fn is None:
        #     # primary fns have derivative 1: d(m**4)/d(m**4)
        #     return 1  
        # f = jnp.where(secondary_fn is None,
        #               jnp.zeros((a, b)),
                    #   jnp.array(
                    #        [*(jax.grad(secondary_fn)(primary_observables_dict)
                    #                ).values()
                    #                ]
                    #                ),
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
             return jnp.stack(
                  jax.tree.map(lambda fn: fn(A_dict),
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
                         W_arr):
     """
     schaefer eq (3.4),(3.5)
     tau_int_F = 0.5 + 1.0/v_f * jnp.sum(gamma_f_t) = 0.5 + jnp.sum(rho_f_t)
     truncated at 0<W<T

     """
     M = gamma.shape[0]  # number of observables
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
     gamma_f_M_tc = jnp.einsum("ma,mb,abtc->mtc", f,f,gamma)
     
     # gamma_f_M_tc has shape (M, T, C)
     # we average over the parallel chains (axis=-1) to get shape (M, T)
     gamma_f_M_t = gamma_f_M_tc.mean(axis=-1)

     v_f_M = gamma_f_M_t[:,0]

     rho_f_M_t = gamma_f_M_t / v_f_M[:,None]
     # cumulative sum of the autocorrelation function excluding the zero lag
     rho_cumulative = jnp.cumsum(rho_f_M_t[:,1:], axis=1)

     # prepend 0.5 to the cumulative sum to account for the zero lag
     tau_cumulative = jnp.concatenate((jnp.full((M,1), 0.5),
                                       rho_cumulative),
                                       axis=1)
     if W_arr.shape != (M,):
         raise ValueError("W_arr must have shape (M,), one window per observable")
     tau_int = jnp.take_along_axis(tau_cumulative, W_arr[:,None], axis=1)
    #  # axis 1 corresponds to the time lags
    #  tau_int = lambda W: 0.5 + jnp.sum(rho_f_M_t[:,1:W+1],
    #                          axis=1)
     return tau_int, rho_f_M_t
     