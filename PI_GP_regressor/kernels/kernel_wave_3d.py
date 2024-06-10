#----------------------------------------
# author: Tobias Leitgeb
# This file contains the implementation of the kernel for the 2d time dependent wave equation d^2u/dt^2 - c^2 d^2u/dx^2 = f(x,t). 
#----------------------------------------
import jax.numpy as jnp
from jax import grad, jit, vmap

"""
params = [l_space, sigma, l_t, c] We use the same length scale for space and a seperate one for time
"""


def single_rbf(x, x_bar, params):
    """Single RBF kernel function for a 3 dimensional input.
    """
    x,y,t = x[0], x[1], x[2]
    x_bar, y_bar, t_bar = x_bar[0], x_bar[1], x_bar[2]
    return params[1]*jnp.exp( -(((x-x_bar)**2+ (y-y_bar)**2))/ (2 * params[0]**2) - (t-t_bar)**2 / (2*params[2]**2))

k_uu = jit(vmap(vmap(single_rbf,(None,0,None)), (0,None,None)))


def k_uf(x, x_bar, params):
    """ Kernel function for the mixed covaricen function L_x' k_uu = k_uf U x F --> R
        the derivatives were calcualted with exp(-gamma ...) --> gamma = 1/(2*l^2)"""
    k_uu_data = single_rbf(x, x_bar, params)
    x,y,t = x[0], x[1], x[2]
    x_bar, y_bar, t_bar = x_bar[0], x_bar[1], x_bar[2]
    gamma_space = 1/(2*params[0]**2)
    gamma_time = 1/(2*params[2]**2)
    c = params[3]
    #d_tt
    prefactor = 2*gamma_time *(2*gamma_time*((t-t_bar)**2)-1)
    # lap_ * k_uu
    prefactor2 = 2*gamma_space *(2*gamma_space*((x-x_bar)**2 + (y-y_bar)**2)-2)
    return (prefactor - prefactor2 * c**2)* k_uu_data

k_uf = jit(vmap(vmap(k_uf,(None,0,None)), (0,None,None)))

def k_fu(x, x_bar, params):
    """ Kernel function for the mixed covaricen function L_x' k_uu = k_uf U x F --> R
        the derivatives were calcualted with exp(-gamma ...) --> gamma = 1/(2*l^2)"""
    k_uu_data = single_rbf(x, x_bar, params)
    x,y,t = x[0], x[1], x[2]
    x_bar, y_bar, t_bar = x_bar[0], x_bar[1], x_bar[2]
    gamma_space = 1/(2*params[0]**2)
    gamma_time = 1/(2*params[2]**2)
    c = params[3]
    #d_tt
    prefactor = 2*gamma_time *(2*gamma_time*((t-t_bar)**2)-1)
    # lap_ * k_uu
    prefactor2 = 2*gamma_space *(2*gamma_space*((x-x_bar)**2 + (y-y_bar)**2)-2)
    return (prefactor - prefactor2 * c**2)* k_uu_data

k_fu = jit(vmap(vmap(k_fu,(None,0,None)), (0,None,None)))

def k_ff(x, x_bar, params):
    """ Kernel function for the mixed covaricen function L_x L_x' k_uu = k_ff F x F --> R
        the derivatives were calcualted with exp(-gamma ...) --> gamma = 1/(2*l^2)"""
    k_uu_data = single_rbf(x, x_bar, params)
    x,y,t = x[0], x[1], x[2]
    x_bar, y_bar, t_bar = x_bar[0], x_bar[1], x_bar[2]
    gamma_space = 1/(2*params[0]**2)
    gamma_time = 1/(2*params[2]**2)
    c = params[3]
    # D D' k_uu = dt^2 dt'^2 - c^2 lap dt'^2 - c^2 dt^2 lap' + c^4 lap lap' ### c2 lap dt^2 = c^2 dt^2 lap

    #dt^2 dt'^2
    dttdtt = 4*gamma_time**2*(4*gamma_time*(t-t_bar)**2 *(gamma_time*(t-t_bar)**2 - 3)+3)
    #c^2 lap dt'^2
    lapdtt = 2*gamma_space *(2*gamma_space*((x-x_bar)**2 + (y-y_bar)**2)-2)*2*gamma_time *(2*gamma_time*((t-t_bar)**2)-1)
    #lap lap` k_uu part
    #consist of 3 terms
    term1 = 4*gamma_space**2*(4*gamma_space*(x-x_bar)**2 *(gamma_space*(x-x_bar)**2 - 3)+3)
    term2 = 4*gamma_space**2*(4*gamma_space*(y-y_bar)**2 *(gamma_space*(y-y_bar)**2 - 3)+3)
    term3 = 4*gamma_space**2*(2*(x-x_bar)**2*gamma_space - 1)*(2*(y-y_bar)**2*gamma_space - 1)
    laplap = (term1 + term2 + 2*term3)
    return (dttdtt - 2*c**2* lapdtt +c**4* laplap)* k_uu_data

k_ff = jit(vmap(vmap(k_ff,(None,0,None)), (0,None,None)))

@jit
def gram_Matrix(X, X_bar, params, noise = [0,0]):
    """computes the gram matrix of the kernel
    noise = [noise_u, noise_f]
    """
    assert X.shape[1] == 3, "X must be a N x 3 array"
    X, Y, T = X[:,0].reshape(-1,1), X[:,1].reshape(-1,1), X[:,2].reshape(-1,1)
    X_bar, Y_bar, T_bar = X_bar[:,0].reshape(-1,1), X_bar[:,1].reshape(-1,1), X_bar[:,2].reshape(-1,1)

    X_u = jnp.hstack([X,Y,T])
    X_f = jnp.hstack([X_bar,Y_bar,T_bar])
    
    k_uu_matrix = k_uu(X_u, X_u, params) #+ noise[0]**2 * jnp.eye(len(X)) 
    k_uf_matrix = k_uf(X_u, X_f, params)                              
    k_fu_matrix = k_fu(X_f, X_u, params) 
    k_ff_matrix = k_ff(X_f, X_f, params) #+ noise[1]**2 * jnp.eye(len(Y))
    #combine all the matrices to the full gram matrix
    K = jnp.block([[k_uu_matrix, k_uf_matrix], [k_fu_matrix, k_ff_matrix]])
    return K