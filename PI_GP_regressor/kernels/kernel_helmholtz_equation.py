#----------------------------------------
# author: Tobias Leitgeb
# This file contains the implementation of the kernel for the helmholtz equation d^2u/dx^2 + d^2u/dy^2 + k^2 u = f.
#The computation of the kernel is done with JAX. Once with automatic differentiation and once with the analytical derivatives. 
# Could contain an errro. The example with a helmholtz equation did not work and was not used in the thesis.
#----------------------------------------

import jax.numpy as jnp
from jax import grad, jit, vmap

def single_rbf(x, x_bar, params):
    """Single RBF kernel function for a two dimensional input. This function is not vectorized yet.
    """
    x,y = x[0], x[1]
    x_bar, y_bar = x_bar[0], x_bar[1]
    return params[1]*jnp.exp( -(((x-x_bar)**2+ (y-y_bar)**2))/ (2 * params[0]**2))

k_uu = jit(vmap(vmap(single_rbf,(None,0,None)), (0,None,None)))

def k_uf(x, x_bar, params):
    """ Kernel function for the mixed covaricen function L_x' k_uu = k_uf U x F --> R
        the derivatives were calcualted with exp(-gamma ...) --> gamma = 1/(2*sigma^2)"""
    k_uu_data = single_rbf(x, x_bar, params)
    x,y = x[0], x[1]
    x_bar, y_bar = x_bar[0], x_bar[1]
    gamma = 1/(2*params[0]**2)
    prefactor = 2*gamma *(2*gamma*((x-x_bar)**2 + (y-y_bar)**2)-2)
    return (prefactor + params[2]**2)* k_uu_data

k_uf = jit(vmap(vmap(k_uf,(None,0,None)), (0,None,None)))

def k_fu(x, x_bar, params):
    """ Kernel function for the mixed covaricen function L_x L_x' k_uu = k_fu F x U --> R
        the derivatives were calcualted with exp(-gamma ...) --> gamma = 1/(2*sigma^2)"""
    k_uu_data = single_rbf(x, x_bar, params)
    x,y = x[0], x[1]
    x_bar, y_bar = x_bar[0], x_bar[1]
    gamma = 1/(2*params[0]**2)
    prefactor = 2*gamma *(2*gamma*((x-x_bar)**2 + (y-y_bar)**2)-2)
    return (prefactor + params[2]**2)* k_uu_data
k_fu = jit(vmap(vmap(k_fu,(None,0,None)), (0,None,None)))

def k_ff(x, x_bar, params):
    """ Kernel function for the mixed covaricen function L_x L_x' k_uu = k_ff F x F --> R
        the derivatives were calcualted with exp(-gamma ...) --> gamma = 1/(2*sigma^2)"""
    k_uu_data = single_rbf(x, x_bar, params)
    x,y = x[0], x[1]
    x_bar, y_bar = x_bar[0], x_bar[1]
    gamma = 1/(2*params[0]**2)
    #consist of 3 terms
    term1 = 4*gamma**2*(4*gamma*(x-x_bar)**2 *(gamma*(x-x_bar)**2 - 3)+3)
    term2 = 4*gamma**2*(4*gamma*(y-y_bar)**2 *(gamma*(y-y_bar)**2 - 3)+3)
    term3 = 4*gamma**2*(2*(x-x_bar)**2*gamma - 1)*(2*(y-y_bar)**2*gamma - 1)
    return (term1 + params[2]**4*term2 + params[2]**2*2*term3)* k_uu_data

k_ff = jit(vmap(vmap(k_ff,(None,0,None)), (0,None,None)))

@jit
def gram_Matrix(XT, YS, params, noise = [0,0]):
    """computes the gram matrix of the kernel
    noise = [noise_u, noise_f]
    """
    assert XT.shape[1] == 2, "X must be a 2d array"
    X, T = XT[:,0].reshape(-1,1), XT[:,1].reshape(-1,1)
    Y, S = YS[:,0].reshape(-1,1), YS[:,1].reshape(-1,1)

    #unfortunately we have to stack the data in a certain way to make the kernel work...
    XX = jnp.hstack([X,T])
    TT = jnp.hstack([X,T])
    XY = jnp.hstack([X,T])
    TS = jnp.hstack([Y,S])
    YX = jnp.hstack([Y,S])
    ST = jnp.hstack([X,T])
    YY = jnp.hstack([Y,S])
    SS = jnp.hstack([Y,S])
    
    k_uu_matrix = k_uu(XX, TT, params) + noise[0] * jnp.eye(len(X)) #xxtt --- we have to slice xt xt
    k_uf_matrix = k_uf(XY, TS, params)                              #xyts 
    k_fu_matrix = k_fu(YX, ST, params) 
    k_ff_matrix = k_ff(YY, SS, params) + noise[1] * jnp.eye(len(Y))
    #combine all the matrices to the full gram matrix
    K = jnp.block([[k_uu_matrix, k_uf_matrix], [k_fu_matrix, k_ff_matrix]])
    return K