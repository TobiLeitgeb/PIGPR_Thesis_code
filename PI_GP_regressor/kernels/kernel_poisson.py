#----------------------------------------
# author: Tobias Leitgeb
# This file contains the implementation of the kernel for the poisson equation laplace u = f. 
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
    return prefactor* k_uu_data

k_uf = jit(vmap(vmap(k_uf,(None,0,None)), (0,None,None)))

def k_fu(x, x_bar, params):
    """ Kernel function for the mixed covaricen function L_x L_x' k_uu = k_fu F x U --> R
        the derivatives were calcualted with exp(-gamma ...) --> gamma = 1/(2*sigma^2)"""
    k_uu_data = single_rbf(x, x_bar, params)
    x,y = x[0], x[1]
    x_bar, y_bar = x_bar[0], x_bar[1]
    gamma = 1/(2*params[0]**2)
    prefactor = 2*gamma *(2*gamma*((x-x_bar)**2 + (y-y_bar)**2)-2)
    return prefactor* k_uu_data
k_fu = jit(vmap(vmap(k_fu,(None,0,None)), (0,None,None)))

def k_ff(x, x_bar, params):
    """ Kernel function for the mixed covaricen function L_x L_x' k_uu = k_ff F x F --> R
        the derivatives were calcualted with exp(-gamma ...) --> gamma = 1/(2*sigma^2)"""
    k_uu_data = single_rbf(x, x_bar, params)
    x,y = x[0], x[1]
    x_bar, y_bar = x_bar[0], x_bar[1]
    gamma = 1/(2*params[0]**2)
    #lap lap' k_uu
    #consist of 3 terms d_xxxx = d^2/dx^2 d^2/dx'^2 k_uu, d_yyyy = d^2/dy^2 d^2/dy'^2 , d_xxyy = d^2/dx^2 d^2/dy'^2 
    
    d_xxxx = 4*gamma**2*(4*gamma*(x-x_bar)**2 *(gamma*(x-x_bar)**2 - 3)+3)
    d_yyyy = 4*gamma**2*(4*gamma*(y-y_bar)**2 *(gamma*(y-y_bar)**2 - 3)+3)
    d_xxyy = 4*gamma**2*(2*(x-x_bar)**2*gamma - 1)*(2*(y-y_bar)**2*gamma - 1)
    return (d_xxxx + d_yyyy + 2*d_xxyy)*k_uu_data

k_ff = jit(vmap(vmap(k_ff,(None,0,None)), (0,None,None)))

@jit
def gram_Matrix(X, X_bar, params, noise = [0,0]):
    """computes the gram matrix of the kernel
    noise = [noise_u, noise_f]
    """
    assert X.shape[1] == 2, "X must be a 2d array"
    X, T = X[:,0].reshape(-1,1), X[:,1].reshape(-1,1)
    X_bar, T_bar = X_bar[:,0].reshape(-1,1), X_bar[:,1].reshape(-1,1)

    X_u = jnp.hstack([X,T])
    X_f = jnp.hstack([X_bar,T_bar])
    
    k_uu_matrix = k_uu(X_u, X_u, params) + noise[0] * jnp.eye(len(X)) 
    k_uf_matrix = k_uf(X_u, X_f, params)                              
    k_fu_matrix = k_fu(X_f, X_u, params) 
    k_ff_matrix = k_ff(X_f, X_f, params) + noise[1] * jnp.eye(len(X_bar))
    #combine all the matrices to the full gram matrix
    K = jnp.block([[k_uu_matrix, k_uf_matrix], [k_fu_matrix, k_ff_matrix]])
    return K