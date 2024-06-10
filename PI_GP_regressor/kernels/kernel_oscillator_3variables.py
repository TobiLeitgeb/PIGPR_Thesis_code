
#----------------------------------------
# author: Tobias Leitgeb
# This file contains the implementation of the kernel for a damped oscillator. The computation of the kernel is done with JAX. Once with automatic differentiation and once with the analytical derivatives.

#----------------------------------------
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np



@jit
def rbf_kernel(x1, x2, params):
    x1, x2 = x1.flatten(), x2.flatten()
    l, sigma_f = params[0], params[1]
    return jnp.squeeze(sigma_f**2 * jnp.exp(-0.5/l**2 * jnp.sum((x1 - x2)**2)))
#k_uu
k_uu = jit(vmap(vmap(rbf_kernel, (None, 0, None)), (0, None, None)))

#k_ff
#k_ff = m^2 dt^2 dt'^2 + 2mk dt'^2 + b^2 dtdt' + k^2
def k_ff_jax(x,y, params):
    x, y = jnp.squeeze(x), jnp.squeeze(y)
    m, g, k = params[2], params[3], params[4]
    #dx^2 dy'^2
    #calculates second derivative with respect to the second input
    dk_yy = grad(grad(rbf_kernel, argnums=1), argnums=1)
    #calcculates second derivative of dk_yy with respect to first input
    dk_xxyy = grad(grad(dk_yy, argnums=0), argnums=0) (x, y, params)
    #dx'^2
    dk_yy = grad(grad(rbf_kernel, argnums=1), argnums=1) (x, y, params)
    #dxdx'
    dk_xy = grad(grad(rbf_kernel, argnums=1), argnums=0) (x, y, params)
    #k^2
    #calculates the general kernel without derivatives
    K_no_div = rbf_kernel(x, y, params)
    #arrange terms [m^2 dt^2 dt'^2 + 2mk dt'^2 + b^2 dtdt' + k^2]k(x,y)
    return m**2 * dk_xxyy + 2*m*k*dk_yy + \
           g**2 * dk_xy + k**2 * K_no_div
#vectorize the function to form cov matrix
k_ff_jax = jit(vmap(vmap(k_ff_jax, (None, 0, None)), (0, None, None)))

#k_uf
#k_uf = m dt'^2 + b dt' + k
@jit
def k_uf_jax(X, Y, params):
    x, y = jnp.squeeze(x), jnp.squeeze(y)
    m = params[2]
    b = params[3]
    k = params[4]
    #dt'^2
    dk_yy = grad(grad(rbf_kernel, argnums=1), argnums=1)(x, y, params)
    #dt'
    dk_y = grad(rbf_kernel, argnums=1)(x, y, params)
    #k
    k_normal = rbf_kernel(x, y, params)
    return m * dk_yy + b * dk_y + k * k_normal

k_uf_jax = jit(vmap(vmap(k_uf_jax, (None, 0, None)), (0, None, None)))

@jit
def k_fu_jax(X, Y, params):
    x, y = jnp.squeeze(x), jnp.squeeze(y)
    m = params[2]
    b = params[3]
    k = params[4]
    #dt^2
    dk_xx = grad(grad(rbf_kernel, argnums=0), argnums=0)(x, y, params)
    #dt
    dk_x = grad(rbf_kernel, argnums=0)(x, y, params)
    #k
    k_normal = rbf_kernel(x, y, params)
    return m * dk_xx + b * dk_x + k * k_normal

k_fu_jax = jit(vmap(vmap(k_fu_jax, (None, 0, None)), (0, None, None)))

def k_uf(x, y, params):
    m = params[2]
    b = params[3]
    k = params[4]
    gamma = 0.5 / params[0]**2
    #dt'^2
    k_yy = 2*gamma*(2*gamma * (x-y)**2 - 1) 
    #dt'
    k_y = 2*gamma*(x-y) 
    #no dev
    k_normal = rbf_kernel(x, y, params)
    return jnp.squeeze(m * k_yy + b * k_y + k) * rbf_kernel(x, y, params)
k_uf = jit(vmap(vmap(k_uf, (None, 0, None)), (0, None, None)))


def k_fu(x,y,params):
    m = params[2]
    b = params[3]
    k = params[4]
    gamma = 0.5 / params[0]**2
    #dt^2
    k_xx = 2*gamma*(2*gamma * (x-y)**2 - 1) 
    #dt'
    k_x = -2*gamma*(x-y) 
    return jnp.squeeze(m * k_xx + b * k_x + k ) * rbf_kernel(x, y, params)
k_fu = jit(vmap(vmap(k_fu, (None, 0, None)), (0, None, None)))


def k_ff(x,y,params):
    m, b, k = params[2], params[3], params[4]
    #rewrite 1/2l^2 = gamma
    gamma = 0.5 / params[0]**2
    #dt^2 dt'^2
    dif = (x-y)
    #The following terms k_xxyy k_yy k_xy are the additional prefactor 
    # terms that come from taking the derivatives of the exponential 
    k_xxyy = (16*gamma**4* dif**4 - 48*gamma**3*dif**2 + 12*gamma**2) 
    #dt'^2
    k_yy = 2*gamma*(2*gamma * dif**2 - 1) 
    #dtdt'
    k_xy = (2*gamma - 4*gamma**2*dif**2) 
    #calculates the general kernel without derivatives
    K_no_div = rbf_kernel(x, y, params)
    #arrange terms [m^2 dt^2 dt'^2 + 2mk dt'^2 + b^2 dtdt' + k^2]k(x,y)
    #multiply the calculated terms with the general kernel
    return ((m**2 * k_xxyy + 2*m*k*k_yy 
             + b**2 * k_xy + k**2) * K_no_div
        ).squeeze()
#vectorize the function to form cov matrix
k_ff = jit(vmap(vmap(k_ff, (None, 0, None)), (0, None, None)))



@jit
def gram_Matrix(X, Y, params, noise = [0,0]):
    """computes the gram matrix of the kernel
    params = [l_x, sigma_f_sq, l_t, alpha]
    noise = [noise_u, noise_f]
    """
    k_uu_matrix = k_uu(X, X, params) + noise[0] * jnp.eye(len(X))
    k_uf_matrix = k_uf(X, Y, params)
    k_fu_matrix = k_uf_matrix.T
    k_ff_matrix = k_ff(Y, Y, params) + noise[1] * jnp.eye(len(Y))
    #combine all the matrices to the full gram matrix
    K = jnp.block([[k_uu_matrix, k_uf_matrix], [k_fu_matrix, k_ff_matrix]])
    return K