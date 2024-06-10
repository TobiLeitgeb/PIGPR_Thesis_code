#----------------------------------------
# author: Tobias Leitgeb
# This file contains the implementation of the kernel for the heat equation dT/dt -alpha d^2T/dx^2 = f(x,t). The computation of the kernel is done with JAX. Once with automatic differentiation and once with the analytical derivatives. 
#----------------------------------------

from jax import numpy as jnp
from jax import jit, grad, vmap

@jit
def rbf_kernel_single_x(x: float, y: float, params: list) -> float:
    """general RBF kernel k(x,y)"""
    l_x, sigma_f_sq = params[0], params[1]
    sqdist = jnp.sum(x-y)**2
    return sigma_f_sq * jnp.exp(-0.5 / l_x**2 * sqdist)
@jit
def rbf_kernel_single_t(t: float, s: float, l_t: float) -> float:
    """general RBF kernel. takes scalar inputs t,s and returns k(t,s)"""
    sqdist = jnp.sum(t-s)**2
    value = jnp.exp(-0.5 / l_t**2 * sqdist)
    return value

@jit
def k_uu_jax(X, Y, params):
    """
    computes k_uu part of the block matrix K
    """
    assert X.shape[1] == 2, "X must be a 2d array"
    X, T = X[:,0].flatten(), X[:,1].flatten()
    Y, S = Y[:,0].flatten(), Y[:,1].flatten()
    # vectorize the kernel so that it can take arrays as input
    vectorized_rbf_kernel_x = vmap(vmap(rbf_kernel_single_x, (None, 0, None)), (0, None, None)) 
    vectorized_rbf_kernel_t = vmap(vmap(rbf_kernel_single_t, (None, 0, None)), (0, None, None)) 
    params = params[:-1] #remove alpha from the parameters since it is not used in the kernel
    l_t = params[2]
    #combine the two kernels to the full kernel k(x,t,y,s)
    return vectorized_rbf_kernel_x(X, Y, params) * vectorized_rbf_kernel_t(T, S, l_t)

@jit
def k_ff_jax(X, Y, params):
    """computes k_ff part of the block matrix K. It corresponds to the part with double the operator L: k_ff = L k_uu L'^T
    #k_ff = d^2/dtds K_uu + alpha^2 d^2/dx^2 d^2/dy^2 K_uu
       params = [l_x, sigma_f_sq, l_t, alpha]
    """
    assert X.shape[1] == 2, "X must be a 2d array"
    X, T = X[:,0].flatten(), X[:,1].flatten()
    Y, S = Y[:,0].flatten(), Y[:,1].flatten()
    alpha = params[-1]
    l_t = params[2]
    params = params[:-1]
    # flatten the data so that it can be used in the grad function (only 1d arrays are allowed in grad)
    X,Y,T,S = X.flatten(), Y.flatten(), T.flatten(), S.flatten() 
    #vectorizazion of both the kernel functions 
    rbf_kernel_x = vmap(vmap(rbf_kernel_single_x, (None, 0, None)), (0, None, None))
    rbf_kernel_t = vmap(vmap(rbf_kernel_single_t, (None, 0, None)), (0, None, None))
    #compute the derivatives seperately and then multiply by the other kernel.
    dk_dtds = grad(grad(rbf_kernel_single_t, argnums = 1), argnums = 0) #derivative with respect to t and s
    vectorized_dtds = vmap(vmap(dk_dtds, (None, 0, None)), (0, None, None))

    dk_dydy = grad(grad(rbf_kernel_single_x, argnums = 1), argnums = 1) #second derivative of the k with respect y
    dk_dxdxdydy = grad(grad(dk_dydy, argnums = 0), argnums = 0) # second derivative with respect to x of dk_dydy
    #vectorize the fourth derivative
    vectorized_dxdxdydy = vmap(vmap(dk_dxdxdydy, (None, 0, None)), (0, None, None))

    #combine the seperate parts k_ts + alpha^2 k_xxyy and multiply by the missing kernel terms respectively
    k_ts = vectorized_dtds(T,S,params[2]) * rbf_kernel_x(X,Y,params) 
    k_xxyy= vectorized_dxdxdydy(X,Y,params) * rbf_kernel_t(T,S,l_t) 
    return k_ts + alpha**2 * k_xxyy

@jit
def k_uf_jax(X, Y, params):
    """computes the cross kernel k_uf. It corresponds to the part with single operator L: k_uf = k_uu L'^T
    K_uf = d/ds K_uu - alpha d^2/dy^2 K_uu
    """
    assert X.shape[1] == 2, "X must be a 2d array"
    X, T = X[:,0].flatten(), X[:,1].flatten()
    Y, S = Y[:,0].flatten(), Y[:,1].flatten()
    alpha = params[-1]
    l_t = params[2]
    params = params[:-1]
    X,Y,T,S = X.flatten(), Y.flatten(), T.flatten(), S.flatten() 
    #vectorizazion of both the kernel functions 
    vectorized_rbf_kernel_x = vmap(vmap(rbf_kernel_single_x, (None, 0, None)), (0, None, None))
    vectorized_rbf_kernel_t = vmap(vmap(rbf_kernel_single_t, (None, 0, None)), (0, None, None))
    #compute the derivatives seperately and then multiply by the other kernel.
    # dk/ds
    dk_ds = grad(rbf_kernel_single_t, argnums = 1) 
    vectorized_dk_ds = vmap(vmap(dk_ds, (None, 0, None)), (0, None, None))

    # dk/dy^2
    dk_dydy = grad(grad(rbf_kernel_single_x, argnums = 1), argnums = 1) #derivative with respect to y
    vectorized_dk_dydy = vmap(vmap(dk_dydy, (None, 0, None)), (0, None, None))
    
    k_s = vectorized_dk_ds(T,S,l_t) * vectorized_rbf_kernel_x(X,Y,params)
    k_yy = vectorized_dk_dydy(X,Y,params) * vectorized_rbf_kernel_t(T,S,l_t)
    return k_s - alpha * k_yy

@jit
def k_fu_jax(X, Y, params):
    """computes the cross kernel k_fu. It corresponds to the part with single operator L: k_fu = L k_uu 
    K_fu = d/dt K_uu - alpha d^2/dx^2 K_uu"""
    assert X.shape[1] == 2, "X must be a 2d array"
    X, T = X[:,0].flatten(), X[:,1].flatten()
    Y, S = Y[:,0].flatten(), Y[:,1].flatten()
    alpha = params[-1]
    l_t = params[2]
    params = params[:-1]
    
    X,Y,T,S = X.flatten(), Y.flatten(), T.flatten(), S.flatten() #this is not my favourite solution, but it works atm. However there still is a problem with multidimesional data
    # first I vectorize both the kernel functions seperately
    vectorized_rbf_kernel_x = vmap(vmap(rbf_kernel_single_x, (None, 0, None)), (0, None, None))
    vectorized_rbf_kernel_t = vmap(vmap(rbf_kernel_single_t, (None, 0, None)), (0, None, None))
    #compute the derivatives seperately and then multiply by the other kernel.
    # dk/dt
    dk_dt = grad(rbf_kernel_single_t, argnums = 0)
    vectorized_dk_dt = vmap(vmap(dk_dt, (None, 0, None)), (0, None, None))

    # dk/dx^2
    dk_dxdx = grad(grad(rbf_kernel_single_x, argnums = 0), argnums = 0) # this is both the second derivative of the kernel with respect y
    vectorized_dk_dxdx = vmap(vmap(dk_dxdx, (None, 0, None)), (0, None, None))

    k_t = vectorized_dk_dt(T,S,l_t) * vectorized_rbf_kernel_x(X,Y,params)
    k_xx = vectorized_dk_dxdx(X,Y,params) * vectorized_rbf_kernel_t(T,S,l_t)
    return k_t - alpha * k_xx

@jit
def gram_Matrix_jax(XT, YS, params, noise = [0,0]):
    """computes the gram matrix of the kernel
    params = [l_x, sigma_f_sq, l_t, alpha]
    noise = [noise_u, noise_f]
    """
    assert XT.shape[1] == 2, "X must be a 2d array"
    X, T = XT[:,0].reshape(-1,1), XT[:,1].reshape(-1,1)
    Y, S = YS[:,0].reshape(-1,1), YS[:,1].reshape(-1,1)

    #unfortunately we have to stack the data in a certain way to make the kernel work...
    X_u = jnp.hstack([X,T])
    X_f = jnp.hstack([Y,S])
    
    k_uu_matrix = k_uu_jax(X_u, X_u, params) + noise[0]**2 * jnp.eye(len(X)) #xxtt --- we have to slice xt xt
    k_uf_matrix = k_uf_jax(X_u, X_f, params)                              #xyts 
    k_fu_matrix = k_fu_jax(X_f, X_u, params) 
    k_ff_matrix = k_ff_jax(X_f, X_f, params) + noise[1]**2 * jnp.eye(len(Y))
    #combine all the matrices to the full gram matrix
    K = jnp.block([[k_uu_matrix, k_uf_matrix], [k_fu_matrix, k_ff_matrix]])
    return K
