#----------------------------------------
# author: Tobias Leitgeb
# This file contains a class for a physics informed GP regressor. 
#----------------------------------------
import GPy
from jax import numpy as jnp
from jax import jit, grad, vmap, jacfwd
import jax
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import qmc
from jax import scipy as jsp
jax.config.update('jax_platform_name', 'cpu')


class PhysicsInformedGP_regressor():
    """
    A class representing a Gaussian Process (GP) regressor with physics-informed properties.

    Attributes:
        _name_kernel (str): The name of the kernel used in the GP.
        gram_matrix (callable): The function that computes the Gram matrix of the kernel.
        k_uu (callable): The function that computes the kernel matrix between training inputs.
        k_uf (callable): The function that computes the kernel matrix between training inputs and test inputs.
        k_fu (callable): The function that computes the kernel matrix between test inputs and training inputs.
        k_ff (callable): The function that computes the kernel matrix between test inputs.
        timedependence (bool): Flag indicating whether the problem has time dependence.
        params (dict): Dictionary of hyperparameters for the kernel.
        MSE (dict): Dictionary to store the mean squared error for training and test data.
        rel_l2_error (dict): Dictionary to store the relative L2 error for training and test data.
        xlabel (str): The label for the x-axis in plots.
        ylabel (str): The label for the y-axis in plots.
        ARD (bool): Flag indicating whether Automatic Relevance Determination (ARD) is used.
        jitter (float): Small value added to the diagonal of the kernel matrix for numerical stability.
        D (int): The dimensionality of the problem.

    Methods:
        __init__(self, kernel: callable, timedependence: bool, params: list, Dimensions, ARD=True) -> None:
            Initializes the PhysicsInformedGP_regressor object.
        __str__(self) -> str:
            Returns a string representation of the PhysicsInformedGP_regressor object.
        set_name_kernel(self, name: str) -> None:
            Sets the name of the kernel.
        set_axis_labels(self, xlabel: str, ylabel: str) -> None:
            Sets the axis labels for the plots.
        set_training_data(self, filename: str, n_training_points, noise, seeds_training: list = [40, 14]) -> None:
            Sets the training data and the raw data.
        set_validation_data(self, n_validation_points) -> None:
            Sets the validation set.
        set_params(self, new_params: list) -> None:
            Sets the hyperparameters of the kernel.
        get_params(self) -> np.ndarray:
            Returns the hyperparameters of the kernel.
        log_marginal_likelohood(self, params) -> float:
            Computes the log marginal likelihood of the GP.
        log_marginal_likelihood_to_optimize(self) -> callable:
            Returns the log marginal likelihood as a function of the hyperparameters.
        grad_log_marginal_likelihood(self) -> callable:
            Computes the gradient of the log marginal likelihood.
        train(self, method: str, n_restarts: int, n_threads: int, opt_dictionary: dict) -> None:
            Optimizes the hyperparameters of the kernel.
        optimization_restarts_parallel_CG(self, n_restarts: int, n_threads: int, opt_dictionary: dict) -> dict:
            Performs the optimization of the hyperparameters in parallel using the CG method.
        optimization_restarts_parallel_TNC(self, n_restarts: int, n_threads: int, opt_dictionary: dict) -> dict:
            Performs the optimization of the hyperparameters in parallel using the TNC method.
        optimization_restarts_parallel_NM(self, n_restarts: int, n_threads: int, opt_dictionary: dict) -> dict:
            Performs the optimization of the hyperparameters in parallel using the Nelder-Mead method.
        optimization_restarts_parallel_LBFGSB(self, n_restarts: int, n_threads: int, opt_dictionary: dict) -> dict:
            Performs the optimization of the hyperparameters in parallel using the L-BFGS-B method.
    """
    
    _name_kernel = ""

    def __init__(self, kernel: callable, timedependence: bool, params: list,Dimensions, ARD = True) -> None:
        assert callable(kernel[0]), "Please provide a valid kernel"
        assert(type(timedependence) ==
               bool), "Please provide a valid boolean value for timedependence"

        self.gram_matrix = kernel[0]
        self.k_uu, self.k_uf, self.k_fu, self.k_ff = kernel[1], kernel[2], kernel[3], kernel[4]
        self.timedependence = timedependence
        self.params = dict(zip(params, 1*np.ones(len(params))))
        self.MSE = {"u": None, "f": None}
        self.rel_l2_error = {"u": None, "f": None}
        self.xlabel, self.ylabel = "", ""
        self.ARD = ARD
        self.jitter = 1e-7
        self.D = Dimensions
        

    def __str__(self) -> str:
        string = "-----------------------------------------------\n"
        string += "GP with kernel: " + str(self.__class__._name_kernel) + "\n"
        string += "Training data: " + str(self.X_u.shape) + "\n"
        string += "Training targets: " + str(self.targets.shape) + "\n"
        string += "Hyperparameters: " + str(self.params) + "\n"
        string += "Log marginal likelihood: " + str(self.result.fun) + "\n"
        string += "Mean squared error: " + str(self.MSE) + "\n"
        string += "Relative l2 error: " + str(self.rel_l2_error) + "\n"
        string += "-----------------------------------------------\n"
        return string

    def set_name_kernel(self, name: str):
        """sets the name of the kernel"""
        self._name_kernel = name
        pass

    def set_axis_labels(self, xlabel: str, ylabel: str):
        """sets the axis labels for the plots"""
        self.xlabel = xlabel
        self.ylabel = ylabel
        pass

    def set_training_data(self, filename: str, n_training_points, noise, seeds_training: list = [40, 14],n_training_points_f = None) -> None:
        """sets the training data and the raw data"""
        self.filename = filename
        if self.timedependence and self.D == 2:
            x_u, x_f, t_u, t_f, u_train, f_train, self.raw_data = self.get_data_set_2d(
                filename, n_training_points, noise, seeds_training)
            self.X_u = np.hstack([x_u, t_u])
            self.X_f = np.hstack([x_f, t_f])
            self.targets = np.concatenate([u_train, f_train])
            self.u_train = u_train
            self.f_train = f_train
            assert self.X_u.shape[1] == 2, "Please provide a valid training data set"

        elif not self.timedependence:
            x_u, u_train, x_f, f_train, self.raw_data = self.get_data_set_1d(
                filename, n_training_points,n_training_points_f, noise, seeds_training)
            self.X_u = x_u
            self.X_f = x_f
            self.targets = np.concatenate([u_train, f_train])
            self.u_train = u_train
            self.f_train = f_train

        elif self.D == 3:
            self.X_u, self.X_f, self.u_train, self.f_train, self.raw_data = self.get_data_set_3d(
                filename, n_training_points, noise)
            self.targets = np.concatenate([self.u_train, self.f_train])
    
        self.filename = filename
        self.noise = noise
        pass
#np.hstack([x_train_u, y_train_u, t_train_u]), np.hstack([x_train_f, y_train_f, t_train_f]), u_train, f_train, [x, y, u_matrix, f_matrix]
    def set_validation_data(self, n_validation_points):
        """sets the validation set"""
        #assert self.X_u is not None, "Please set the training data first"
        if self.timedependence and self.D == 2:
            self.validation_set = self.get_validation_set_2d(
                self.filename, n_validation_points, self.noise)
        elif self.D == 3:
            self.X_u_test, self.X_f_test, self.u_test, self.f_test, self.raw_data_test = self.get_val_set_3d(
                self.filename, n_validation_points, [1e-8,1e-8])
        else:
            self.validation_set = self.get_validation_set_1d(
                self.filename, n_validation_points, self.noise)
        pass

    def set_params(self, new_params: list):
        """sets the hyperparameters of the kernel"""
        for i, key in enumerate(self.params.keys()):
            self.params[key] = new_params[i]
        pass

    def get_params(self):
        """returns the hyperparameters of the kernel"""
        return np.array(list(self.params.values()))

    def log_marginal_likelohood(self, params):
        """computes the log marginal likelihood of the GP"""
        #First compute the whole block-kernel matrix as defined in chap 3)
        K = self.gram_matrix(self.X_u, self.X_f, params, self.noise) 
        #cholesky decomposition with extra jitter for numerical stability
        L = jnp.linalg.cholesky(K + self.jitter * jnp.eye(len(K)))
        #compute alpha as defined in chap 5.2)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, self.targets))
        mll = 1/2 * jnp.dot(self.targets.T, alpha) + 0.5*jnp.sum(
            jnp.log(jnp.diagonal(L))) + len(self.X_u)/2 * jnp.log(2*jnp.pi)
        return jnp.squeeze(mll)

    def log_marginal_likelihood_to_optimize(self):
        """returns the log marginal likelihood as a function of the hyperparameters, so it can be used in the optimization function"""
        def function_to_optimize(hyperparams):
            mll = self.log_marginal_likelohood(hyperparams)
            return mll
        return function_to_optimize
    
    def grad_log_marginal_likelihood(self):
        return grad(self.log_marginal_likelihood_to_optimize())


    def train(self, method: str, n_restarts: int, n_threads: int, opt_dictionary: dict) -> None:
        """optimizes the hyperparameters of the kernel"""
        assert self.X_u is not None, "Please set the training data first"
        assert method in ['L-BFGS-B', 'TNC',
                          "CG", "Nelder-Mead"], "Please choose a valid optimization method"
        assert n_restarts > 0, "Please choose a valid number of restarts"
        if method == "CG":
            opt_result = self.optimization_restarts_parallel_CG(
                n_restarts, n_threads, opt_dictionary)
        elif method == "TNC":
            opt_result = self.optimization_restarts_parallel_TNC(
                n_restarts, n_threads, opt_dictionary)
        elif method == "L-BFGS-B":
            opt_result = self.optimization_restarts_parallel_LBFGSB(
                n_restarts, n_threads, opt_dictionary)
        elif method == "Nelder-Mead":
            opt_result = self.optimization_restarts_parallel_NM(
                n_restarts, n_threads, opt_dictionary)
        self.set_params(opt_result.x)
        self.result = opt_result
        pass
    
    def optimization_restarts_parallel_CG(self, n_restarts: int, n_threads: int, opt_dictionary: dict) -> dict:
        """
            performs the optimization of the hyperparameters in parallel and returns the best result.
            n_restarts: number of restarts of the optimization
            n_threads: number of threads to use for the parallelization
            opt_dictionary: dictionary containing all the parameters needed for the optimization (initial values, bounds, etc.)
        """

        def single_optimization_run():
            """performs a single optimization run with random initialization of the hyperparameters"""
            theta_initial = opt_dictionary['theta_initial']()

            res = minimize(self.log_marginal_likelihood_to_optimize(), x0=theta_initial,
                           method='CG')
            return res

        results = Parallel(n_jobs=n_threads)(
            delayed(single_optimization_run)() for _ in tqdm(range(n_restarts)))
        # all positive parameters
        results = [res for res in results if np.all(res.x > 0) and res.success]
        self.result_list = results
        best_result = min(results, key=lambda x: x.fun)
        print(best_result)

        return best_result

    def optimization_restarts_parallel_TNC(self, n_restarts: int, n_threads: int, opt_dictionary: dict) -> dict:
        """
        performs the optimization of the hyperparameters in parallel and returns the best result.
        n_restarts: number of restarts of the optimization
        n_threads: number of threads to use for the parallelization
        opt_dictionary: dictionary containing all the parameters needed for the optimization (initial values, bounds, etc.)
        """

        def single_optimization_run():
            """performs a single optimization run with random initialization of the hyperparameters"""
            theta_initial = opt_dictionary['theta_initial']()
            res = minimize(self.log_marginal_likelihood_to_optimize(), x0=theta_initial,
                           method='TNC', jac=jit(self.grad_log_marginal_likelihood()), bounds=opt_dictionary['bounds'],
                           tol=opt_dictionary['gtol'])
            return res
        results = Parallel(n_jobs=n_threads)(
            delayed(single_optimization_run)() for _ in tqdm(range(n_restarts)))
        # all positive parameters
        results = [res for res in results if np.all(res.x > 0) and res.success]
        self.results_list = results
        best_result = min(results, key=lambda x: x.fun)
        print(best_result)

        return best_result
    def optimization_restarts_parallel_NM(self, n_restarts: int, n_threads: int, opt_dictionary: dict) -> dict:
        """
        performs the optimization of the hyperparameters in parallel and returns the best result.
        n_restarts: number of restarts of the optimization
        n_threads: number of threads to use for the parallelization
        opt_dictionary: dictionary containing all the parameters needed for the optimization (initial values, bounds, etc.)
        """

        def single_optimization_run():
            """performs a single optimization run with random initialization of the hyperparameters"""
            theta_initial = opt_dictionary['theta_initial']()
            res = minimize(self.log_marginal_likelihood_to_optimize(), x0=theta_initial,
                           method='Nelder-Mead',
                           tol=opt_dictionary['gtol'])
            return res
        results = Parallel(n_jobs=n_threads)(
            delayed(single_optimization_run)() for _ in tqdm(range(n_restarts)))
        # all positive parameters
        results = [res for res in results if np.all(res.x > 0) ]
        self.results_list = results
        best_result = min(results, key=lambda x: x.fun)
        print("Theta: ", best_result.x, "\n","Log marginal likelihood: ", best_result.fun)

        return best_result

    def optimization_restarts_parallel_LBFGSB(self, n_restarts: int, n_threads: int, opt_dictionary: dict) -> dict:
        """
        performs the optimization of the hyperparameters in parallel and returns the best result.
        n_restarts: number of restarts of the optimization
        n_threads: number of threads to use for the parallelization
        opt_dictionary: dictionary containing all the parameters needed for the optimization (initial values, bounds, etc.)
        """

        def single_optimization_run():
            """performs a single optimization run with random initialization of the hyperparameters"""
            theta_initial = opt_dictionary['theta_initial']()
            res = minimize(self.log_marginal_likelihood_to_optimize(), x0=theta_initial,
                           method='L-BFGS-B', bounds=opt_dictionary['bounds'],
                           options={'gtol': opt_dictionary['gtol']})
            return res
        results = Parallel(n_jobs=n_threads)(
            delayed(single_optimization_run)() for _ in tqdm(range(n_restarts)))
        # all positive parameters
        results = [res for res in results if np.all(res.x > 0) and res.success]
        self.results_list = results
        best_result = min(results, key=lambda x: x.fun)
        print(best_result)

        return best_result

    def predict_model(self, X_star):
        self.mean_u, self.var_u = self.predict_u(X_star)
        self.mean_f, self.var_f = self.predict_f(X_star)

    def predict_u(self, X_star):
        """calculates the mean and variance of the GP at the points X_star"""
        params = self.get_params()
        #compute the kernel matrix
        K = self.gram_matrix(self.X_u, self.X_f, params, self.noise) 
        #cholesky decomposition with extra jitter for numerical stability
        L = jnp.linalg.cholesky(K+ jnp.eye(len(K)) * self.jitter)
        #stack the kernel matrices as defined in eq) 40
        q_1 = self.k_uu(X_star, self.X_u, params) 
        q_2 = self.k_uf(X_star, self.X_f, params)
        kappa = jnp.hstack((q_1, q_2))
        #calculate alpha as defined in chap 5.2)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, self.targets))
        #calculate the predictive mean
        f_star = kappa@alpha
        #calculate predictive variance
        alpha_var = jnp.linalg.solve(L.T, jnp.linalg.solve(L, kappa.T))
        cov_f_star = self.k_uu(X_star, X_star, params) - kappa@alpha_var
        var = jnp.diag(cov_f_star)
        return f_star, var

    def predict_f(self, X_star):
        """predicts the mean and variance of the GP at the points X_star"""
        params = self.get_params()
        #compute the kernel matrix
        K = self.gram_matrix(self.X_u, self.X_f, params, self.noise)
        #cholesky decomposition with extra jitter for numerical stability
        L = jnp.linalg.cholesky(K + jnp.eye(len(K)) * self.jitter)
        #stack the kernel matrices as defined in eq) 40
        q_1 = self.k_fu(X_star, self.X_u, params)
        q_2 = self.k_ff(X_star, self.X_f, params) 
        kappa = jnp.hstack((q_1, q_2))
        #calculate alpha as defined in chap 5.2)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, self.targets))
        f_star = kappa@alpha
        #calculate predictive variance
        alpha_var = jnp.linalg.solve(L.T, jnp.linalg.solve(L, kappa.T))
        cov_f_star = self.k_ff(X_star, X_star, params) - kappa@alpha_var
        var = jnp.diag(cov_f_star)
        return f_star, var

    def error(self):
        """computes the mean squared error and the relative L^2 error of the computed model""" # [t_val_u, u_val_u, t_val_f, f_val_f]
        assert self.validation_set is not None, "Please set the validation set first"
        if self.timedependence:
            #[x_val_u, t_val_u, x_val_f, t_val_f, u_val, f_val]
            x_u, t_u = self.validation_set[0].reshape(
                -1, 1), self.validation_set[1].reshape(-1, 1)
            X_u_star = np.hstack((x_u, t_u))
            u_values = self.validation_set[4].reshape(-1, 1)

            x_f, t_f = self.validation_set[2].reshape(
                -1, 1), self.validation_set[3].reshape(-1, 1)
            X_f_star = np.hstack((x_f, t_f))
            f_values = self.validation_set[5].reshape(-1, 1)
        else:
            X_star = self.validation_set[0].reshape(-1, 1)
            u_values = self.validation_set[1].reshape(-1, 1)
            X_star_f = self.validation_set[2].reshape(-1, 1)
            f_values = self.validation_set[3].reshape(-1, 1)

        mean_ground_truth_u, var = self.predict_u(X_u_star)
        mean_ground_truth_f, var = self.predict_f(X_f_star)
        self.MSE["u"] = np.mean(
            (mean_ground_truth_u.ravel() - u_values.ravel())**2).item()
        self.MSE["f"] = np.mean(
            (mean_ground_truth_f.ravel() - f_values.ravel())**2).item()
        
        self.rel_l2_error["u"] = self.relative_l2_error(
            u_values.ravel(), mean_ground_truth_u.ravel())
        self.rel_l2_error["f"] = self.relative_l2_error(
            f_values.ravel(), mean_ground_truth_f.ravel())

    def relative_l2_error(self, ground_truth,predicted_solution):
        """computes the relative l2 error between the ground truth and the predicted solution"""
        assert ground_truth.shape == predicted_solution.shape, "Please provide a valid ground truth and predicted solution"
        return np.linalg.norm(ground_truth-predicted_solution)/np.linalg.norm(ground_truth)
    
    ##----------------------------- PLOTING FUNCTIONS ---------------------------------##
    def use_GPy(self, X_star, save_path: str = None, heat_map: bool = False):
        """uses the GPy library to compute the GP"""
        if not self.timedependence:
            kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
            model_GPy = GPy.models.GPRegression(self.X_u, self.u_train, kernel)
            model_GPy.Gaussian_noise.variance.fix(self.noise[0])
            model_GPy.optimize_restarts(num_restarts=20, verbose=False)

            # MSE error
            t_val, u_val = self.validation_set[0], self.validation_set[1]
            mean, var = model_GPy.predict(t_val.reshape(-1, 1))
            L2_u = self.relative_l2_error(u_val.ravel(), mean.ravel())
            self.rel_l2_error["u"] = self.relative_l2_error(
                u_val.ravel(), mean.ravel())
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            y_data, var = model_GPy.predict(X_star)
            ax[0].plot(X_star, y_data, label="GP prediction", color="blue")

            ax[0].fill_between(X_star.ravel(), y_data.ravel() - 2*np.sqrt(var.ravel()), y_data.ravel(
            ) + 2*np.sqrt(var.ravel()), color="blue", alpha=0.2, label="95% confidence interval")
            ax[0].scatter(self.validation_set[0], self.validation_set[1],
                          label="validation set", color="orange", marker="x", s=15)
            ax[0].scatter(self.X_u, self.u_train,
                          label="training points", color="red", marker="o", s=15)
            ax[0].set_xlim(0, max(X_star))
            ax[0].set_title("u(t) prediction")
            ax[0].grid(alpha=0.7)
            ax[0].legend()

            kernel2 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
            model_GPy2 = GPy.models.GPRegression(self.X_f, self.f_train, kernel2)
            model_GPy2.Gaussian_noise.variance.fix(self.noise[1])
            model_GPy2.optimize_restarts(num_restarts=20, verbose=False)
            # MSE error
            t_val, f_val = self.validation_set[2], self.validation_set[3]
            mean,var = model_GPy2.predict(t_val.reshape(-1,1))
            y_data, var = model_GPy2.predict(X_star)
            L2_f = self.relative_l2_error(f_val.ravel(), mean.ravel())
            self.rel_l2_error["f"] = self.relative_l2_error(
                f_val.ravel(), mean.ravel())
            ax[1].plot(X_star, y_data, label="GP prediction", color="blue")
            ax[1].fill_between(X_star.ravel(), y_data.ravel() - 2*np.sqrt(var.ravel()), y_data.ravel(
            ) + 2*np.sqrt(var.ravel()), alpha=0.2, color="blue", label="95% confidence interval")
            ax[1].scatter(self.validation_set[2], self.validation_set[3],
                          label="validation set", color="orange", marker="x", s=15)
            ax[1].scatter(self.X_f, self.f_train, label="training points",
                          color="blue", marker="o", s=15)
            ax[1].set_xlim(0, max(X_star))
            ax[1].set_title("f(t) prediction")
            ax[1].grid(alpha=0.7)
            ax[1].legend()
            plt.suptitle("GPy predictions")
            if save_path != None:
                plt.savefig(save_path)
            print("---------GPY--------")
            print("L2 u: ", L2_u)
            print("L2 f: ", L2_f)
            self.GPy_models = [model_GPy, model_GPy2]
            return L2_u, L2_f
            
        else:
            kernel_1 = GPy.kern.RBF(
                input_dim=2, variance=1., lengthscale=1., ARD=self.ARD)
            model_GPy = GPy.models.GPRegression(self.X_u, self.u_train, kernel_1)
            model_GPy.Gaussian_noise.variance.fix(self.noise[0])
            model_GPy.optimize_restarts(num_restarts=20, verbose=False)

            kernel_2 = GPy.kern.RBF(
                input_dim=2, variance=1., lengthscale=1., ARD=self.ARD)
            model_GPy_2 = GPy.models.GPRegression(
                self.X_f, self.f_train, kernel_2)
            model_GPy_2.Gaussian_noise.variance.fix(self.noise[1])
            model_GPy_2.optimize_restarts(num_restarts=20, verbose=False)
            mean, var = model_GPy.predict(X_star)
            original_shape = (int(np.sqrt(len(mean))), int(np.sqrt(len(mean))))
            mean = mean.reshape(original_shape)
            mean2, var2 = model_GPy_2.predict(X_star)
            mean2 = mean2.reshape(original_shape)
            self.GPy_models = [model_GPy, model_GPy_2]
            if heat_map:
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                cont1 = ax[0].imshow(mean, cmap='viridis', alpha=1, extent=[
                                     0, 1, 0, 1], origin='lower')
                #cont1 = ax[0].contourf(x_star.reshape(self.mean_u.shape), t_star.reshape(self.mean_u.shape), self.mean_u, cmap='viridis', alpha=0.8,levels = 1000)
                ax[0].set_title('Predictive mean for u(t,x)')
                ax[0].set_xlabel(self.xlabel)
                ax[0].set_ylabel(self.ylabel)
                ax[0].scatter(self.X_u[:, 0], self.X_u[:, 1], c='r', marker='o')

                cont2 = ax[1].imshow(mean2, cmap="viridis", alpha=1, extent=[
                                     0, 1, 0, 1], origin='lower')
                ax[1].scatter(self.X_f[:, 0], self.X_f[:, 1], c='b', marker='o')
                ax[1].set_title('Predictive mean for f(t,x)')
                ax[1].set_xlabel(self.xlabel)
                ax[1].set_ylabel(self.ylabel)
                plt.legend()
                plt.suptitle("GPy predictions")
                fig.colorbar(cont1, ax=ax[0])
                fig.colorbar(cont2, ax=ax[1])
                if save_path != None:
                    plt.savefig(save_path, bbox_inches='tight')
            else:
                fig, ax = plt.subplots(
                    1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))
                ax[0].plot_surface(X_star[:, 0].reshape(mean.shape), X_star[:, 1].reshape(
                    mean.shape), mean, cmap='viridis', edgecolor='none', alpha=0.5)
                ax[0].set_title('Predictive mean for u(t,x))')
                ax[0].set_xlabel(self.xlabel)
                ax[0].set_ylabel(self.ylabel)
                ax[0].scatter(self.X_u[:, 0], self.X_u[:, 1],
                              self.u_train, c='r', marker='o')

                ax[1].plot_surface(X_star[:, 0].reshape(mean2.shape), X_star[:, 1].reshape(
                    mean2.shape), mean2, cmap="viridis", edgecolor='none', alpha=0.5)
                ax[1].scatter(self.X_f[:, 0], self.X_f[:, 1],
                              self.f_train, c='b', marker='o')
                ax[1].set_title('Predictive mean for f(t,x)')
                ax[1].set_xlabel(self.xlabel)
                ax[1].set_ylabel(self.ylabel)
                plt.legend()
                plt.suptitle("GPy predictions")
                if save_path != None:
                    plt.savefig(save_path)

    def plot_prediction(self, X_star, title: str, save_path: str, heat_map: bool = False):
        """plots the prediction of the GP at the points X_star"""
        assert self.mean_u is not None, "Please predict the mean and variance first"
        assert self.var_u is not None, "Please predict the mean and variance first"
        assert self.mean_f is not None, "Please predict the mean and variance first"
        assert self.var_f is not None, "Please predict the mean and variance first"

        if self.timedependence:
            assert X_star.shape[1] == 2, "Please provide a valid test data set"
            x_star, t_star = X_star[:,
                                    0].reshape(-1, 1), X_star[:, 1].reshape(-1, 1)
            original_size = (int(np.sqrt(len(x_star))),
                             int(np.sqrt(len(x_star))))
            #x_star, t_star = np.meshgrid(x_star,t_star)
        else:
            assert X_star.shape[1] == 1, "Please provide a valid test data set"
            x_star = X_star

        if self.timedependence:
            self.mean_u = self.mean_u.reshape(original_size)
            self.mean_f = self.mean_f.reshape(original_size)

            if not heat_map:
                fig, ax = plt.subplots(
                    1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 5))
                ax[0].plot_surface(x_star.reshape(self.mean_u.shape), t_star.reshape(
                    self.mean_u.shape), self.mean_u, cmap='viridis', edgecolor='none', alpha=0.5)
                ax[0].set_title('(a)')
                ax[0].set_xlabel(self.xlabel)
                ax[0].set_ylabel(self.ylabel)
                ax[0].scatter(self.X_u[:, 0], self.X_u[:, 1],
                              self.u_train, c='r', marker='o')

                ax[1].plot_surface(x_star.reshape(self.mean_f.shape), t_star.reshape(
                    self.mean_f.shape), self.mean_f, cmap="viridis", edgecolor='none', alpha=0.5)
                ax[1].scatter(self.X_f[:, 0], self.X_f[:, 1],
                              self.f_train, c='b', marker='o')
                ax[1].set_title("(b)")
                ax[1].set_xlabel(self.xlabel)
                ax[1].set_ylabel(self.ylabel)
                plt.legend()
                # plt.suptitle(title)
                if save_path != None:
                    plt.savefig(save_path, bbox_inches='tight', dpi=250)

            if heat_map:
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                cont1 = ax[0].imshow(self.mean_u, cmap='viridis', alpha=1, extent=[
                                     0, 1, 0, 1], origin='lower')
                #cont1 = ax[0].contourf(x_star.reshape(self.mean_u.shape), t_star.reshape(self.mean_u.shape), self.mean_u, cmap='viridis', alpha=0.8,levels = 1000)
                ax[0].set_title('(a)')
                ax[0].set_xlabel(self.xlabel)
                ax[0].set_ylabel(self.ylabel)
                ax[0].scatter(self.X_u[:, 0], self.X_u[:, 1], c='r', marker='o')

                cont2 = ax[1].imshow(self.mean_f, cmap="viridis", alpha=1, extent=[
                                     0, 1, 0, 1], origin='lower')
                ax[1].scatter(self.X_f[:, 0], self.X_f[:, 1], c='b', marker='o')
                ax[1].set_title('(b)')
                ax[1].set_xlabel(self.xlabel)
                ax[1].set_ylabel(self.ylabel)
                plt.legend()
                # plt.suptitle(title)
                fig.colorbar(cont1, ax=ax[0])
                fig.colorbar(cont2, ax=ax[1])
                if save_path != None:
                    plt.savefig(save_path, bbox_inches='tight', dpi=250)

        else:

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            fontsize = 12
           # ax[0].scatter(self.validation_set[0], self.validation_set[1],
            #              c='orange', marker='x', label='Validation set', alpha=0.5, s=15)
            ax[0].plot(self.X_u, self.u_train, 'r.',
                       markersize=10, label='Observations')
            ax[0].plot(x_star, self.mean_u, 'b', label='Prediction')
            ax[0].fill_between(x_star.flatten(), self.mean_u.flatten() + 2 * np.sqrt(self.var_u.flatten()),
                               self.mean_u.flatten() - 2 * np.sqrt(self.var_u.flatten()), alpha=0.2, color='blue', label="2$\\sigma$")
            ax[0].plot(self.raw_data[0], self.raw_data[1],"--", label="Analytical solution",color="black")
            ax[0].legend(loc='upper right', fontsize=12)
            ax[0].set_xlabel("t", fontsize=fontsize)
            ax[0].set_ylabel("u(t)", fontsize=fontsize)
            ax[0].set_title("(a)", fontsize=fontsize)
            ax[0].grid(alpha=0.7)

            #ax[1].scatter(self.validation_set[2], self.validation_set[3],
             #             c='orange', marker='x', label='Validation set', alpha=0.5, s=15)
            ax[1].plot(self.X_f, self.f_train, 'r.',
                       markersize=10, label='Observations')
            ax[1].plot(x_star, self.mean_f, 'b', label='Prediction')
            ax[1].fill_between(x_star.flatten(), self.mean_f.flatten() - 2 * np.sqrt(self.var_f.flatten()),
                               self.mean_f.flatten() + 2 * np.sqrt(self.var_f.flatten()), alpha=0.2, color='blue', label="2$\\sigma$")
            ax[1].plot(self.raw_data[0], self.raw_data[2],"--", label="Analytical solution",color ="black")
            ax[1].legend(loc='upper right', fontsize=12)
            ax[1].set_xlabel("t", fontsize=fontsize)
            ax[1].set_ylabel("f(t)", fontsize=fontsize)
            ax[1].set_title("(b)", fontsize=fontsize)
            ax[1].grid(alpha=0.7)
            plt.tight_layout()
            # fig.suptitle(title)
            if save_path != None:
                plt.savefig(save_path, bbox_inches='tight')
        pass

    def plot_difference(self, title: str, save_path: str):
        """plots the difference between the analytical solution and the predicted mean"""
        assert self.timedependence == True, "Contourplot of difference only for 2d case"
        data = self.raw_data
        x_star, t_star = self.raw_data[0].reshape(
            -1, 1), self.raw_data[1].reshape(-1, 1)
        u_grid = self.raw_data[2]
        f_grid = self.raw_data[3]
        size = (int(np.sqrt(len(x_star))), int(np.sqrt(len(x_star))))
        mean_u, var = self.predict_u(np.hstack((x_star, t_star)))
        mean_f, var = self.predict_f(np.hstack((x_star, t_star)))
        mean_u, mean_f = mean_u.reshape(size), mean_f.reshape(size)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        #cont = ax[0].contourf(x_star.reshape(mean_u.shape), t_star.reshape(mean_u.shape), np.abs(mean_u - u_grid.reshape(size)), cmap='viridis', alpha=0.8,levels = 100)
        cont = ax[0].imshow(np.abs(mean_u - u_grid.reshape(size)),
                            cmap='viridis', alpha=1, extent=[0, 1, 0, 1], origin='lower')
        ax[0].set_title(' |$f_*$ - u(t,x)|')
        ax[0].set_xlabel(self.xlabel)
        ax[0].set_ylabel(self.ylabel)
        ax[0].scatter(self.X_u[:, 0], self.X_u[:, 1], c='r', marker='o')
        fig.colorbar(cont, ax=ax[0])

        #cont2 = ax[1].contourf(x_star.reshape(mean_f.shape), t_star.reshape(mean_f.shape), np.abs(mean_f - f_grid.reshape(size)), cmap="viridis", alpha=0.8,levels = 100)
        cont2 = ax[1].imshow(np.abs(mean_f - f_grid.reshape(size)),
                             cmap='viridis', alpha=1, extent=[0, 1, 0, 1], origin='lower')
        ax[1].scatter(self.X_f[:, 0], self.X_f[:, 1], c='b', marker='o')
        ax[1].set_title(' |$f_*$ - f(t,x)|')
        ax[1].set_xlabel(self.xlabel)
        ax[1].set_ylabel(self.ylabel)
        fig.colorbar(cont2, ax=ax[1])
        plt.suptitle(title)
        if save_path != None:
            plt.savefig(save_path, bbox_inches='tight')

    # def plot_prediction_and_analytical
    def plot_difference_GPy(self, title, save_path):
        assert self.timedependence == True, "Contourplot of difference only for 2d case"
        data = self.raw_data
        x_star, t_star = self.raw_data[0].reshape(
            -1, 1), self.raw_data[1].reshape(-1, 1)
        u_grid = self.raw_data[2]
        f_grid = self.raw_data[3]
        size = (int(np.sqrt(len(x_star))), int(np.sqrt(len(x_star))))
        mean_u, var = self.GPy_models[0].predict(np.hstack((x_star, t_star)))
        mean_f, var = self.GPy_models[1].predict(np.hstack((x_star, t_star)))
        mean_u, mean_f = mean_u.reshape(size), mean_f.reshape(size)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        #cont = ax[0].contourf(x_star.reshape(mean_u.shape), t_star.reshape(mean_u.shape), np.abs(mean_u - u_grid.reshape(size)), cmap='viridis', alpha=0.8,levels = 100)
        cont = ax[0].imshow(np.abs(mean_u - u_grid.reshape(size)),
                            cmap='viridis', alpha=1, extent=[0, 1, 0, 1], origin='lower')
        ax[0].set_title(' |$f_*$ - u(t,x)|')
        ax[0].set_xlabel(self.xlabel)
        ax[0].set_ylabel(self.ylabel)
        ax[0].scatter(self.X_u[:, 0], self.X_u[:, 1], c='r', marker='o')
        fig.colorbar(cont, ax=ax[0])

        #cont2 = ax[1].contourf(x_star.reshape(mean_f.shape), t_star.reshape(mean_f.shape), np.abs(mean_f - f_grid.reshape(size)), cmap="viridis", alpha=0.8,levels = 100)
        cont2 = ax[1].imshow(np.abs(mean_f - f_grid.reshape(size)),
                             cmap='viridis', alpha=1, extent=[0, 1, 0, 1], origin='lower')
        ax[1].scatter(self.X_f[:, 0], self.X_f[:, 1], c='b', marker='o')
        ax[1].set_title(' |$f_*$ - f(t,x)|')
        ax[1].set_xlabel(self.xlabel)
        ax[1].set_ylabel(self.ylabel)
        fig.colorbar(cont2, ax=ax[1])
        plt.suptitle(title)
        if save_path != None:
            plt.savefig(save_path, bbox_inches='tight')
        MSE_u = np.mean((mean_u - u_grid.reshape(size))**2).item()
        MSE_f = np.mean((mean_f - f_grid.reshape(size))**2).item()
        print("MSE_u: ", MSE_u)
        print("MSE_f: ", MSE_f)
        print("relative error u: ", self.relative_l2_error(
            u_grid.reshape(size), mean_u))
        print("relative error f: ", self.relative_l2_error(
            f_grid.reshape(size), mean_f))

    def plot_variance(self, X_star, title: str, save_path: str):
        """plots the variance of the GP at the points X_star"""
        assert self.timedependence == True, "Contourplot of variance only for 2d case"
        assert self.mean_u is not None, "Please predict the mean and variance first"
        assert self.var_u is not None, "Please predict the mean and variance first"
        assert self.mean_f is not None, "Please predict the mean and variance first"
        assert self.var_f is not None, "Please predict the mean and variance first"

        x_star, t_star = X_star[:,
                                0].reshape(-1, 1), X_star[:, 1].reshape(-1, 1)
        original_size = (int(np.sqrt(len(x_star))), int(np.sqrt(len(x_star))))
        self.var_u = self.var_u.reshape(original_size)
        self.var_f = self.var_f.reshape(original_size)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        #cont_0 = ax[0].contourf(x_star.reshape(self.var_u.shape), t_star.reshape(self.var_u.shape), np.sqrt(self.var_u), cmap='viridis', alpha=0.8)
        cont_0 = ax[0].imshow(np.sqrt(self.var_u), cmap='viridis', alpha=1, extent=[
                              0, 1, 0, 1], origin='lower')
        ax[0].set_title('Predictive std for u(t,x))')
        ax[0].set_xlabel(self.xlabel)
        ax[0].set_ylabel(self.ylabel)
        ax[0].scatter(self.X_u[:, 0], self.X_u[:, 1], marker='o')
        fig.colorbar(cont_0, ax=ax[0])

        #cont_1 = ax[1].contourf(x_star.reshape(self.var_f.shape), t_star.reshape(self.var_f.shape), np.sqrt(self.var_f), cmap="viridis", alpha=0.8)
        cont_1 = ax[1].imshow(np.sqrt(self.var_f), cmap='viridis', alpha=1, extent=[
                              0, 1, 0, 1], origin='lower')
        ax[1].scatter(self.X_f[:, 0], self.X_f[:, 1], marker='o')
        ax[1].set_title('Predictive std for f(t,x)')
        ax[1].set_xlabel(self.xlabel)
        ax[1].set_ylabel(self.ylabel)
        fig.colorbar(cont_1, ax=ax[1])

        plt.suptitle(title)
        if save_path != None:
            plt.savefig(save_path, bbox_inches='tight')

    def plot_variance_GPy(self, title, save_path=None):
        assert self.timedependence == True, "Contourplot of difference only for 2d case"
        data = self.raw_data
        x_star, t_star = self.raw_data[0].reshape(
            -1, 1), self.raw_data[1].reshape(-1, 1)
        u_grid = self.raw_data[2]
        f_grid = self.raw_data[3]
        size = (int(np.sqrt(len(x_star))), int(np.sqrt(len(x_star))))
        mean_u, var_u = self.GPy_models[0].predict(np.hstack((x_star, t_star)))
        mean_f, var_f = self.GPy_models[1].predict(np.hstack((x_star, t_star)))
        var_u, var_f = var_u.reshape(u_grid.shape), var_f.reshape(u_grid.shape)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        #cont = ax[0].contourf(x_star.reshape(u_grid.shape), t_star.reshape(u_grid.shape), var_u, cmap='viridis', alpha=0.8)
        cont = ax[0].imshow(var_u, cmap='viridis', alpha=1,
                            extent=[0, 1, 0, 1], origin='lower')
        ax[0].set_title('GPy std for u(t,x)')
        ax[0].set_xlabel(self.xlabel)
        ax[0].set_ylabel(self.ylabel)
        ax[0].scatter(self.X_u[:, 0], self.X_u[:, 1], c='r', marker='o')
        fig.colorbar(cont, ax=ax[0])

        #cont2 = ax[1].contourf(x_star.reshape(u_grid.shape), t_star.reshape(u_grid.shape), var_f, cmap="viridis", alpha=0.8)
        cont2 = ax[1].imshow(var_f, cmap='viridis', alpha=1, extent=[
                             0, 1, 0, 1], origin='lower')
        ax[1].scatter(self.X_f[:, 0], self.X_f[:, 1], c='b', marker='o')
        ax[1].set_title(' GPy std for f(t,x)')
        ax[1].set_xlabel(self.xlabel)
        ax[1].set_ylabel(self.ylabel)
        fig.colorbar(cont2, ax=ax[1])
        plt.suptitle(title)
        if save_path != None:
            plt.savefig(save_path, bbox_inches='tight')

    
    def plot_raw_data(self, Training_points=False, heat_map=False):
        if self.timedependence:
            x_star, t_star = self.raw_data[0].reshape(
                -1, 1), self.raw_data[1].reshape(-1, 1)
            u_grid = self.raw_data[2]
            f_grid = self.raw_data[3]
            size = (int(np.sqrt(len(x_star))), int(np.sqrt(len(x_star))))
            if heat_map:
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                #cont = ax[0].contour(x_star.reshape(size), t_star.reshape(size), u_grid.reshape(size), cmap='viridis', alpha=0.8, levels = 100)
                cont = IM = ax[0].imshow(u_grid.reshape(
                    size), cmap="viridis", origin='lower', extent=(0, 1, 0, 1))
                ax[0].set_title('u(t,x)')
                ax[0].set_xlabel(self.xlabel)
                ax[0].set_ylabel(self.ylabel)
                #cont2 = ax[1].contourf(x_star.reshape(size), t_star.reshape(size), f_grid.reshape(size), cmap="viridis", alpha=0.8,levels = 100)
                cont2 = ax[1].imshow(f_grid.reshape(
                    size), cmap="viridis", origin='lower', extent=(0, 1, 0, 1))
                ax[1].set_title('f(t,x)')
                ax[1].set_xlabel(self.xlabel)
                ax[1].set_ylabel(self.ylabel)
                fig.colorbar(cont, ax=ax[0])
                fig.colorbar(cont2, ax=ax[1])
                if Training_points:
                    ax[0].scatter(self.X_u[:, 0], self.X_u[:, 1], c='r',
                                  marker='o', label='Training points')
                    ax[1].scatter(self.X_f[:, 0], self.X_f[:, 1], c='b',
                                  marker='o', label='Training points')
            else:
                fig, ax = plt.subplots(1, 2, figsize=(
                    12, 5), subplot_kw={"projection": "3d"})
                cont = ax[0].plot_surface(x_star.reshape(size), t_star.reshape(
                    size), u_grid.reshape(size), cmap='viridis', alpha=0.8, edgecolor='none')
                ax[0].set_title('u(t,x)')
                ax[0].set_xlabel(self.xlabel)
                ax[0].set_ylabel(self.ylabel)

                cont2 = ax[1].plot_surface(x_star.reshape(size), t_star.reshape(
                    size), f_grid.reshape(size), cmap="viridis", alpha=0.8)
                ax[1].set_title('f(t,x)')
                ax[1].set_xlabel(self.xlabel)
                ax[1].set_ylabel(self.ylabel)

                if Training_points:
                    ax[0].scatter(self.X_u[:, 0], self.X_u[:, 1], self.u_train,
                                  c='r', marker='o', label='Training points')
                    ax[1].scatter(self.X_f[:, 0], self.X_f[:, 1], self.f_train,
                                  c='b', marker='o', label='Training points')
            plt.legend()
        else:
            x_star = self.raw_data[0]
            u_grid = self.raw_data[1]
            f_grid = self.raw_data[2]
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].plot(x_star, u_grid, markersize=10, label='Observations')
            ax[0].set_xlabel(self.xlabel)
            ax[0].set_ylabel(self.ylabel)
            ax[0].set_title("u(t) raw data")
            ax[0].grid(alpha=0.7)
            ax[0].legend()
            ax[1].plot(x_star, f_grid, markersize=10, label='Observations')
            ax[1].set_xlabel(self.xlabel)
            ax[1].set_ylabel(self.ylabel)
            ax[1].set_title("f(t) raw data")
            ax[1].grid(alpha=0.7)
            ax[1].legend()
            
            if Training_points:
                ax[0].scatter(self.X_u, self.u_train, c='r',
                              marker='o', label='Training points')
                ax[1].scatter(self.X_f, self.f_train, c='b',
                              marker='o', label='Training points')
            plt.legend()

    def plot_validation_set(self):
        assert self.validation_set is not None, "Please set the validation set first"
        if self.timedependence:
            x_star, t_star = self.validation_set[0].reshape(
                -1, 1), self.validation_set[1].reshape(-1, 1)
            u_values = self.validation_set[4]
            f_values = self.validation_set[5]
            size = (int(np.sqrt(len(x_star))), int(np.sqrt(len(x_star))))
            fig, ax = plt.subplots(1, 2, figsize=(
                12, 5), subplot_kw={"projection": "3d"})
            cont = ax[0].plot_surface(x_star.reshape(size), t_star.reshape(
                size), u_values.reshape(size), cmap='viridis', alpha=0.8, edgecolor='none')
            ax[0].set_title('u(t,x)')
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('t')
            fig.colorbar(cont, ax=ax[0])

            x_star, t_star = self.validation_set[2].reshape(
                -1, 1), self.validation_set[3].reshape(-1, 1)
            cont2 = ax[1].plot_surface(x_star.reshape(size), t_star.reshape(
                size), f_values.reshape(size), cmap="viridis", alpha=0.8)
            ax[1].set_title('f(t,x)')
            ax[1].set_xlabel('x')
            ax[1].set_ylabel('t')
            fig.colorbar(cont2, ax=ax[1])
        else:
            x_star_u = self.validation_set[0]
            u_values = self.validation_set[1]
            x_star_f = self.validation_set[2]
            f_values = self.validation_set[3]
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].scatter(x_star_u, u_values, label='Observations', color = "orange", marker = "x", s = 15)
            ax[0].set_xlabel('$t$')
            ax[0].set_ylabel('$u(t)$')
            ax[0].set_title("u(t) validation set")
            ax[0].plot(self.raw_data[0], self.raw_data[1],"--",color = "green", label="Analytical solution")
            ax[0].grid(alpha=0.7)
            ax[0].legend()

            ax[1].scatter(x_star_f, f_values, label='Observations', color = "orange", marker = "x", s = 15)
            ax[1].set_xlabel('$t$')
            ax[1].set_ylabel('$f(t)$')
            ax[1].set_title("f(t) validation set")
            ax[1].plot(self.raw_data[0], self.raw_data[2],"--",color = "green", label="Analytical solution")
            ax[1].grid(alpha=0.7)
            ax[1].legend()
    
    @staticmethod
    def get_data_set_1d(filename, n_training_points, n_training_points_f, noise: list, seeds=[50, 41]):
        try:
            df = pd.read_csv(filename)
        except:
            df = pd.read_csv(filename)

        t = df['t'].values
        u = df['u'].values
        f = df['f'].values
        
        d = 1  # number of dimensions in your Sobol sequence. You have a 2D grid, so d=2
        # training data for u(t,x)
        engine = qmc.Sobol(d, seed=seeds[0], scramble=True)
        sample = engine.random(n_training_points)
        # sample is in [0,1]^d, so we need to scale it to the range of x and t
        np.random.seed(1)
        indices = sample * np.array(len(t))
        indices = np.floor(indices).astype(int)
        t_train_u = t[indices]
        u_train_u = u[indices] + \
            np.random.normal(0, np.sqrt(noise[0]), u[indices].shape)

        # training data for f(t)
        np.random.seed(2)
        engine = qmc.Sobol(d, seed=seeds[1])
        sample = engine.random(n_training_points_f)
        indices = sample * np.array(len(t))
        indices = np.floor(indices).astype(int)
        t_train_f = t[indices]
        f_train_f = f[indices] + \
            np.random.normal(0, np.sqrt(noise[1]), f[indices].shape)

        return t_train_u.reshape(-1, 1), u_train_u.reshape(-1, 1), t_train_f.reshape(-1, 1), f_train_f.reshape(-1, 1), [t, u, f]

    @staticmethod
    def get_validation_set_1d(filename, n_validation_points, noise: list):
        try:
            df = pd.read_csv(filename)
        except:
            df = pd.read_csv(filename)

        t = df['t'].values
        u = df['u'].values
        f = df['f'].values

        d = 1  # number of dimensions in your Sobol sequence. You have a 2D grid, so d=2
        # training data for u(t,x)
        engine = qmc.Sobol(d, seed=50)
        sample = engine.random(n_validation_points)
        # sample is in [0,1]^d, so we need to scale it to the range of x and t
        indices = sample * np.array(len(t))
        indices = np.floor(indices).astype(int)
        t_val_u = t[indices]
        u_val_u = u[indices] + np.random.normal(0, np.sqrt(noise[0]), u[indices].shape)

        # training data for f(t)
        engine = qmc.Sobol(d, seed=7)
        sample = engine.random(n_validation_points)
        indices = sample * np.array(len(t))
        indices = np.floor(indices).astype(int)
        t_val_f = t[indices]
        f_val_f = f[indices] + np.random.normal(0, np.sqrt(noise[1]), f[indices].shape)

        return [t_val_u, u_val_u, t_val_f, f_val_f]

    @staticmethod
    def get_data_set_2d(filename, n_training_points, noise: list,seeds):
        # load data from mathematica calculation
        try:
            df = pd.read_csv(filename)
        except:
            df = pd.read_csv(filename)
        x = df['space'].values
        t = df['time'].values
        u = df['u'].values
        f = df['f'].values

        x_mesh = x.reshape(int(np.sqrt(len(x))), int(np.sqrt(len(x))))
        t_mesh = t.reshape(x_mesh.shape)
        u_grid = u.reshape(x_mesh.shape)
        f_grid = f.reshape(x_mesh.shape)

        #
        x_axis = np.unique(x_mesh)
        t_axis = np.unique(t_mesh)

        d = 2  # number of dimensions in your Sobol sequence. You have a 2D grid, so d=2
        # training data for u(t,x)
        engine = qmc.Sobol(d, seed=seeds[0])
        sample = engine.random(n_training_points)
        # sample is in [0,1]^d, so we need to scale it to the range of x and t
        indices = sample * np.array([len(x_axis), len(t_axis)])
        indices = np.floor(indices).astype(int)
        x_train_u = x_axis[indices[:, 0]]
        t_train_u = t_axis[indices[:, 1]]
        u_train = u_grid[indices[:, 1], indices[:, 0]] + \
            np.random.normal(
                0, np.sqrt(noise[0]), u_grid[indices[:, 1], indices[:, 0]].shape)

        # same thing for f(t,x)
        engine = qmc.Sobol(d, seed=seeds[1])
        sample = engine.random(n_training_points)
        indices = sample * np.array([len(x_axis), len(t_axis)])
        indices = np.floor(indices).astype(int)
        x_train_f = x_axis[indices[:, 0]]
        t_train_f = t_axis[indices[:, 1]]
        f_train = f_grid[indices[:, 1], indices[:, 0]] + \
            np.random.normal(
                0, np.sqrt(noise[1]), f_grid[indices[:, 1], indices[:, 0]].shape)

        return x_train_u.reshape(-1, 1), x_train_f.reshape(-1, 1), t_train_u.reshape(-1, 1), t_train_f.reshape(-1, 1), u_train.reshape(-1, 1), f_train.reshape(-1, 1), [x_mesh, t_mesh, u_grid, f_grid]


    @staticmethod
    def get_data_set_3d(filename, n_training_points, noise: list,t_max = 4):
        data = pd.read_csv('u_matrix_with_time_columns.csv', delimiter=',')
        meshdata = pd.read_csv('spatial_grid.csv', delimiter=',')
        fdata = pd.read_csv('f_matrix.csv', delimiter=',')

        grid = meshdata.values
        u_matrix = data.values
        u_matrix = np.insert(u_matrix, 0, 0, axis=0)
            
        f_matrix = fdata.values
        f_matrix = np.insert(f_matrix, 0, 0, axis=0)
        
        x = np.linspace(0, 2,101)
        y = np.linspace(0, 2,101)
        X, Y = np.meshgrid(x,y)
        u_matrix = u_matrix.reshape(101,101,-1)
        f_matrix = f_matrix.reshape(101,101,-1)

        engine = qmc.Sobol(3, seed=77)
        sample = engine.random(n_training_points)

        indices = sample * np.array([len(x), len(y), len(u_matrix[0,0,:])])
        indices = np.floor(indices).astype(int)
        indices = indices[np.argsort(indices[:, 2])]
        x_train_u = x[indices[:, 0]].reshape(-1,1)
        y_train_u = y[indices[:, 1]].reshape(-1,1)
        t_train_u = np.linspace(0,t_max,len(u_matrix[0,0,:]))[indices[:, 2]].reshape(-1,1)
        u_train = u_matrix[indices[:, 0], indices[:, 1], indices[:, 2]].reshape(-1,1)

        engine = qmc.Sobol(3, seed=10)
        sample = engine.random(n_training_points)

        indices = sample * np.array([len(x), len(y), len(f_matrix[0,0,:])])
        indices = np.floor(indices).astype(int)
        indices = indices[np.argsort(indices[:, 2])]
        x_train_f = x[indices[:, 0]].reshape(-1,1)
        y_train_f = y[indices[:, 1]].reshape(-1,1)
        t_train_f = np.linspace(0,t_max,len(f_matrix[0,0,:]))[indices[:, 2]].reshape(-1,1)
        f_train = f_matrix[indices[:, 0], indices[:, 1], indices[:, 2]].reshape(-1,1)

        return np.hstack([x_train_u, y_train_u, t_train_u]), np.hstack([x_train_f, y_train_f, t_train_f]), u_train, f_train, [x, y, u_matrix, f_matrix]
    
    @staticmethod
    def get_val_set_3d(filename, n_training_points, noise: list,t_max = 4):
        data = pd.read_csv('u_matrix_with_time_columns.csv', delimiter=',')
        meshdata = pd.read_csv('spatial_grid.csv', delimiter=',')
        fdata = pd.read_csv('f_matrix.csv', delimiter=',')

        grid = meshdata.values
        u_matrix = data.values
        u_matrix = np.insert(u_matrix, 0, 0, axis=0)
            
        f_matrix = fdata.values
        f_matrix = np.insert(f_matrix, 0, 0, axis=0)
        
        x = np.linspace(0, 2,101)
        y = np.linspace(0, 2,101)
        X, Y = np.meshgrid(x,y)
        u_matrix = u_matrix.reshape(101,101,-1)
        f_matrix = f_matrix.reshape(101,101,-1)

        engine = qmc.Sobol(3, seed=89)
        sample = engine.random(n_training_points)

        indices = sample * np.array([len(x), len(y), len(u_matrix[0,0,:])])
        indices = np.floor(indices).astype(int)
        indices = indices[np.argsort(indices[:, 2])]
        x_train_u = x[indices[:, 0]].reshape(-1,1)
        y_train_u = y[indices[:, 1]].reshape(-1,1)
        t_train_u = np.linspace(0,t_max,len(u_matrix[0,0,:]))[indices[:, 2]].reshape(-1,1)
        u_train = u_matrix[indices[:, 0], indices[:, 1], indices[:, 2]].reshape(-1,1)

        engine = qmc.Sobol(3, seed=54)
        sample = engine.random(n_training_points)

        indices = sample * np.array([len(x), len(y), len(f_matrix[0,0,:])])
        indices = np.floor(indices).astype(int)
        indices = indices[np.argsort(indices[:, 2])]
        x_train_f = x[indices[:, 0]].reshape(-1,1)
        y_train_f = y[indices[:, 1]].reshape(-1,1)
        t_train_f = np.linspace(0,t_max,len(f_matrix[0,0,:]))[indices[:, 2]].reshape(-1,1)
        f_train = f_matrix[indices[:, 0], indices[:, 1], indices[:, 2]].reshape(-1,1)

        return np.hstack([x_train_u, y_train_u, t_train_u]), np.hstack([x_train_f, y_train_f, t_train_f]), u_train, f_train, [x, y, u_matrix, f_matrix]

    @staticmethod
    def get_validation_set_2d(filename, n_validation_points, noise: list):
        try:
            df = pd.read_csv(filename)
        except:
            df = pd.read_csv(filename)
        x = df['space'].values
        t = df['time'].values
        u = df['u'].values
        f = df['f'].values

        x_mesh = x.reshape(int(np.sqrt(len(x))), int(np.sqrt(len(x))))
        t_mesh = t.reshape(x_mesh.shape)
        u_grid = u.reshape(x_mesh.shape)
        f_grid = f.reshape(x_mesh.shape)

        x_axis = np.unique(x_mesh)
        t_axis = np.unique(t_mesh)

        d = 2  # number of dimensions in your Sobol sequence. You have a 2D grid, so d=2
        # training data for u(t,x)
        engine = qmc.Sobol(d)
        sample = engine.random(n_validation_points)
        # sample is in [0,1]^d, so we need to scale it to the range of x and t
        indices = sample * np.array([len(x_axis), len(t_axis)])
        indices = np.floor(indices).astype(int)
        x_val_u = x_axis[indices[:, 0]]
        t_val_u = t_axis[indices[:, 1]]
        u_val = u_grid[indices[:, 1], indices[:, 0]] + \
            np.random.normal(
                0, np.sqrt(noise[0]), u_grid[indices[:, 1], indices[:, 0]].shape)

        # same thing for f(t,x)
        engine = qmc.Sobol(d)
        sample = engine.random(n_validation_points)
        indices = sample * np.array([len(x_axis), len(t_axis)])
        indices = np.floor(indices).astype(int)
        x_val_f = x_axis[indices[:, 0]]
        t_val_f = t_axis[indices[:, 1]]
        f_val = f_grid[indices[:, 1], indices[:, 0]] + \
            np.random.normal(
                0, np.sqrt(noise[1]), f_grid[indices[:, 1], indices[:, 0]].shape)

        return [x_val_u, t_val_u, x_val_f, t_val_f, u_val, f_val]

       # just a function to make different combination of the plots
    def plot_merged_1d_plots(self, X_star, path, figsize=(15, 20)):
        font_size = 12
        fig, ax = plt.subplots(2, 2, figsize=figsize)
        ax = ax.flatten()

        ax[0].plot(X_star, self.mean_u, 'b', label='$\\mu_u$')
        ax[0].fill_between(X_star.flatten(), self.mean_u.flatten() + 2 * np.sqrt(self.var_u.flatten()),
                           self.mean_u.flatten() - 2 * np.sqrt(self.var_u.flatten()), alpha=0.2, color='blue', label="$2\\sigma$")
        #ax[0].scatter(self.validation_set[0], self.validation_set[1],
        #              c='orange', marker='x', label='Validation set', alpha=0.5, s=15)
        ax[0].plot(self.X_u, self.u_train, 'r.',
                   markersize=10, label='training points')
        ax[0].plot(self.raw_data[0], self.raw_data[1],"--", label="Analytical solution",color = "black")
        ax[0].legend(loc='upper right', fontsize=10)
        ax[0].set_xlabel("t", fontsize=font_size)
        ax[0].set_ylabel("u(t)", fontsize=font_size)
        ax[0].set_title("(a)", fontsize=font_size)
        ax[0].grid(alpha=0.7)

        ax[1].plot(X_star, self.mean_f, 'b', label='$\\mu_f$')
        ax[1].fill_between(X_star.flatten(), self.mean_f.flatten() - 2 * np.sqrt(self.var_f.flatten()),
                           self.mean_f.flatten() + 2 * np.sqrt(self.var_f.flatten()), alpha=0.2, color='blue', label="$2\\sigma$")
        #ax[1].scatter(self.validation_set[2], self.validation_set[3],
        #              c='orange', marker='x', label='Validation set', alpha=0.5, s=15)
        ax[1].plot(self.X_f, self.f_train, 'r.',
                   markersize=10, label='training points')
        ax[1].plot(self.raw_data[0], self.raw_data[2],"--", label="Analytical solution",color = "black")
        ax[1].legend(loc='upper right', fontsize=10)
        ax[1].set_xlabel("t", fontsize=font_size)
        ax[1].set_ylabel("f(t)", fontsize=font_size)
        ax[1].set_title("(b)", fontsize=font_size)
        ax[1].grid(alpha=0.7)

        kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        model_GPy = GPy.models.GPRegression(self.X_u, self.u_train, kernel)
        model_GPy.Gaussian_noise.variance.fix(self.noise[0])
        model_GPy.optimize_restarts(num_restarts=20, verbose=False)

        # L_sq error
        t_val, u_val = self.validation_set[0], self.validation_set[1]
        mean, var = model_GPy.predict(t_val.reshape(-1, 1))
        L_sq_u = self.relative_l2_error(u_val, mean)
        y_data, var = model_GPy.predict(X_star)

        ax[2].plot(X_star, y_data, label="$\\mu_u$", color="blue")
        ax[2].fill_between(X_star.ravel(), y_data.ravel() - 2*np.sqrt(var.ravel()),
                           y_data.ravel() + 2*np.sqrt(var.ravel()), color="blue", alpha=0.2, label="$2\\sigma$")
        #ax[2].scatter(self.validation_set[0], self.validation_set[1],
        #              label="validation set", color="orange", marker="x", s=15)
        ax[2].plot(self.X_u, self.u_train, 'r.',
                   markersize=10, label='training points')
        ax[2].plot(self.raw_data[0], self.raw_data[1],"--", label="Analytical solution",color = "black")
        ax[2].set_xlabel("t", fontsize=font_size)
        ax[2].set_ylabel("u(t)", fontsize=font_size)
        ax[2].grid(alpha=0.7)
        ax[2].legend()
        ax[2].set_title("(c)", fontsize=font_size)

        kernel2 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        model_GPy2 = GPy.models.GPRegression(self.X_f, self.f_train, kernel2)
        model_GPy2.Gaussian_noise.variance.fix(self.noise[1])
        model_GPy2.optimize_restarts(num_restarts=20, verbose=False)
        # L_sq error
        t_val, f_val = self.validation_set[2], self.validation_set[3]
        mean,var = model_GPy2.predict(t_val.reshape(-1,1))
        y_data, var = model_GPy2.predict(X_star)
        L_sq_f = self.relative_l2_error(f_val, mean)

        ax[3].plot(X_star, y_data, label="$\\mu_f$", color="blue")
        ax[3].fill_between(X_star.ravel(), y_data.ravel() - 2*np.sqrt(var.ravel()),
                           y_data.ravel() + 2*np.sqrt(var.ravel()), alpha=0.2, color="blue", label="$2\\sigma$")
        #ax[3].scatter(self.validation_set[2], self.validation_set[3],
        #              label="validation set", color="orange", marker="x", s=15)
        ax[3].plot(self.X_f, self.f_train, 'r.',
                   markersize=10, label='training points')
        ax[3].plot(self.raw_data[0], self.raw_data[2],"--", label="Analytical solution",color = "black")
        ax[3].set_xlabel("t", fontsize=font_size)
        ax[3].set_ylabel("f(t)", fontsize=font_size)
        ax[3].grid(alpha=0.7)
        ax[3].legend()
        ax[3].set_title("(d) ", fontsize=font_size)
        plt.tight_layout()
        print("---------GPY--------")
        print("L_sq u: ", L_sq_u)
        print("L_sq f: ", L_sq_f)
        self.GPy_models = [model_GPy, model_GPy2]
        if path != None:
            plt.savefig(path, bbox_inches='tight', dpi=300)

    def plot_merged_2d_plots(self, X_star, path, figsize=(15, 15)):
        x_star, t_star = X_star[:,0].reshape(-1, 1), X_star[:, 1].reshape(-1, 1)
        original_size = (int(np.sqrt(len(x_star))),
                            int(np.sqrt(len(x_star))))
        font_size = 12.5
        fig = plt.figure(figsize=figsize)
        ax = [fig.add_subplot(3,2,1,projection="3d"), fig.add_subplot(3,2,2,projection="3d"), 
              fig.add_subplot(3,2,3), fig.add_subplot(3,2,4), 
              fig.add_subplot(3,2,5), fig.add_subplot(3,2,6)]
        #mean u 3d
        ax[0].plot_surface(x_star.reshape(original_size), t_star.reshape(original_size), self.mean_u.reshape(original_size), cmap='viridis', alpha=0.8, edgecolor='none')
        ax[0].set_title('(a) $\\mu_u$', fontsize=font_size)
        ax[0].set_xlabel(self.xlabel, fontsize=font_size)
        ax[0].set_ylabel(self.ylabel, fontsize=font_size)
        ax[0].scatter(self.X_u[:, 0], self.X_u[:, 1], self.u_train, c='r', marker='o',label= 'Training points')
        ax[0].set_zlabel("u(t,x)", fontsize=font_size)
        ax[0].legend()
        #mean u 3d
        ax[1].plot_surface(x_star.reshape(original_size), t_star.reshape(original_size), self.mean_f.reshape(original_size), cmap="viridis", alpha=0.8)
        ax[1].set_title('(b) $\\mu_f$', fontsize=font_size)
        ax[1].set_xlabel(self.xlabel, fontsize=font_size)
        ax[1].set_ylabel(self.ylabel, fontsize=font_size)
        ax[1].scatter(self.X_f[:, 0], self.X_f[:, 1], self.f_train, c='r',label='Training points', marker='o')
        ax[1].set_zlabel("f(t,x)", fontsize=font_size)
        ax[1].legend()

        #variance 2d set to 0 if negative(only valid if negative is very small)
        var_u = np.array(self.var_u.copy())
        var_u[var_u < 0] = 0
        var_u = var_u.reshape(original_size)
        cont = ax[2].imshow(np.sqrt(var_u), cmap='viridis', alpha=1, extent=[0, 1, 0, 1], origin='lower')
        ax[2].set_title('(c) $\\sigma_u$', fontsize=font_size)
        ax[2].set_xlabel(self.xlabel, fontsize=font_size)
        ax[2].set_ylabel(self.ylabel, fontsize=font_size)
        
        ax[2].scatter(self.X_u[:, 0], self.X_u[:, 1], c='r', marker='o')
        fig.colorbar(cont, ax=ax[2])

        #variance 2d set to 0 if negative(only if negative is very small)
        var_f = np.array(self.var_f.copy())
        var_f[var_f < 0] = 0
        var_f = var_f.reshape(original_size)

        cont2 = ax[3].imshow(np.sqrt(var_f), cmap='viridis', alpha=1, extent=[0, 1, 0, 1], origin='lower')
        ax[3].set_title('(d) $\\sigma_f$', fontsize=font_size)
        ax[3].set_xlabel(self.xlabel, fontsize=font_size)
        ax[3].set_ylabel(self.ylabel, fontsize=font_size)
        #ax[3].tick_params(axis='both', which='major', labelsize=font_size)
        ax[3].scatter(self.X_f[:, 0], self.X_f[:, 1], c='r', marker='o')
        fig.colorbar(cont2, ax=ax[3])
        #difference u
        data = self.raw_data
        x_star, t_star = self.raw_data[0].reshape(
            -1, 1), self.raw_data[1].reshape(-1, 1)
        u_grid = self.raw_data[2]
        f_grid = self.raw_data[3]
        size = (int(np.sqrt(len(x_star))), int(np.sqrt(len(x_star))))
        mean_u, var = self.predict_u(np.hstack([x_star, t_star]))
        mean_f, var = self.predict_f(np.hstack([x_star, t_star]))
        mean_u, mean_f = mean_u.reshape(size), mean_f.reshape(size)

        diff_u = np.abs(u_grid - mean_u)
        diff_f = np.abs(f_grid - mean_f)
        cont3 = ax[4].imshow(diff_u, cmap='viridis', alpha=1, extent=[0, 1, 0, 1], origin='lower')
        ax[4].set_title('(e) $|u_{true} - \\mu_u|$', fontsize=font_size)
        ax[4].set_xlabel(self.xlabel, fontsize=font_size)
        ax[4].set_ylabel(self.ylabel, fontsize=font_size)
        ax[4].scatter(self.X_u[:, 0], self.X_u[:, 1], c='r', marker='o')
        fig.colorbar(cont3, ax=ax[4])
        #difference f
        cont4 = ax[5].imshow(diff_f, cmap='viridis', alpha=1, extent=[0, 1, 0, 1], origin='lower')
        ax[5].set_title('(f) $|f_{true} - \\mu_f|$', fontsize=font_size)
        ax[5].set_xlabel(self.xlabel, fontsize=font_size)
        ax[5].set_ylabel(self.ylabel, fontsize=font_size)
        ax[5].scatter(self.X_f[:, 0], self.X_f[:, 1], c='r', marker='o')
        fig.colorbar(cont4, ax=ax[5])
        plt.tight_layout()
        if path != None:
            plt.savefig(path, bbox_inches='tight', dpi=300)

    def plot_GPYmeandiff(self,X_star,savepath, figsize=(10,10)):
        mean_u, var = self.GPy_models[0].predict(X_star)
        mean_f, var = self.GPy_models[1].predict(X_star)
        mean_u, mean_f = mean_u.reshape(-1,1), mean_f.reshape(-1,1)
        original_size = (int(np.sqrt(len(X_star))),
                            int(np.sqrt(len(X_star))))
        font_size = 10

        fig = plt.figure(figsize=figsize)
        ax = [fig.add_subplot(2,2,1,projection="3d"), fig.add_subplot(2,2,2,projection="3d"),
                fig.add_subplot(2,2,3), fig.add_subplot(2,2,4)]
        #mean u 3d
        ax[0].plot_surface(X_star[:,0].reshape(original_size), X_star[:,1].reshape(original_size), mean_u.reshape(original_size), cmap='viridis', alpha=0.8, edgecolor='none')
        ax[0].set_title('(a) $\\mu_u$')
        ax[0].set_xlabel(self.xlabel, fontsize=font_size)
        ax[0].set_ylabel(self.ylabel, fontsize=font_size)
        ax[0].scatter(self.X_u[:, 0], self.X_u[:, 1], self.u_train, c='r', marker='o')
        ax[0].set_zlabel("u(t,x)", fontsize=font_size)
        #mean f 3d
        ax[1].plot_surface(X_star[:,0].reshape(original_size), X_star[:,1].reshape(original_size), mean_f.reshape(original_size), cmap="viridis", alpha=0.8)
        ax[1].set_title('(b) $\\mu_f$')
        ax[1].set_xlabel(self.xlabel, fontsize=font_size)
        ax[1].set_ylabel(self.ylabel, fontsize=font_size)
        ax[1].scatter(self.X_f[:, 0], self.X_f[:, 1], self.f_train, c='r', marker='o')
        ax[1].set_zlabel("f(t,x)", fontsize=font_size)
        #difference u
        data = self.raw_data
        x_star, t_star = self.raw_data[0].reshape(-1, 1), self.raw_data[1].reshape(-1, 1)
        u_grid = self.raw_data[2]
        f_grid = self.raw_data[3]
        size = (int(np.sqrt(len(x_star))), int(np.sqrt(len(x_star))))
        mean_u, var = self.GPy_models[0].predict(np.hstack([x_star, t_star]))
        mean_f, var = self.GPy_models[1].predict(np.hstack([x_star, t_star]))
        mean_u, mean_f = mean_u.reshape(size), mean_f.reshape(size)

        diff_u = np.abs(u_grid - mean_u)
        diff_f = np.abs(f_grid - mean_f)
        cont3 = ax[2].imshow(diff_u, cmap='viridis', alpha=1, extent=[0, 1, 0, 1], origin='lower')
        ax[2].set_title('(c) $|u_{true} - \\mu_u|$')
        ax[2].set_xlabel(self.xlabel, fontsize=font_size)
        ax[2].set_ylabel(self.ylabel, fontsize=font_size)
        ax[2].scatter(self.X_u[:, 0], self.X_u[:, 1], c='r', marker='o')
        fig.colorbar(cont3, ax=ax[2])
        #difference f
        cont4 = ax[3].imshow(diff_f, cmap='viridis', alpha=1, extent=[0, 1, 0, 1], origin='lower')
        ax[3].set_title('(d) $|f_{true} - \\mu_f|$')
        ax[3].set_xlabel(self.xlabel, fontsize=font_size)
        ax[3].set_ylabel(self.ylabel, fontsize=font_size)
        ax[3].scatter(self.X_f[:, 0], self.X_f[:, 1], c='r', marker='o')
        fig.colorbar(cont4, ax=ax[3])
        plt.tight_layout()
        if savepath != None:
            plt.savefig(savepath, bbox_inches='tight', dpi=300)
        
    def plot_1d_cut(self, axis: list,index: list, savepath = None):
        "cut along an axis of the 2d plot"
        titles = ["(a)", "(b)","(c)","(d)"]
        axis_titles = [self.xlabel,self.ylabel]
        x_star, t_star = self.raw_data[0].reshape(
            -1, 1), self.raw_data[1].reshape(-1, 1)
        u_grid = self.raw_data[2]
        f_grid = self.raw_data[3]
        size = (int(np.sqrt(len(x_star))), int(np.sqrt(len(x_star))))
        
        mean_u, var_u = self.predict_u(np.hstack([x_star, t_star]))
        mean_f, var_f = self.predict_f(np.hstack([x_star, t_star]))
        var_u, var_f = var_u.reshape(size), var_f.reshape(size)
        mean_u, mean_f = mean_u.reshape(size), mean_f.reshape(size)
        #gpy
        mean_u_gpy, var_u_gpy = self.GPy_models[0].predict(np.hstack([x_star, t_star]))
        mean_f_gpy, var_f_gpy = self.GPy_models[1].predict(np.hstack([x_star, t_star]))
        var_u_gpy, var_f_gpy = var_u_gpy.reshape(size), var_f_gpy.reshape(size)


        fontsize = 10
        fig, ax = plt.subplots(2, 2, figsize=(13, 8))
        
        ax = ax.flatten()
        for i in range(len(ax)):
            if axis[i] == 0:
                print(mean_u.shape)
                u_gt_cut = u_grid[index[i], :]
                f_gt_cut = f_grid[index[i], :]
                u_mean_cut = mean_u[index[i], :]
                u_var_cut = var_u[index[i], :]
                f_mean_cut = mean_f[index[i], :]
                f_var_cut = var_f[index[i], :]
                xlabel = axis_titles[0]
                ylabel = f"u({axis_titles[0]},{axis_titles[1]}={int(index[i]/100)})"
                x_axis = np.linspace(0, 1, 101)
                #gpy
                u_mean_cut_gpy = mean_u_gpy[index[i], :]
                u_var_cut_gpy = var_u_gpy[index[i], :]
                f_mean_cut_gpy = mean_f_gpy[index[i], :]
                f_var_cut_gpy = var_f_gpy[index[i], :]

                

            elif axis[i] == 1:
                u_gt_cut = u_grid[:, index[i]]
                f_gt_cut = f_grid[:, index[i]]
                u_mean_cut = mean_u[:, index[i]]
                u_var_cut = var_u[:, index[i]]
                f_mean_cut = mean_f[:, index[i]]
                f_var_cut = var_f[:, index[i]]
                #gpy
                u_mean_cut_gpy = mean_u_gpy[:, index[i]]
                u_var_cut_gpy = var_u_gpy[:, index[i]]
                f_mean_cut_gpy = mean_f_gpy[:, index[i]]
                f_var_cut_gpy = var_f_gpy[:, index[i]]



                xlabel = axis_titles[1]
                ylabel = f"u({axis_titles[0]}={int(index[i]/100)},{axis_titles[1]})"      
                x_axis = np.linspace(0, 1, 101)
            ax[i].plot(x_axis, u_gt_cut, label="Ground truth", color="black")
            ax[i].plot(x_axis, u_mean_cut, label="Mean", color="blue")
            ax[i].fill_between(x_axis, u_mean_cut - 2*np.sqrt(u_var_cut), 
                               u_mean_cut + 2*np.sqrt(u_var_cut), color="blue", alpha=0.2, label="$2\\sigma$")
            
            ax[i].plot(x_axis, u_mean_cut_gpy, label="Mean GPy", color="red")
            ax[i].fill_between(x_axis, u_mean_cut_gpy - 2*np.sqrt(u_var_cut_gpy), 
                               u_mean_cut_gpy + 2*np.sqrt(u_var_cut_gpy), color="red", alpha=0.2, label="$2\\sigma$ GPy")
            ax[i].set_xlabel(xlabel,fontsize=fontsize)
            ax[i].set_ylabel(ylabel,fontsize=fontsize)
            ax[i].legend()
            ax[i].grid(alpha=0.7)
            ax[i].set_title(titles[i],fontsize=fontsize)
            plt.tight_layout()
            if savepath is not None:
                plt.savefig(savepath,bbox_inches = "tight",dpi = 300)

    def plot_pictures_3d(self,numberofpics,figsize = (20,5)):
        fig, ax = plt.subplots(2,10,figsize=(20,5))
        ax = ax.flatten()

        for i in range(20):
            ax[i].imshow(self.raw_data[2][:,:,i*5], cmap='viridis',extent=[0,2,0,2])
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('y')
            ax[i].set_title('t = ' + str(i*5))
        fig, ax = plt.subplots(2,10,figsize=(20,5))

        ax = ax.flatten()
        for i in range(20):
            ax[i].imshow(self.raw_data[3][:,:,i*5], cmap='viridis',extent=[0,2,0,2])
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('y')
            ax[i].set_title('t = ' + str(i*5))

