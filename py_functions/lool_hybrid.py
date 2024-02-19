#Libraries to apply in the functions
import numpy as np
import pandas as pd
from MuyGPyS._test.sampler import UnivariateSampler2D, print_results
from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.deformation import Isotropy, l2
from MuyGPyS.gp.hyperparameter import AnalyticScale, Parameter, DownSampleScale
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.gp.tensors import  make_predict_tensors, make_train_tensors
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize import Bayes_optimize
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.loss import lool_fn, looph_fn, mse_fn
from scipy.stats import norm

#Initial function to compute the CRPS statistic
def crps(pred, sd, trueobs):
    z = np.abs((trueobs - pred) / sd)
    
    scores = sd * (z * (2 * norm.cdf(z, 0, 1) - 1) + 2 * norm.pdf(z, 0, 1) - 1 / np.sqrt(np.pi))
    
    return np.mean(scores)

#Function to estimate and predict hyperparameters using lool loss 
def optim_hybrid (X_train, y_train, X_test, y_test, nn_count, 
    batch_count, train_count, n_sims):
    """
    Args:
            X_train: features obtained from train data
            y_train: target obtained from train data
	    X_test: features obtained from test data
	    y_test: target obtained from test data
	    nn_count: total number of nearest neighbors
	    batch_count: number to sample from when selecting a batch of hyperparameters
	    train_count: total count of the train data
	    max_iter: maximum number of iteration
            n_sims: number of simulations
    """
    results_list = []

    for j in range(n_sims):
      #Constructing Nearest Neighbor Lookup
      nbrs_lookup = NN_Wrapper(X_train, 
                                 nn_count, 
                                 nn_method = "exact",
                                 algorithm = "ball_tree")

      #Creating a MuyGPS Object
      exp_nu = Parameter("log_sample", (0.1, 5.0))
      muygps = MuyGPS(
                kernel=Matern(
                smoothness=exp_nu,
                deformation = Isotropy(
                l2,
                length_scale=Parameter(1.0),
                ),
            ),
                noise=HomoscedasticNoise(1e-7),
                scale = DownSampleScale(),
              )
      #Collecting batches
      batch_indices, batch_nn_indices = sample_batch( nbrs_lookup, int(batch_count), int(train_count))
      
      # Convert lists to NumPy arrays
      batch_indices = np.array(batch_indices)
      batch_nn_indices = np.array(batch_nn_indices)
      #Computing all pairwise differences betweens points and their nearest neighbors
      (crosswise_dists, pairwise_dists, batch_response, batch_nn_response) = make_train_tensors(
                                                                   batch_indices, 
                                                                   batch_nn_indices, 
                                                                   X_train, 
                                                                   y_train,)

      #Optimizing the parameter
      muygps_bayes = Bayes_optimize( muygps,
                                          batch_response, 
                                          batch_nn_response, 
                                          crosswise_dists, 
                                          pairwise_dists, 
                                          loss_fn = lool_fn,
                                          init_points = 5,
                                          allow_duplicate_points=True, 
                                          random_state = 1,
                                          verbose = False,
                                          n_iter = 15)

      muygps_bayes = muygps_bayes.optimize_scale(pairwise_dists, batch_nn_response)
          
      #Finding the indices of the nearest neighbors of the test set
      test_count, _ = X_test.shape
      indices = np.arange(test_count)
      test_nn_indices, _ = nbrs_lookup.get_nns(X_test)

      #Computing the difference and target tensors for the test sample
      (test_crosswise_diffs, test_pairwise_diffs, test_nn_response) = make_predict_tensors(
                                                            indices,
                                                            test_nn_indices,
                                                            X_test,
                                                            X_train,
                                                            y_train)

      #Creating the kernel tensors
      Kcross = muygps_bayes.kernel(test_crosswise_diffs)
      K = muygps_bayes.kernel(test_pairwise_diffs)

      # Computing posterior means and variances associated with training prediction
      mean_pred = muygps_bayes.posterior_mean(K, Kcross, test_nn_response)
      bayes_var = muygps_bayes.posterior_variance(K, Kcross)  

      #Prediction Evaluation
      CI = np.sqrt(bayes_var) * 1.96
      CI_coverage = (np.count_nonzero(np.abs(y_test - mean_pred) < CI) / test_count)
      crps_val = crps(pred=mean_pred, sd= np.sqrt(bayes_var),trueobs = y_test)    
      
      results_list.append({"RMSE": np.sqrt(mse_fn(mean_pred, y_test)),
                           "CRPS": crps_val,
                           "MAD": np.median(np.abs(mean_pred - y_test)),
                           "median diagonal variance": np.median(bayes_var),
                           "median confidence interval size": np.median(CI * 2),
                           "coverage": CI_coverage})

    # Convert the list of dictionaries to a pandas DataFrame
    results_df = pd.DataFrame(results_list)
    
    return results_df