#Libraries to apply in the functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MuyGPyS.gp.deformation import Isotropy, l2
from MuyGPyS.gp.hyperparameter import Parameter, AnalyticScale, DownSampleScale
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.tensors import make_train_tensors
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize import Bayes_optimize
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.gp.kernels import RBF
from MuyGPyS.gp.tensors import make_predict_tensors
from MuyGPyS._test.sampler import UnivariateSampler2D, print_results
from MuyGPyS.optimize.loss import lool_fn, looph_fn, mse_fn
from scipy.stats import norm

#Initial function to compute the CRPS statistic
def crps(pred, sd, trueobs):
    z = np.abs((trueobs - pred) / sd)
    
    scores = sd * (z * (2 * norm.cdf(z, 0, 1) - 1) + 2 * norm.pdf(z, 0, 1) - 1 / np.sqrt(np.pi))
    
    return np.mean(scores)

#Function to conduct the down-sampling approach
def robust_optim(
    X_train,
    y_train,
    X_test,
    y_test,
    nn_count,
    batch_count,
    train_count,
    down_count,
    iteration_count=5,
    test_iteration_count=5,
):
    """
    Args:
            X_train: features obtained from train data
            y_train: target obtained from train data
            X_test: features obtained from test data
            y_test: target obtained from test data
            nn_count: total number of nearest neighbors
            batch_count: number to sample from when selecting a batch of hyperparameters
            train_count: total count of the train data

    """
    if down_count > nn_count:
        raise ValueError(
            f"downsampled size {down_count} must be smaller than nn count {nn_count}"
        )

    central_values = []  # To store central values for each subsample
    nus = []

    # Constructing Nearest Neighbor Lookup
    nbrs_lookup = NN_Wrapper(
        X_train, int(nn_count), nn_method="exact", algorithm="ball_tree"
    )

    # Creating a MuyGPS Object
    exp_nu = Parameter("log_sample", (0.1, 5.0))
    muygps = MuyGPS(
        kernel=Matern(
            smoothness=exp_nu,
            deformation=Isotropy(
                l2,
                length_scale=Parameter(1.0),
            ),
        ),
        noise=HomoscedasticNoise(1e-5),
        scale = DownSampleScale(),
    )
    # Collecting batches
    batch_indices, batch_nn_indices = sample_batch(
        nbrs_lookup, int(batch_count), int(train_count)
    )
    for i in range(iteration_count):
        # Excluding neighbors
        sampled_indices = np.random.choice(
            np.arange(nn_count),
            size=down_count,
            replace=False,
        )
        downsampled_nn_indices = batch_nn_indices[:, sampled_indices]

        # must guarantee that
        #   downsampled_nn_indices.shape = (batch_count, down_count)

        (
            crosswise_diffs,
            pairwise_diffs,
            batch_response,
            batch_nn_response,
        ) = make_train_tensors(batch_indices, downsampled_nn_indices, X_train, y_train)

        central_value = np.median(np.unique(batch_nn_response))
        central_values.append(central_value)

        # whatever we do, we must guarantee that:
        #   b == batch_count
        #   k == down_count
        #   f == feature_count
        #   r == response_count
        #
        #   pairwise_diffs.shape    = (b, k, k, f)
        #   crosswise_diffs.shape   = (b, k, f)
        #   batch_nn_response.shape = (b, k, r)

        muygps_bayes = Bayes_optimize(
            muygps,
            batch_response,
            batch_nn_response,
            crosswise_diffs,
            pairwise_diffs,
            loss_fn=looph_fn,
            allow_duplicate_points=True,
            random_state = 1,
            verbose=False,
            n_iter=int(15),
            init_points = 5
        )
        nu = muygps_bayes.kernel.smoothness()
        nus.append(nu)

        muygps_bayes = muygps_bayes.optimize_scale(
            pairwise_diffs,
            batch_nn_response
        )
 
        print(
            f"iteration {i} finds central value = {central_value}, nu = {nu}"
        )

    # Calculate the median of central values
    median_central_value = np.median(central_values)
    print(f"Median central value: {median_central_value}")

    # Robust smoothness parameter
    robust_nu = np.median(nus)
    print(f"median nu: {robust_nu}")

    robust_muygps = MuyGPS(
        kernel=Matern(
            smoothness=Parameter(robust_nu),
            deformation=Isotropy(
                l2,
                length_scale=Parameter(1.0),
            ),
        ),
        noise=HomoscedasticNoise(1e-7),
        scale = DownSampleScale(),
    )

    # setting sigma_sq for our new MuyGPs object
    (
        crosswise_diffs,
        pairwise_diffs,
        batch_response,
        batch_nn_response,
    ) = make_train_tensors(batch_indices, batch_nn_indices, X_train, y_train)
    robust_muygps = robust_muygps.optimize_scale(
        pairwise_diffs,
        batch_nn_response
    )

    # Finding the indices of the nearest neighbors of the test set
    test_count, _ = X_test.shape
    indices = np.arange(test_count)
    test_nn_indices, _ = nbrs_lookup.get_nns(X_test)

    means = np.zeros((test_count, test_iteration_count))
    variances = np.zeros((test_count, test_iteration_count))

    ### NEW STUFF STARTS HERE
    for i in range(test_iteration_count):
        # Excluding neighbors
        sampled_indices = np.random.choice(
            np.arange(nn_count),
            size=down_count,
            replace=False,
        )
        downsampled_nn_indices = test_nn_indices[:, sampled_indices]

        # Computing the difference and target tensors for the test sample
        (
            test_crosswise_diffs,
            test_pairwise_diffs,
            test_nn_response,
        ) = make_predict_tensors(
            indices,
            downsampled_nn_indices,
            X_test,
            X_train,
            y_train,
        )

        # Creating the kernel tensors
        Kcross = robust_muygps.kernel(test_crosswise_diffs)
        K = robust_muygps.kernel(test_pairwise_diffs)

        # Computing posterior means and variances associated with training prediction
        this_pred = robust_muygps.posterior_mean(K, Kcross, test_nn_response)
        this_var = robust_muygps.posterior_variance(K, Kcross)
        means[:, i, None] = this_pred.reshape(test_count, 1)
        variances[:, i, None] = this_var.reshape(test_count, 1)

    mean_pred = np.median(means, axis=1).reshape(test_count, 1)
    bayes_var = np.median(variances, axis=1).reshape(test_count, 1)

    # Prediction Evaluation
    CI = np.sqrt(bayes_var) * 1.96
    CI_coverage = np.count_nonzero(np.abs(y_test - mean_pred) < CI) / test_count
    crps_val = crps(pred=mean_pred, sd= np.sqrt(bayes_var),trueobs = y_test) 

    results_list = {"RMSE": [np.sqrt(mse_fn(mean_pred, y_test))],
                         "CRPS": [crps_val],
                           "MAD": [np.median(np.abs(mean_pred - y_test))],
                           "median diagonal variance": [np.median(bayes_var)],
                           "median confidence interval size": [np.median(CI * 2)],
                           "coverage": [CI_coverage]}

    # Convert the list of dictionaries to a pandas DataFrame
    results_df = pd.DataFrame(results_list)
    
    return results_df
