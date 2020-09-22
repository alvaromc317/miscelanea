import numpy as np
import time
import logging
import sys
from tabulate import tabulate
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# CREATE LOGGER #######################################################################################################
def configure_logging(logging_level='INFO', log_to_file=False, destination_folder=''):
    """
    :param logging_level: Logging level
    :param log_to_file: Boolean. Should log be stored into a file? If yes, provide destination_folder
    :param destination_folder: If log should be stored in file, destination folder of where to save it
    :return: A logger to be used in other functions or code
    """
    # Define format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
    # Create logger
    logger = logging.getLogger()
    # Define level
    logging_level = logging.getLevelName(logging_level)
    logger.setLevel(logging_level)
    # Create stream or file logging
    if log_to_file is False:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(filename=destination_folder + 'logging_' + '.log')
    handler.setLevel(logging_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# COMPARE ARRAYS ######################################################################################################
def equal_array(array0, array1, consider_sign=True, tol=1e-5, return_columns=False):
    """
    :param array0: A uni dimensional or multidimensional array
    :param array1: A uni dimensional or multidimensional array
    :param consider_sign: If true, it will check for matching signs. Especially useful in PCA where
                          opposed signs provide same result
    :param tol: Tolerance to 0
    :param return_columns: Boolean. If a discrepancy between the two arrays is found, should the function provide the
                           conflictive columns?
    :return: If arrays are equal, there is NO RETURN
    """
    # Check dimensions
    n0 = array0.shape
    n1 = array1.shape
    if n0 != n1:
        logging.error(f'Shapes do not match. Received: {n0} and {n1}')
    else:
        # Reshape as matrix
        if array1.ndim == 1:
            array0 = array0.reshape((1, array0.shape[0]))
            array1 = array1.reshape((1, array1.shape[0]))
        if consider_sign:
            sign = np.sign(array0[0, :]) * np.sign(array1[0, :])
            array1 = sign * array1
        dif = array0 - array1
        abs_mean_dif = np.abs(np.mean(dif))
        # Check by column
        col_dif = np.abs(np.mean(dif, axis=0))
        bool_dif = col_dif > tol
        if any(bool_dif):
            logging.warning(f'Arrays are not equal. Mean difference is {abs_mean_dif}.')
            problematic_cols = np.where(bool_dif)[0].tolist()
            if return_columns:
                return problematic_cols


# BENCHMARK FUNCTION ##################################################################################################
def benchmark(func, params, niters=40, warm_start=10, max_time=600):
    """
    :param func: A function
    :param params: The parameters of the function stored as a dictionary
    :param niters: Number of repetitions of the benchmark
    :param warm_start: How many iterations to "burn" in order to heat the computation (exclude from final results)
    :param max_time: max time in seconds that the benchmark will take. Minimum time is one iteration
    :return: Dictionary containing n_iters, execution_time (as an array), mean and sd
    """
    execution_time = []
    i = 0
    for i in range(niters):
        in_loop_start_time = time.time()
        func(**params)
        execution_time.append(time.time() - in_loop_start_time)
        total_time = np.sum(execution_time)
        if total_time > max_time:
            logging.warning(f'Reached max_time: {max_time}. Exiting benchmark loop')
            break
    if i > (warm_start - 1):
        mean = np.mean(execution_time[(warm_start - 1):])
        std = np.std(execution_time[(warm_start - 1):])
    else:
        logging.warning(f'Not enough iterations for warm_start. Computing mean and std of {i+1} iterations')
        mean = np.mean(execution_time)
        std = np.std(execution_time)
    results = dict(
        n_iters=i+1,
        execution_time=np.asarray(execution_time),
        mean=np.round(mean, 4),
        std=np.round(std, 4)
    )
    # Create output table
    header = ['n_iters', 'mean', 'std']
    data = [results['n_iters'], results['mean'], results['std']]
    print(tabulate(data, headers=header))
    return results


# RENAME FILES IN FOLDER ##############################################################################################
def rename_files_in_folder(folder, base_name, extension):
    """
    :param folder: A path to a folder
    :param base_name: The base name to be used
    :param extension: The extension of the files that should be renamed
    :return: No return.
    """
    file_list = os.listdir(folder)
    file_list = [elt for elt in file_list if elt.endswith(extension)]  # Remove files with an extension different
    for idx, file in enumerate(file_list):
        os.rename(folder + file, folder + base_name + '_' + str(idx) + extension)


# GIVEN BETA AS A PAIR INDEX - VALUE, REBUILD THE ARRAY
def rebuild_sparse_beta(n_beta, index, value):
    beta = np.repeat(0.0, n_beta)
    beta[index] = value
    return beta
