from __future__ import print_function
import numpy as np
import warnings
from sklearn.externals import joblib
from data_fn import Data
from alpha_fn import Alpha
from fit_fn import Fit
from simulation_fn import Simulation
from evaluation_fn import Evaluation
np.random.seed(123)
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    """
    Description:
        1. Chosen strategy works with daily data and rebalance portfolio every day at market open.
        2. Model trains on historical data and returns predictions of every day close price.
        3. Core model consists of two neural networks working with different sets of targets 
           aimed to predict single value and sequences.
        4. Final prediction is a linear combination of these two predictions.
    
    Assumptions:
        1. The configuration for the model was found after a little trial and error and is by 
           no means optimized! Calibrating the model parameters is a long journey. Anyway current
           set of parameters does not 'suffer' a lot from high bias / high variance.
    
    Config
        1. options.ini file with all necessary parameters stored in 'data/' folder.
        2. copy of config file in 'data/config_copy/'.
    
    """

    # WARNING!
    # It takes a lot of time.

    # Prepare data.
    # Data().prepare_data()

    # Create alphas.
    # Alpha().run_alphas()

    # Fit the model.
    # positions_strategy = Fit().run_model()

    # Load saved positions matrix.
    positions_strategy = joblib.load('data/pp_data/positions/positions.pickle')
    #
    # # Run simulation
    strategy = Simulation(positions_strategy).run_simulation()
    #
    # # Show results
    Evaluation(strategy).run_evaluation(save_report=False)
