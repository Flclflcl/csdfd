__credits__ = 'Olalekan Ogunmolu, Rodrigo Perez-Dattari (TU Delft), Rachel Thomson (MIT), Jethro Tan (PFN)'
__license__ = 'MIT'

import numpy as np

general_params = {
    'shape_name': 'AGV_interp_5_00_small_rl'
}




lyapunov_learner_params = {
    'L': 7,
    'd': 2,
    'w': 1e-3,      # 优化目标权重，对V_dot的权重
    'Mu': np.array(()),
    'P': np.array(()),
    'tol_mat_bias': 1e-1,
    'int_lyap_random': False,
    'optimizePriors': True,
    'upperBoundEigenValue': True,
    'int_lyap_re': False
}

gmm_params = {
    'num_clusters': 6,
    'max_iterations': 500
}

event_trigger_params = {
    'retrain': False
}
