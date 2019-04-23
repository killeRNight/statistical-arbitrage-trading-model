from __future__ import print_function
import pandas as pd
import numpy as np
import os
import time
import h5py
from sklearn.externals import joblib
from sklearn.linear_model import BayesianRidge
from keras.layers import Input, Dense, Activation, LeakyReLU, BatchNormalization, GRUCell, Flatten
from keras.layers import Dropout, Conv1D, Conv2D, Lambda, RNN, GlobalAveragePooling1D, MaxPooling1D, AveragePooling2D
from keras.utils.generic_utils import get_custom_objects
from keras.models import model_from_json
from keras.models import Model
from keras import regularizers
from catboost import CatBoostRegressor, Pool
from keras import layers
from keras import optimizers
from keras import backend as K
import configparser
import helper as hl
import tensorflow as tf
from tensorflow import set_random_seed
import warnings
np.random.seed(27)
set_random_seed(27)
pd.set_option('display.max_colwidth', 1500)
pd.set_option('display.notebook_repr_html', True)
pd.set_option('display.width', 900)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Fit():

    """
    Fit the model.
    : input: data object and alphas.
    : return: predicted positions matrix
    : store: save predictions to disk.

    Brief description:
        - Train the model on given set of alphas-labels and return predictions
          on each day.

    Prepare data:
        Feature Eng:
            - d-SAE model extracts high level deep features from given alpha matrix.
        Feature selection:
            - gBoosting based feature importance filter drops irrelevant features.
        Stacking (alpha):
            - add events to alpha matrix.
        Stacking (alpha):
            - bayesian ridge model predicts target values and stack predictions to
              alpha matrix.
        Stacking (target):
            - add two more days to target matrix (original).
        Stacking (target):
            - add total return (3 days) to target values (original).

    Model:
        ALPHA:
            - 2-rRNet w/ DeepConv features in pair with Seq2Seq DeepConv-GRU predict
             '3 day+total' and '3 day' returns.
        BETA:
            - 2-rRNet in pair with Feature Pyramid Network predict '1 day' return.

    Prediction:
        - Predictions are made using MC Dropout sampling and weighting sequence
          predictions together with single day forecasts.

    Details:
        First training (general):
            - general + fine tuning of all layers.
        Transfer Learning (if activated):
            - transfer learning w/ newly initialized last layers + fine tuning of
              all layers + freezing.
            - model freeze selected layers for a certain period of time specified in
              options.ini, after a given period layers become trainable again.

    Assumption:
        The configuration for the model was found after a little trial and error
        and is by no means optimized! Calibrating the model parameters is a long journey.
        Anyway current set of parameters does not 'suffer' a lot from high bias / high variance.

    """

    def __init__(self):

        # Path to options file.
        self.path_options = 'options.ini'

        # Load config file.
        config = configparser.ConfigParser()                                             # define config object.
        config.optionxform = str                                                         # hold keys register
        config.read(self.path_options)                                                   # read config file.

        # Path variables.
        self.path_data = config['PATH'].get('DataObj')                                   # address: data object
        self.path_alphas = config['PATH'].get('Alphas')                                  # address: alphas
        self.path_events = config['PATH'].get('Events')                                  # address: events
        self.path_tensors = config['PATH'].get('Tensors')                                # address: tensors
        self.path_positions = config['PATH'].get('PositionsObj')                         # address: predictions

        # Load data object
        self.data = joblib.load(self.path_data)                                          # data object

        # Internal variables.
        self.alpha_storage = None                                                        # alpha model storage
        self.beta_storage = None                                                         # beta model storage
        self.selected_alphas = None                                                      # selected alphas
        self.start_transfer_learning = False                                             # start tf mark
        self.freeze_control = 0                                                          # layer freeze control

    # Fn: (1)
    # Model is trained on each refit day and
    # then predictions are made.
    def run_model(self):

        # Load config file.
        config = configparser.ConfigParser()                                             # define config object.
        config.optionxform = str                                                         # hold keys register
        config.read(self.path_options)                                                   # read config file.

        # Create positions folder is it doesn`t exist.
        if not os.path.exists(self.path_positions.replace('positions.pickle', '')):
            os.makedirs(self.path_positions.replace('positions.pickle', ''))

        print('')
        print('Fit model:')
        """        
        ----------------------------------------------------------------------------------------------------------                                                     
                                                         DATA
                                        Load all necessary data variables and 
                                                    model parameters.                                                                                                          
        ----------------------------------------------------------------------------------------------------------
        """
        # ---------------------------
        #            DATA
        # ----------------------------
        # Dates. arr.
        dates = self.data['dates']
        # Industry data. arr.
        industry = self.data['industry.sector']
        # Selected alphas. arr.
        alphas_list = os.listdir(self.path_alphas)
        events_list = os.listdir(self.path_events)
        # Data holders sizes. val.
        numstocks = self.data['numstocks']
        numdates = self.data['numdates']
        print('Days available: {}'.format(len(dates)))

        # -------------------------------
        #          PARAMETERS
        # -------------------------------

        # General
        lookback = config['DEFAULT'].getint('Lookback')                                 # days for training
        fit_startdate = config['DEFAULT'].getint('FitStartDate')                        # {20100104: 20180205}
        delay = config['DEFAULT'].getint('Delay')                                       # delay days
        refit_freq = config['DEFAULT'].getint('RefitFreq')                              # refit frequency

        # Target value
        ndays = config['FIT'].getint('TargetNDays')                                     # return mean value
        ret_cap = config['FIT'].getfloat('TargetRetCap')                                # return cap.
        min_target = config['FIT'].getfloat('TargetMinValOr')                           # target values min cap
        min_target_z = config['FIT'].getfloat('TargetMinValZc')                         # target values max cap
        industry_balancing = config['FIT'].getboolean('TargetIndBalancing')             # perform industry balancing

        # Alpha adjustment
        min_alpha = config['FIT'].getfloat('AlphaMin')                                  # alpha values min cap
        max_alpha = config['FIT'].getfloat('AlphaMax')                                  # alpha values max cap

        # Model
        loss_adj = config['FIT'].getfloat('ModelLossAdj')                               # stock loss fn. adj
        reg_ffn = config['FIT'].getfloat('ModelRegFFN')                                 # reg. ffn
        reg_conv = config['FIT'].getfloat('ModelRegConvNet')                            # reg. con
        reg_gru = config['FIT'].getfloat('ModelRegGRU')                                 # reg. rnn
        layer_capacity = config['FIT'].getint('ModelLayerCapacity')                     # neurons n. resnet
        n_layer_1 = config['FIT'].getint('ModelNeuronsLayer1')                          # neurons n. 1-st layer
        n_layer_2 = config['FIT'].getint('ModelNeuronsLayer2')                          # neurons n. 2-nd layer
        deep_features_n = config['FIT'].getint('ModelDeepFeaturesNumber')               # deep features number
        ch_layer_1 = config['FIT'].getint('ModelChannelLayer1')                         # channels n. 1-st layer
        ch_layer_2 = config['FIT'].getint('ModelChannelLayer2')                         # channels n. 2-nd layer
        ch_layer_3 = config['FIT'].getint('ModelChannelLayer3')                         # channels n. 3-rd layer
        ch_reduction = config['FIT'].getint('ModelChannelSpatialReduction')             # ch. n. spatial reduction
        kernel_conv1d = config['FIT'].getint('ModelKernelConv1D')                       # conv1d kernel size
        kernel_conv2d = tuple([np.int(i) for i in                                       # conv2d kernel size
                               config['FIT'].get('ModelKernelConv2D').split(',')])      # .......
        dropout_rate = config['FIT'].getfloat('ModelDropout')                           # dropout rate
        leaky_slope = config['FIT'].getfloat('ModelLeakySlope')                         # activation fn. slope
        learning_rate = config['FIT'].getfloat('ModelLearningRate')                     # learning rate
        epochs_n_general = config['FIT'].getint('ModelEpochsNumberGeneral')             # epochs n. general fit
        epochs_n_fn = config['FIT'].getint('ModelEpochsNumberFineTuning')               # epochs n. fine tuning
        batches_n = config['FIT'].getint('ModelBatchesNumber')                          # batches n.
        rnn_cells = [np.int(i) for i in                                                 # rnn cells
                     config['FIT'].get('ModelRNNLayers').split(',')]                    # .......
        mc_sampling_n = config['FIT'].getint('ModelMCDropoutSamplingNumber')            # sampling n.
        ret_sizing = config['FIT'].getfloat('ModelPredictionReturnSizing')              # total return check
        ret_adj = config['FIT'].getfloat('ModelPredictionReturnAdj')                    # adjust t+1 day return
        s2s_weight = config['FIT'].getfloat('ModelSeq2SeqWeight')                       # seq2seq model weight
        am_weight = config['FIT'].getfloat('ModelAlphaWeight')                          # alpha model weight

        # D-SAE
        dae_learning_rate = config['FIT'].getfloat('DAELearningRate')                   # autoencoder lr.
        dae_epochs = config['FIT'].getint('DAEEpochsNumber')                            # autoencoder epochs n.
        dae_reg = config['FIT'].getfloat('DAEReg')                                      # autoencoder reg. kernel
        dae_batch_n = config['FIT'].getint('DAEBatchesNumber')                          # autoencoder batches n.
        dae_layer_1 = config['FIT'].getint('DAENeuronsLayer1')                          # autoencoder neurons n. 1st
        dae_features = config['FIT'].getint('DAEFeaturesNumber')                        # autoencoder features

        # Tensor
        tensor_or_periods = config['TENSOR'].getint('TensorOrTimesteps')                # tensor or. timesteps
        tensor_sc_periods = config['TENSOR'].getint('TensorScTimesteps')                # tensor sc. timesteps
        tensor_sc_comps = config['TENSOR'].getint('TensorScComponents')                 # tensor sc. components

        # Other
        adv_period = config['FIT'].getint('ADVPeriod')                                  # adv mean for n. days
        scaler_original = config['FIT'].getint('ScalerOriginal')                        # scaler original data
        bayesian_ridge_iterations = config['FIT'].getint('BayesianRidgeIter')           # bayesian ridge iter.
        gradient_boosting_estimators = config['FIT'].getint('GBoostingEstimators')      # g. boosting estimators
        gradient_boosting_lr = config['FIT'].getfloat('GBoostingLr')                    # g. boosting lr.
        gradient_boosting_min_imp = config['FIT'].getint('GBoostingMinImportance')      # g. boosting min. imp.

        # Transfer learning
        activate_tf = config['FIT'].getboolean('ModelActivateTransferLearning')         # transfer learning activation
        epochs_n_tf = config['FIT'].getint('ModelEpochsNumberTransferLearning')         # epochs n. transfer lr.
        freeze_period = config['FIT'].getint('ModelLayersFreezePeriod')                 # layers freeze period
        alpha_freeze = ['dense_8', 'dense_11', 'conv1d_1']                              # freeze layers (conv1d_2?)
        beta_freeze = ['dense_17', 'dense_20', 'conv2d_1']                              # freeze layers (conv2d_2?)

        # Refit mask. arr.
        refit_dates_mask = hl.fn_refit_mask(dates, fit_startdate, refit_freq)
        # ---------------------------------
        fit_start_idx = np.where(self.data['dates'] > fit_startdate)[0]
        assert (fit_start_idx[0] - lookback - 10 > 0), 'Fit start date is not correct, please select later date.'
        assert (len(fit_start_idx) > 0), 'Fit start date is not correct, please select earlier date.'

        # -------------------------------
        #           VARIABLES
        # -------------------------------

        # TARGET: original returns matrix
        ret_mat = hl.handle_returns(self.data['close'], ndays, ret_cap,
                                    min_target, industry_balancing, industry)
        target_1 = hl.ts_delay(ret_mat, -ndays - delay)
        target_2 = hl.ts_delay(ret_mat, -ndays - delay - 1)
        target_3 = hl.ts_delay(ret_mat, -ndays - delay - 2)

        # TARGET: zscored returns matrix
        ret_mat_z = hl.handle_returns_z(self.data['close'], ndays, ret_cap,
                                        min_target_z, industry_balancing, industry)
        target_z = hl.ts_delay(ret_mat_z, -ndays - delay)

        # REGIME: average market return
        close = self.data['close']
        ret_average = (close - hl.ts_delay(close, ndays)) / hl.ts_delay(close, ndays)
        ret_average = np.nanmean(ret_average, axis=0)
        regime_mat = np.zeros((close.shape[0], close.shape[1]), dtype=np.float32)
        for i in range(regime_mat.shape[1]):
            regime_mat[:, i] = ret_average[i]
        regime_mat = hl.nan_to_zero(regime_mat)
        regime_mat = (regime_mat > 0) * 1.0 + (regime_mat < 0) * (-1.0)

        # LIQUIDITY: ranked average trading volume
        adv_mat = hl.ts_mean(self.data['close'] * self.data['volume'], adv_period)
        adv_mat = hl.cs_rank(hl.zero_to_nan(adv_mat))
        adv_mat = hl.nan_to_zero(adv_mat)
        adv_mat[adv_mat == 0] = 1

        # TENSOR (original): time series data holder.
        # available tensors (open, high, low, close)
        tensor_fields_original = ['open', 'high', 'low', 'close']
        tensor_data_original = np.empty((numstocks, numdates,
                                         tensor_or_periods, len(tensor_fields_original)))
        for ii, field in enumerate(tensor_fields_original):
            with h5py.File('data/pp_data/tensors/' + field + '_original.h5', 'r') as hf:
                tensor_data_original[:, :, :, ii] = hf[field + '_original'][:]

        # TENSOR (wavelet): time series data holder.
        # available tensors (open, high, low, close, adv)
        tensor_fields_wavelet = ['open', 'high', 'low', 'close']
        tensor_data_wavelet = np.empty((numstocks, numdates,
                                        tensor_sc_periods, tensor_sc_comps,
                                        len(tensor_fields_wavelet)))
        for jj, field in enumerate(tensor_fields_wavelet):
            with h5py.File('data/pp_data/tensors/' + field + '_wavelet.h5', 'r') as hf:
                tensor_data_wavelet[:, :, :, :, jj] = hf[field + '_wavelet'][:]

        # Delete config.
        del config

        # Data size
        data_size = (target_1.nbytes + target_2.nbytes + target_3.nbytes + target_z.nbytes
                     + adv_mat.nbytes + tensor_data_original.nbytes + tensor_data_wavelet.nbytes) / (1024 ** 3)
        print('Data size: {:.2f}Gb'.format(data_size))
        """

        FUNCTIONS
        Functions used to prepare data and train the model.

            1.  fe: De-noising stacked autoencoder.
            2.  fs: GBoosting based features importance filter.
            3.  fe: Stacking Bayesian regression prediction.
            4.  nn: Activation function (GeLU).
            5.  nn: Loss function.
            6.  nn: Loss function (+total return).
            7.  nn: Residual block.
            8.  nn: Squeeze-and-excitation network.
            9.  nn: Feature pyramid network block.
            10. nn: Spatial reduction block.
            11. ALPHA model.
            12. BETA model.

        """

        # Fn: (1)
        # Feature eng: de-noising stacked autoencoder.
        def fe_extract_features_dae(alpha_dae):
            # Opt.
            opt_dae = optimizers.RMSprop(lr=dae_learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
            # Add noise.
            dae_noise = (np.random.randint(0, 100, alpha_dae.shape[0]) / 500).reshape(-1, 1)
            target_dae = np.copy(alpha_dae)
            alpha_dae = np.add(alpha_dae, dae_noise)
            input_dae = Input(shape=(alpha_dae.shape[1],))
            # DAE architecture.
            enc = Dense(dae_layer_1, kernel_regularizer=regularizers.l2(dae_reg))(input_dae)
            enc = Activation('relu')(enc)
            enc_deep = Dense(dae_features, kernel_regularizer=regularizers.l2(dae_reg))(enc)
            enc_deep = Activation('relu')(enc_deep)
            dec_deep = Dense(dae_layer_1, kernel_regularizer=regularizers.l2(dae_reg))(enc_deep)
            dec_deep = Activation('relu')(dec_deep)
            dec = Dense(alpha_dae.shape[1], kernel_regularizer=regularizers.l2(dae_reg))(dec_deep)
            dec = Activation('linear')(dec)
            # DAE model.
            dae_model = Model(inputs=input_dae, outputs=dec)
            # Define encoder.
            encoder_model = Model(inputs=input_dae, outputs=enc_deep)
            # Compile and train.
            dae_model.compile(optimizer=opt_dae, loss='mse')
            # dae_model.summary()
            dae_model.fit(x=alpha_dae, y=target_dae,
                          epochs=dae_epochs, shuffle=False, verbose=0,
                          batch_size=int(alpha_dae.shape[0] / dae_batch_n))
            # Return result
            return encoder_model

        # Fn: (2)
        # Feature selection: gradient boosting regression (CatBoost).
        def fs_gradient_boosting_regression(alpha_gb, target_gb, min_imp=gradient_boosting_min_imp):
            params = {'iterations': gradient_boosting_estimators, 'depth': 3, 'task_type': 'CPU',
                      'learning_rate': gradient_boosting_lr, 'loss_function': 'RMSE',
                      'verbose': 0, 'random_seed': 27}
            # Fit regression model
            train_pool = Pool(alpha_gb, target_gb)
            clf = CatBoostRegressor(**params)
            clf.fit(train_pool)
            # Get feature importance
            feature_importance = clf.get_feature_importance()
            feature_importance = np.divide(feature_importance, np.max(feature_importance))
            feature_importance = np.multiply(feature_importance, 100.00)
            # Update alpha matrix.
            features_hold = np.where(feature_importance >= min_imp)[0]
            alpha_gb = alpha_gb[:, features_hold]
            return alpha_gb, features_hold

        # Fn: (3)
        # Feature eng: stacking bayesian ridge regression (mean and confidence).
        def fe_stacking_bayesian_ridge(alpha_br, target_br):
            # Define the model and fit.
            model_bayes = BayesianRidge(n_iter=bayesian_ridge_iterations, tol=0.001)
            model_bayes.fit(alpha_br, target_br)
            # Predicted mean and posterior dist. std.
            predict_mean, predict_std = model_bayes.predict(alpha_br, return_std=True)
            # Confidence measure.
            predict_confidence = np.abs(predict_mean / predict_std)
            low_confidence_idx = (predict_confidence < np.percentile(predict_confidence, 100 - 50))
            confidence_var = np.ones((alpha_br.shape[0], 1), dtype=np.float32, order='F')
            confidence_var[low_confidence_idx] = 0
            # Stack prediction mean and uncertainty dummy vars.
            alpha_br = np.hstack((alpha_br, predict_mean.reshape(-1, 1)))
            alpha_br = np.hstack((alpha_br, confidence_var))
            return model_bayes, alpha_br

        # Fn: (4)
        # Neural network: activation function.
        # Gaussian Error Linear Unite
        def nn_gelu(x):
            return 0.5 * x * (1 + K.tanh(x * 0.7978845608 * (1 + 0.044715 * K.pow(x, 2))))

        # Fn: (5)
        # Neural network: loss function.
        def nn_loss_sk(y_true, y_pred):
            loss = K.switch(K.less(y_true * y_pred, 0),
                            loss_adj * y_pred ** 2 - K.sign(y_true) * y_pred + K.abs(y_true),
                            K.square(y_pred - y_true))
            return K.mean(loss, axis=-1)

        # Fn: (6)
        # Neural network: loss function (including total return).
        def nn_loss_sk_total(y_true, y_pred):
            # Day 1 loss.
            loss_1 = K.switch(K.less(y_true[:, 0] * y_pred[:, 0], 0),
                              loss_adj * y_pred[:, 0] ** 2 - K.sign(y_true[:, 0]) * y_pred[:, 0] + K.abs(y_true[:, 0]),
                              K.square(y_pred[:, 0] - y_true[:, 0]))
            # Day 2 loss.
            loss_2 = K.switch(K.less(y_true[:, 1] * y_pred[:, 1], 0),
                              loss_adj * y_pred[:, 1] ** 2 - K.sign(y_true[:, 1]) * y_pred[:, 1] + K.abs(y_true[:, 1]),
                              K.square(y_pred[:, 1] - y_true[:, 1]))
            # Day 3 loss.
            loss_3 = K.switch(K.less(y_true[:, 2] * y_pred[:, 2], 0),
                              loss_adj * y_pred[:, 2] ** 2 - K.sign(y_true[:, 2]) * y_pred[:, 2] + K.abs(y_true[:, 2]),
                              K.square(y_pred[:, 2] - y_true[:, 2]))
            # Total return loss.
            loss_4 = K.switch(K.less(y_true[:, 3] * y_pred[:, 3], 0),
                              loss_adj * y_pred[:, 3] ** 2 - K.sign(y_true[:, 3]) * y_pred[:, 3] + K.abs(y_true[:, 3]),
                              K.square(y_pred[:, 3] - y_true[:, 3]))
            # Loss value.
            loss = loss_1 * 0.4 + loss_2 * 0.2 + loss_3 * 0.2 + loss_4 * 0.3
            return K.mean(loss, axis=-1)

        # Fn: (7)
        # Neural network: residual block.
        def nn_residual_block(entry_data, hidden_shape):
            y = Dense(hidden_shape, kernel_regularizer=regularizers.l2(reg_ffn))(entry_data)
            y = LeakyReLU(leaky_slope)(y)
            y = BatchNormalization()(y)
            y = Dropout(dropout_rate)(y, training=True)
            y = layers.add([entry_data, y])
            return y

        # Fn: (8)
        # Neural network: squeeze-and-excitation block used to weight feature maps.
        def nn_senet_block(entry_data, ratio=10):
            nb_channel = K.int_shape(entry_data)[-1]
            y = GlobalAveragePooling1D()(entry_data)
            y = Dense(nb_channel // ratio, activation='relu', kernel_regularizer=regularizers.l2(reg_ffn))(y)
            y = Dense(nb_channel, activation='sigmoid', kernel_regularizer=regularizers.l2(reg_ffn))(y)
            return layers.multiply([entry_data, y])

        # Fn: (9)
        # Neural network: top-down pathway and lateral connection.
        def nn_feature_pyramid_network_block(bottom_tensor, top_tensor):

            # Fn:
            # Up-sampling input tensor shape to ref data.
            # Nearest neighbour method.
            def resize_like(input_tensor, ref_tensor):
                h = ref_tensor.get_shape()[1]
                w = input_tensor.get_shape()[2]
                return tf.image.resize_nearest_neighbor(input_tensor, [h, w])

            # Up-sampling top-up tensor and channels adjustment of bottom-up tensor.
            if top_tensor is not None:
                top_ch = K.int_shape(top_tensor)[-1]
                top_tensor = Lambda(resize_like, arguments={'ref_tensor': bottom_tensor})(top_tensor)
                bottom_tensor = Conv2D(nb_filter=top_ch, kernel_size=(1, 1),
                                       kernel_regularizer=regularizers.l2(reg_conv))(bottom_tensor)
                bottom_tensor = LeakyReLU(leaky_slope)(bottom_tensor)
                bottom_tensor = BatchNormalization()(bottom_tensor)
                feature_map = layers.add([bottom_tensor, top_tensor])
                return feature_map
            else:
                feature_map = bottom_tensor
                return feature_map

        # Fn: (10)
        # Neural network: reduce channels number and weight in input tensor.
        def nn_spatial_reduction_block(input_tensor):
            w = K.int_shape(input_tensor)[2]
            input_tensor = Conv2D(nb_filter=ch_reduction, kernel_size=(1, 1),
                                  kernel_regularizer=regularizers.l2(reg_conv))(input_tensor)
            input_tensor = AveragePooling2D(pool_size=(1, w))(input_tensor)
            input_tensor = LeakyReLU(leaky_slope)(input_tensor)
            input_tensor = BatchNormalization()(input_tensor)
            return input_tensor

        # Fn: (11)
        # First model (alpha) used to predict stock prices.
        # 2-rRNet with DeepConv features + Seq2Seq DeepConv-GRU.
        # (Fine-tuning 3 lvl)
        def nn_model_alpha(alpha_nn, tensor_nn, regime_nn, ones_nn, decoder_nn, target_nn, adv_nn):

            # Reshape target value to match ED req.
            target_rnn = target_nn.reshape(target_nn.shape[0], target_nn.shape[1], 1)

            # Input
            input_alpha = Input(shape=(alpha_nn.shape[1],))
            input_tensor = Input(shape=(tensor_nn.shape[1], tensor_nn.shape[2]))
            input_regime = Input(shape=(regime_nn.shape[1],))
            input_ones = Input(shape=(ones_nn.shape[1],))
            input_decoder = Input(shape=(None, target_rnn.shape[2]))

            # ---------------------------------------------
            #             DEEP CONV. NET.
            #    (Features for FFN and Input for ED-GRU)
            # ---------------------------------------------
            # DeepConvNet architecture
            conv_node = Conv1D(nb_filter=ch_layer_1, filter_length=kernel_conv1d,
                               kernel_regularizer=regularizers.l2(reg_conv))(input_tensor)
            conv_node = LeakyReLU(leaky_slope)(conv_node)
            conv_node = BatchNormalization()(conv_node)
            conv_node = Conv1D(nb_filter=ch_layer_2, filter_length=kernel_conv1d,
                               kernel_regularizer=regularizers.l2(reg_conv))(conv_node)
            conv_node = LeakyReLU(leaky_slope)(conv_node)
            conv_node = BatchNormalization()(conv_node)
            conv_node = nn_senet_block(conv_node)                       # Squeeze-and-excitation

            # Deep Conv. Net. extracted features.
            deep_features = GlobalAveragePooling1D()(conv_node)
            deep_features = Dense(deep_features_n)(deep_features)
            deep_features = Activation('linear')(deep_features)

            # Input to encoder-decoder model.
            input_encoder = MaxPooling1D()(conv_node)

            # ---------------------------------------------
            #              Regime RESNET
            # ---------------------------------------------
            # Pre-layer: adding deep features to alphas and normalization input.
            input_norm = layers.concatenate([input_alpha, deep_features])
            input_norm = BatchNormalization()(input_norm)
            # 1st node.
            res_note = Dense(layer_capacity, kernel_regularizer=regularizers.l2(reg_ffn))(input_norm)
            res_note = Dropout(dropout_rate)(res_note, training=True)
            res_note = LeakyReLU(leaky_slope)(res_note)
            res_note = BatchNormalization()(res_note)
            res_note = nn_residual_block(res_note, layer_capacity)      # Residual Block
            res_note = Dense(4)(res_note)
            res_note = Activation('linear')(res_note)
            # 2nd node.
            ord_node = Dense(n_layer_1, kernel_regularizer=regularizers.l2(reg_ffn))(input_norm)
            ord_node = Dropout(dropout_rate)(ord_node, training=True)
            ord_node = LeakyReLU(leaky_slope)(ord_node)
            ord_node = BatchNormalization()(ord_node)
            ord_node = Dense(n_layer_2, kernel_regularizer=regularizers.l2(reg_ffn))(ord_node)
            ord_node = Dropout(dropout_rate)(ord_node, training=True)
            ord_node = Activation('gelu')(ord_node)
            ord_node = BatchNormalization()(ord_node)
            ord_node = Dense(4)(ord_node)
            ord_node = Activation('linear')(ord_node)
            # Market regimes.
            reg_one = Activation('hard_sigmoid')(input_regime)
            reg_one = layers.multiply([reg_one, res_note])
            reg_two = layers.subtract([input_ones, reg_one])
            reg_two = layers.multiply([reg_two, ord_node])
            # Union.
            reg_out = layers.add([reg_one, reg_two])
            reg_out = Dense(4, activation='linear')(reg_out)

            # ---------------------------------------------
            #             ENCODER-DECODER GRU
            # ---------------------------------------------
            # Encoder architecture
            encoder_cells = []
            for hidden_neurons in rnn_cells:
                encoder_cells.append(GRUCell(hidden_neurons,
                                             kernel_regularizer=regularizers.l2(reg_gru)))
            encoder = RNN(encoder_cells, return_state=True)
            encoder_outputs_and_states = encoder(input_encoder)
            encoder_states = encoder_outputs_and_states[1:]
            # Decoder architecture
            decoder_cells = []
            for hidden_neurons in rnn_cells:
                decoder_cells.append(GRUCell(hidden_neurons,
                                             kernel_regularizer=regularizers.l2(reg_gru)))
            decoder = RNN(decoder_cells, return_sequences=True, return_state=True)
            decoder_outputs_and_states = decoder(input_decoder, initial_state=encoder_states)
            # Out
            decoder_outputs = decoder_outputs_and_states[0]
            decoder_outputs = Dense(1, activation='linear')(decoder_outputs)

            # Output layers
            out_single = reg_out
            out_multi = decoder_outputs

            # Optimizer.
            adam_opt = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

            # Model final in-out structure.
            model = Model(inputs=[input_alpha, input_tensor, input_regime, input_ones, input_decoder],
                          outputs=[out_single, out_multi])
            model.compile(optimizer=adam_opt,
                          loss=[nn_loss_sk_total, nn_loss_sk],
                          loss_weights=[1 - s2s_weight, s2s_weight])
            # model.summary()

            # Train the model.
            print('      training neural network - ALPHA')

            # GENERAL FIT <1st>
            # Fit works with all input data.
            model.fit(x=[alpha_nn, tensor_nn, regime_nn, ones_nn, decoder_nn],
                      y=[target_nn, target_rnn[:, :-1]],
                      epochs=epochs_n_general, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            # FINE-TUNING <2nd>
            # Fit takes as input only the latest values trying to capture
            # the most relevant market information.
            latest_index = int(target_nn.shape[0] * 0.1)
            model.fit(x=[alpha_nn[latest_index:], tensor_nn[latest_index:],
                         regime_nn[latest_index:], ones_nn[latest_index:],
                         decoder_nn[latest_index:]],
                      y=[target_nn[latest_index:], target_rnn[latest_index:, :-1]],
                      epochs=epochs_n_fn, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            # FINE-TUNING <3rd>
            # Fit takes as input only the most liquid stocks trying
            # to focus only on predicting them.
            liquid_idx = (adv_nn > 1.25)
            model.fit(x=[alpha_nn[liquid_idx], tensor_nn[liquid_idx],
                         regime_nn[liquid_idx], ones_nn[liquid_idx],
                         decoder_nn[liquid_idx]],
                      y=[target_nn[liquid_idx], target_rnn[liquid_idx, :-1]],
                      epochs=epochs_n_fn, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            # FINE-TUNING <4th>
            # Fit takes model predictions and use only bad predicted data points.
            # --> first let`s take model predictions and rank them.
            model_prediction = model.predict([alpha_nn, tensor_nn, regime_nn, ones_nn, decoder_nn])
            # 2-RRNet predictions handling.
            pred_ffn = model_prediction[0]
            pred_ffn = pred_ffn / scaler_original
            pred_ffn = hl.nn_handle_multi_day_predictions(pred_ffn, ret_sizing, ret_adj)
            # S2S-DeepConv-GRU predictions handling.
            pred_en = model_prediction[1].reshape(model_prediction[1].shape[0], model_prediction[1].shape[1])
            pred_en = pred_en / scaler_original
            pred_en = hl.nn_handle_multi_day_predictions(pred_en, ret_sizing, ret_adj)
            # Final prediction.
            prediction_internal = hl.cs_zscore(pred_ffn) * (1 - s2s_weight) + hl.cs_zscore(pred_en) * s2s_weight
            # --> now define pnl var.
            pnl_predicted = prediction_internal * target_nn[:, 0]
            pnl_predicted = hl.nan_to_zero(hl.cs_rank(pnl_predicted))
            pnl_predicted[pnl_predicted == 0] = 1
            pnl_predicted_idx = (pnl_predicted < 1.5)
            # --> finally fit the model.
            model.fit(x=[alpha_nn[pnl_predicted_idx], tensor_nn[pnl_predicted_idx],
                         regime_nn[pnl_predicted_idx], ones_nn[pnl_predicted_idx],
                         decoder_nn[pnl_predicted_idx]],
                      y=[target_nn[pnl_predicted_idx], target_rnn[pnl_predicted_idx, :-1]],
                      epochs=epochs_n_fn, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            return model

        # Fn: (12)
        # Second model (beta) used to predict stock prices.
        # 2-rRNet with Feature Pyramid Network.
        # (Fine-tuning 3 lvl)
        def nn_model_beta(alpha_nn, tensor_nn, regime_nn, ones_nn, target_nn, adv_nn):

            # Input
            input_alpha = Input(shape=(alpha_nn.shape[1],))
            input_tensor = Input(shape=(tensor_nn.shape[1], tensor_nn.shape[2], tensor_nn.shape[3]))
            input_regime = Input(shape=(regime_nn.shape[1],))
            input_ones = Input(shape=(ones_nn.shape[1],))

            # ---------------------------------------------
            #           Feature Pyramid Network
            #           (deep features for FFN)
            # ---------------------------------------------
            # DeepConvNet architecture
            conv_node_1 = Conv2D(nb_filter=ch_layer_1, kernel_size=kernel_conv2d,
                                 kernel_regularizer=regularizers.l2(reg_conv))(input_tensor)
            conv_node_1 = LeakyReLU(leaky_slope)(conv_node_1)
            conv_node_1 = BatchNormalization()(conv_node_1)
            conv_node_2 = Conv2D(nb_filter=ch_layer_2, kernel_size=kernel_conv2d,
                                 kernel_regularizer=regularizers.l2(reg_conv))(conv_node_1)
            conv_node_2 = LeakyReLU(leaky_slope)(conv_node_2)
            conv_node_2 = BatchNormalization()(conv_node_2)
            conv_node_3 = Conv2D(nb_filter=ch_layer_3, kernel_size=kernel_conv2d,
                                 kernel_regularizer=regularizers.l2(reg_conv))(conv_node_2)
            conv_node_3 = LeakyReLU(leaky_slope)(conv_node_3)
            conv_node_3 = BatchNormalization()(conv_node_3)
            # Top-down feature extraction.
            f_map_3 = nn_feature_pyramid_network_block(bottom_tensor=conv_node_3, top_tensor=None)
            f_map_2 = nn_feature_pyramid_network_block(bottom_tensor=conv_node_2, top_tensor=f_map_3)
            f_map_1 = nn_feature_pyramid_network_block(bottom_tensor=conv_node_1, top_tensor=f_map_2)
            # Double filter to reduce the aliasing effect of up-sampling.
            f_map_3 = nn_spatial_reduction_block(f_map_3)
            f_map_2 = nn_spatial_reduction_block(f_map_2)
            f_map_1 = nn_spatial_reduction_block(f_map_1)
            # Unite feature maps and flatten results in one vector.
            f_map = layers.concatenate([f_map_3, f_map_2, f_map_1], axis=1)
            f_map = Flatten()(f_map)
            # DeepConvNet extracted features.
            deep_features = Dense(deep_features_n)(f_map)
            deep_features = Activation('linear')(deep_features)

            # ---------------------------------------------
            #              Regime RESNET
            # ---------------------------------------------
            # Pre-layer: adding deep features to alphas and normalization input.
            input_norm = layers.concatenate([input_alpha, deep_features])
            input_norm = BatchNormalization()(input_norm)
            # 1st node.
            res_note = Dense(layer_capacity, kernel_regularizer=regularizers.l2(reg_ffn))(input_norm)
            res_note = Dropout(dropout_rate)(res_note, training=True)
            res_note = LeakyReLU(leaky_slope)(res_note)
            res_note = BatchNormalization()(res_note)
            res_note = nn_residual_block(res_note, layer_capacity)          # Residual Block
            res_note = Dense(1)(res_note)
            res_note = Activation('linear')(res_note)
            # 2nd node.
            ord_node = Dense(n_layer_1, kernel_regularizer=regularizers.l2(reg_ffn))(input_norm)
            ord_node = Dropout(dropout_rate)(ord_node, training=True)
            ord_node = LeakyReLU(leaky_slope)(ord_node)
            ord_node = BatchNormalization()(ord_node)
            ord_node = Dense(n_layer_2, kernel_regularizer=regularizers.l2(reg_ffn))(ord_node)
            ord_node = Dropout(dropout_rate)(ord_node, training=True)
            ord_node = Activation('gelu')(ord_node)
            ord_node = BatchNormalization()(ord_node)
            ord_node = Dense(1)(ord_node)
            ord_node = Activation('linear')(ord_node)
            # Market regimes.
            reg_one = Activation('hard_sigmoid')(input_regime)
            reg_one = layers.multiply([reg_one, res_note])
            reg_two = layers.subtract([input_ones, reg_one])
            reg_two = layers.multiply([reg_two, ord_node])
            # Union.
            reg_out = layers.add([reg_one, reg_two])
            reg_out = Dense(1, activation='linear')(reg_out)

            # Output layers
            out_single = reg_out

            # Optimizer.
            adam_opt = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

            # Model final in-out structure.
            model = Model(inputs=[input_alpha, input_tensor, input_regime, input_ones],
                          outputs=[out_single])
            model.compile(optimizer=adam_opt,
                          loss=[nn_loss_sk],
                          loss_weights=[1.0])
            # model.summary()

            # Train the model.
            print('      training neural network - BETA')

            # GENERAL FIT <1st>
            # Fit works with all input data.
            model.fit(x=[alpha_nn, tensor_nn, regime_nn, ones_nn],
                      y=[target_nn],
                      epochs=epochs_n_general, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            # FINE-TUNING <2nd>
            # Fit takes as input only the latest values trying to capture
            # the most relevant market information.
            latest_index = int(target_nn.shape[0] * 0.1)
            model.fit(x=[alpha_nn[latest_index:], tensor_nn[latest_index:],
                         regime_nn[latest_index:], ones_nn[latest_index:]],
                      y=[target_nn[latest_index:]],
                      epochs=epochs_n_fn, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            # FINE-TUNING <3rd>
            # Fit takes as input only the most liquid stocks trying
            # to focus only on predicting them.
            liquid_idx = (adv_nn > 1.25)
            model.fit(x=[alpha_nn[liquid_idx], tensor_nn[liquid_idx],
                         regime_nn[liquid_idx], ones_nn[liquid_idx]],
                      y=[target_nn[liquid_idx]],
                      epochs=epochs_n_fn, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            # FINE-TUNING <4th>
            # Fit takes model predictions and use only bad predicted data points.
            # --> first let`s take model predictions and rank them.
            model_prediction = model.predict([alpha_nn, tensor_nn, regime_nn, ones_nn])
            pred = hl.cs_zscore(model_prediction.ravel())
            # --> now define pnl var.
            pnl_predicted = pred * target_nn
            pnl_predicted = hl.nan_to_zero(hl.cs_rank(pnl_predicted))
            pnl_predicted[pnl_predicted == 0] = 1
            pnl_predicted_idx = (pnl_predicted < 1.5)

            # --> finally fit the model.
            model.fit(x=[alpha_nn[pnl_predicted_idx], tensor_nn[pnl_predicted_idx],
                         regime_nn[pnl_predicted_idx], ones_nn[pnl_predicted_idx]],
                      y=[target_nn[pnl_predicted_idx]],
                      epochs=epochs_n_fn, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            return model

        # Fn: (13)
        # Transfer learning on top of ALPHA model.
        # ........................................
        # (Fine-tuning 3 lvl)
        def nn_model_alpha_transfer_learning(alpha_nn, tensor_nn, regime_nn, ones_nn,
                                             decoder_nn, target_nn, adv_nn):

            # Reshape target value to match ED req.
            target_rnn = target_nn.reshape(target_nn.shape[0], target_nn.shape[1], 1)

            # Get old layers from the trained model.
            out_single = self.alpha_storage.layers[-4].output
            out_multi = self.alpha_storage.layers[-3].output[0]

            # Set new layers.
            out_single = Dense(4, activation='linear')(out_single)
            out_multi = Dense(1, activation='linear')(out_multi)

            # Optimizer.
            adam_opt = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

            # Model final in-out structure.
            model = Model(inputs=self.alpha_storage.input,
                          outputs=[out_single, out_multi])
            model.compile(optimizer=adam_opt,
                          loss=[nn_loss_sk_total, nn_loss_sk],
                          loss_weights=[1 - s2s_weight, s2s_weight])

            # Train the model.
            print('      training neural network - ALPHA (transfer learning)')

            # Freeze layer or turn it to trainable.
            # If our model stands for a period longer than
            # 'freeze_period', we un-freeze layers. If not,
            # we keep selected layers not trainable.
            if self.freeze_control >= freeze_period:
                print('          un-freeze layers')
                for layer in model.layers:
                    layer.trainable = True
            else:
                print('          freeze layers')
                for layer in model.layers:
                    if layer.name in alpha_freeze:
                        print('            ', layer.name)
                        layer.trainable = False

            # TRANSFER LEARNING <1st>
            # Fit works with all input data.
            model.fit(x=[alpha_nn, tensor_nn, regime_nn, ones_nn, decoder_nn],
                      y=[target_nn, target_rnn[:, :-1]],
                      epochs=epochs_n_tf, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            # FINE-TUNING <2nd>
            # Fit takes as input only the latest values trying to capture
            # the most relevant market information.
            latest_index = int(target_nn.shape[0] * 0.1)
            model.fit(x=[alpha_nn[latest_index:], tensor_nn[latest_index:],
                         regime_nn[latest_index:], ones_nn[latest_index:],
                         decoder_nn[latest_index:]],
                      y=[target_nn[latest_index:], target_rnn[latest_index:, :-1]],
                      epochs=epochs_n_fn, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            # FINE-TUNING <3rd>
            # Fit takes as input only the most liquid stocks trying
            # to focus only on predicting them.
            liquid_idx = (adv_nn > 1.25)
            model.fit(x=[alpha_nn[liquid_idx], tensor_nn[liquid_idx],
                         regime_nn[liquid_idx], ones_nn[liquid_idx],
                         decoder_nn[liquid_idx]],
                      y=[target_nn[liquid_idx], target_rnn[liquid_idx, :-1]],
                      epochs=epochs_n_fn, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            # FINE-TUNING <4th>
            # Fit takes model predictions and use only bad predicted data points.
            # --> first let`s take model predictions and rank them.
            model_prediction = model.predict([alpha_nn, tensor_nn, regime_nn, ones_nn, decoder_nn])
            # 2-RRNet predictions handling.
            pred_ffn = model_prediction[0]
            pred_ffn = pred_ffn / scaler_original
            pred_ffn = hl.nn_handle_multi_day_predictions(pred_ffn, ret_sizing, ret_adj)
            # S2S-DeepConv-GRU predictions handling.
            pred_en = model_prediction[1].reshape(model_prediction[1].shape[0], model_prediction[1].shape[1])
            pred_en = pred_en / scaler_original
            pred_en = hl.nn_handle_multi_day_predictions(pred_en, ret_sizing, ret_adj)
            # Final prediction.
            prediction_internal = hl.cs_zscore(pred_ffn) * (1 - s2s_weight) + hl.cs_zscore(pred_en) * s2s_weight
            # --> now define pnl var.
            pnl_predicted = prediction_internal * target_nn[:, 0]
            pnl_predicted = hl.nan_to_zero(hl.cs_rank(pnl_predicted))
            pnl_predicted[pnl_predicted == 0] = 1
            pnl_predicted_idx = (pnl_predicted < 1.5)
            # --> finally fit the model.
            model.fit(x=[alpha_nn[pnl_predicted_idx], tensor_nn[pnl_predicted_idx],
                         regime_nn[pnl_predicted_idx], ones_nn[pnl_predicted_idx],
                         decoder_nn[pnl_predicted_idx]],
                      y=[target_nn[pnl_predicted_idx], target_rnn[pnl_predicted_idx, :-1]],
                      epochs=epochs_n_fn, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            return model

        # Fn: (14)
        # Transfer learning on top of BETA model.
        # ........................................
        # (Fine-tuning 3 lvl)
        def nn_model_beta_transfer_learning(alpha_nn, tensor_nn, regime_nn, ones_nn,
                                            target_nn, adv_nn):

            # Get old layer from the trained model and set new.
            out_single = self.beta_storage.layers[-2].output
            out_single = Dense(1, activation='linear')(out_single)

            # Optimizer.
            adam_opt = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

            # Model final in-out structure.
            model = Model(inputs=self.beta_storage.input,
                          outputs=[out_single])
            model.compile(optimizer=adam_opt,
                          loss=[nn_loss_sk],
                          loss_weights=[1.0])

            # Train the model.
            print('      training neural network - BETA (transfer learning)')

            # Freeze layer or turn it to trainable.
            # If our model stands for a period longer than
            # 'freeze_period', we un-freeze layers. If not,
            # we keep selected layers not trainable.
            if self.freeze_control >= freeze_period:
                print('          un-freeze layers')
                self.freeze_control = 0
                for layer in model.layers:
                    layer.trainable = True
            else:
                print('          freeze layers')
                for layer in model.layers:
                    if layer.name in beta_freeze:
                        print('            ', layer.name)
                        layer.trainable = False

            # TRANSFER LEARNING <1st>
            # Fit works with all input data.
            model.fit(x=[alpha_nn, tensor_nn, regime_nn, ones_nn],
                      y=[target_nn],
                      epochs=epochs_n_tf, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            # FINE-TUNING <2nd>
            # Fit takes as input only the latest values trying to capture
            # the most relevant market information.
            latest_index = int(target_nn.shape[0] * 0.1)
            model.fit(x=[alpha_nn[latest_index:], tensor_nn[latest_index:],
                         regime_nn[latest_index:], ones_nn[latest_index:]],
                      y=[target_nn[latest_index:]],
                      epochs=epochs_n_fn, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            # FINE-TUNING <3rd>
            # Fit takes as input only the most liquid stocks trying
            # to focus only on predicting them.
            liquid_idx = (adv_nn > 1.25)
            model.fit(x=[alpha_nn[liquid_idx], tensor_nn[liquid_idx],
                         regime_nn[liquid_idx], ones_nn[liquid_idx]],
                      y=[target_nn[liquid_idx]],
                      epochs=epochs_n_fn, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            # FINE-TUNING <4th>
            # Fit takes model predictions and use only bad predicted data points.
            # --> first let`s take model predictions and rank them.
            model_prediction = model.predict([alpha_nn, tensor_nn, regime_nn, ones_nn])
            pred = hl.cs_zscore(model_prediction.ravel())
            # --> now define pnl var.
            pnl_predicted = pred * target_nn
            pnl_predicted = hl.nan_to_zero(hl.cs_rank(pnl_predicted))
            pnl_predicted[pnl_predicted == 0] = 1
            pnl_predicted_idx = (pnl_predicted < 1.5)

            # --> finally fit the model.
            model.fit(x=[alpha_nn[pnl_predicted_idx], tensor_nn[pnl_predicted_idx],
                         regime_nn[pnl_predicted_idx], ones_nn[pnl_predicted_idx]],
                      y=[target_nn[pnl_predicted_idx]],
                      epochs=epochs_n_fn, batch_size=int(alpha_nn.shape[0] / batches_n),
                      shuffle=False, verbose=2)

            return model

        """        
        ----------------------------------------------------------------------------------------------------------
                                                        MODEL
                                    Training loop goes through each refit date
                                        to fit the model on historical data.
        ----------------------------------------------------------------------------------------------------------
        """
        # Update Activation class.
        get_custom_objects().update({'gelu': Activation(nn_gelu)})
        # Define dictionary to store trained models.
        model_dict = dict()

        # Walk through each day.
        for di, date in enumerate(dates):
            if di <= lookback or date <= fit_startdate:
                continue

            # Fit the model.
            if refit_dates_mask[di]:
                print('')
                print('  Fitting on {}'.format(date))

                # Start and end indices
                idx_start = di - delay - lookback
                idx_end = di - delay - 1

                # PREPROCESS DATA.
                # ---------------------------------------------
                # Target 1st day.
                target_reg = np.copy(target_1[:, idx_start: idx_end])
                target_reg[:, -1 - delay:] = np.nan
                target_reg = target_reg.reshape(-1, order='F')
                target_valids = np.isfinite(target_reg)
                target_reg = target_reg[target_valids]
                # Target 2nd day.
                target_2_reg = np.copy(target_2[:, idx_start: idx_end])
                target_2_reg[:, -1 - delay:] = np.nan
                target_2_reg = target_2_reg.reshape(-1, order='F')
                target_2_reg = target_2_reg[target_valids]
                # Target 3rd day.
                target_3_reg = np.copy(target_3[:, idx_start: idx_end])
                target_3_reg[:, -1 - delay:] = np.nan
                target_3_reg = target_3_reg.reshape(-1, order='F')
                target_3_reg = target_3_reg[target_valids]
                # Target (Zscore)
                target_reg_z = np.copy(target_z[:, idx_start: idx_end])
                target_reg_z[:, -1 - delay:] = np.nan
                target_reg_z = target_reg_z.reshape(-1, order='F')
                target_reg_z = target_reg_z[target_valids]

                # Tensor original
                or_reg = np.copy(tensor_data_original[:, idx_start: idx_end])
                or_reg[:, -1 - delay:] = np.nan
                or_reg = or_reg.reshape(or_reg.shape[0] * or_reg.shape[1],
                                        tensor_or_periods, len(tensor_fields_original), order='F')
                or_reg = or_reg[target_valids]

                # Tensor scalogram
                sc_reg = np.copy(tensor_data_wavelet[:, idx_start: idx_end])
                sc_reg[:, -1 - delay:] = np.nan
                sc_reg = sc_reg.reshape(sc_reg.shape[0] * sc_reg.shape[1], tensor_sc_periods,
                                        tensor_sc_comps, len(tensor_fields_wavelet), order='F')
                sc_reg = sc_reg[target_valids]

                # Regime var.
                regime_var = np.copy(regime_mat[:, idx_start: idx_end])
                regime_var[:, -1 - delay:] = np.nan
                regime_var_di = regime_var.reshape(-1, order='F')
                regime_reg = regime_var_di[target_valids]
                # Liquidity var.
                adv_var = np.copy(adv_mat[:, idx_start: idx_end])
                adv_var[:, -1 - delay:] = np.nan
                adv_var_di = adv_var.reshape(-1, order='F')
                adv_reg = adv_var_di[target_valids]
                # -----------------------------------------------

                # Load alphas.
                alpha_reg = np.empty((target_reg.shape[0], len(alphas_list)), dtype=np.float64, order='F')
                for i, alpha_name in enumerate(alphas_list):
                    temp_alpha = joblib.load(self.path_alphas + alpha_name)
                    alpha = np.copy(temp_alpha[:, idx_start: idx_end])
                    alpha = hl.zero_to_nan(alpha)
                    alpha = hl.nan_to_zero(alpha.reshape(-1, order='F')[target_valids])
                    alpha_reg[:, i] = alpha
                # Load events.
                event_reg = np.empty((target_reg.shape[0], len(events_list)), dtype=np.int32, order='F')
                for i, event_name in enumerate(events_list):
                    temp_event = joblib.load(self.path_events + event_name)
                    event = np.copy(temp_event[:, idx_start: idx_end])
                    event = hl.zero_to_nan(event)
                    event = hl.nan_to_zero(event.reshape(-1, order='F')[target_valids])
                    event_reg[:, i] = event

                # Cap alphas min/max values and unite.
                alpha_reg[alpha_reg > max_alpha] = max_alpha
                alpha_reg[alpha_reg < -max_alpha] = -max_alpha
                alpha_reg[np.abs(alpha_reg) < min_alpha] = 0
                alpha_reg = hl.nan_to_zero(alpha_reg)
                target_reg = hl.nan_to_zero(target_reg)

                # Break condition
                if alpha_reg.size == 0 or target_reg.size == 0:
                    continue
                assert alpha_reg.shape[0] == target_reg.shape[0], 'Shapes doesn`t match'
                assert or_reg.shape[0] == target_reg.shape[0], 'Shapes doesn`t match'
                assert sc_reg.shape[0] == target_reg.shape[0], 'Shapes doesn`t match'

                print('      vars. loaded')

                """                
                    Pre-model.   
                    Brief description:
                        - Prepare data to feed the model.   
                    Structure:
                        - Feature Eng: D-SAE model extracts high level deep features from given alpha matrix.
                        - Feature selection: GBoosting based feature importance filter drops irrelevant features.
                        - Stacking (alpha): add events to alpha matrix.
                        - Stacking (alpha): Bayesian Ridge model predicts target values and stack 
                        predictions to alpha matrix. 
                        - Stacking (target): add two more days to target matrix (original).
                        - Stacking (target): add total return (3 days) to target values (original).

                """
                # Extract deep features:
                #   - De-nosing stacked autoencoder
                production_encoder = fe_extract_features_dae(alpha_reg)
                encoded_reg = production_encoder.predict(alpha_reg)
                alpha_reg = np.hstack((alpha_reg, encoded_reg))
                print('          autoencoder trained')

                # Features importance filter:
                #  - Based on GBoosting regressor (CatBoost)
                if not self.start_transfer_learning:
                    alpha_reg, self.selected_alphas = fs_gradient_boosting_regression(alpha_reg, target_reg)
                else:
                    alpha_reg = alpha_reg[:, self.selected_alphas]
                print('          gradient boosting regression trained')

                # Stacking events:
                #     - EPS data.
                #     - DIV data.
                alpha_reg = np.hstack((alpha_reg, event_reg))

                # Stacking predicted target values
                #     - Bayesian Ridge regression
                production_bayes, alpha_reg = fe_stacking_bayesian_ridge(alpha_reg, target_reg_z)
                print('          bayesian ridge regression trained')

                # Stack multiple predictions (3 days)
                target_reg = np.hstack((target_reg.reshape(-1, 1), target_2_reg.reshape(-1, 1)))
                target_reg = np.hstack((target_reg, target_3_reg.reshape(-1, 1)))

                # Stack total return column (4th)
                target_total_ret = ((1 + target_reg[:, 0]) * (1 + target_reg[:, 1]) * (1 + target_reg[:, 2]) - 1)
                target_reg = np.hstack((target_reg, target_total_ret.reshape(-1, 1)))

                # Scale original target and tensor values.
                target_reg = target_reg * scaler_original
                or_reg = or_reg * scaler_original

                # Final check
                alpha_reg = hl.nan_to_zero(alpha_reg)
                target_reg = hl.nan_to_zero(target_reg)
                target_reg_z = hl.nan_to_zero(target_reg_z)
                adv_reg = hl.nan_to_zero(adv_reg)
                regime_reg = hl.nan_to_zero(regime_reg)

                print('      pre-model phase completed')

                """
                    MODEL.        
                    Brief description:
                        - Core model used to predict stock prices on each day.  
                    Structure:                                        
                        - ALPHA: 2-rRNet w/ DeepConv features in pair with Seq2Seq DeepConv-GRU predict
                        '3 day+total' and '3 day' returns together based on two different loss functions.  
                        - BETA: 2-rRNet in pair with Feature Pyramid Network predict
                         '1 day' returns based on original stock loss function.  
                    Details:
                        - General training: general + fine tuning of all layers. 
                        - Transfer learning (if activated): transfer learning w/ newly initialized last 
                          layers + fine tuning of all layers.
                        - The configuration for the model was found after a little trial and error and is by no 
                          means optimized! Calibrating the model parameters is a long journey.

                """
                # Model variable inputs.
                ones_reg = np.ones(shape=(alpha_reg.shape[0], 1))
                regime_reg = np.reshape(regime_reg, (regime_reg.size, 1)) * 100
                decoder_reg = np.zeros((alpha_reg.shape[0], 3, 1))

                # Update freeze control value.
                self.freeze_control += refit_freq

                # GENERAL FIT on first refit day.
                # -------------------------------
                if not self.start_transfer_learning:
                    # Run neural network ALPHA model.
                    production_model_alpha = nn_model_alpha(alpha_reg, or_reg, regime_reg,
                                                            ones_reg, decoder_reg, target_reg, adv_reg)
                    # Run neural network BETA model.
                    production_model_beta = nn_model_beta(alpha_reg, sc_reg, regime_reg,
                                                          ones_reg, target_reg_z, adv_reg)
                    # Activate TF yes/no.
                    self.start_transfer_learning = np.bool(True * activate_tf)
                    self.freeze_control = 0

                # TRANSFER LEARNING (if activated)
                # --------------------------------
                else:
                    # Run neural network ALPHA model (transfer learning).
                    production_model_alpha = nn_model_alpha_transfer_learning(alpha_reg, or_reg, regime_reg, ones_reg,
                                                                              decoder_reg, target_reg, adv_reg)
                    # Run neural network BETA model (transfer learning).
                    production_model_beta = nn_model_beta_transfer_learning(alpha_reg, sc_reg, regime_reg, ones_reg,
                                                                            target_reg_z, adv_reg)

                # Get encoder architecture and weights.
                encoder_weights = production_encoder.get_weights()
                encoder_architecture = production_encoder.to_json()

                # Fill model dictionary.
                model_di = {'production_bayes': production_bayes, 'selected_alphas': self.selected_alphas,
                            'production_model_alpha': production_model_alpha,
                            'production_model_beta': production_model_beta,
                            'encoder_weights': encoder_weights, 'encoder_architecture': encoder_architecture}

                # Save the model to model dictionary.
                model_dict[di] = model_di
                # Update alpha-beta storage.
                self.alpha_storage = production_model_alpha
                self.beta_storage = production_model_beta
                print('      model trained and saved')

        print('')
        print('Fit completed')

        """
        --------------------------------------------------------------------------------------------------------
                                                     PREDICTION
                                    Positions are taken proportional to predicted 
                                            returns from the trained model.
        --------------------------------------------------------------------------------------------------------
        """
        print('')
        print('Prediction started')

        # Refit indecies from fit function.
        refit_indices = np.array(sorted(model_dict.keys()))
        first_index = refit_indices[0]

        # Write first index to data file.
        self.data['first.index'] = first_index
        joblib.dump(self.data, self.path_data)

        # Load model vars.
        production_bayes = None
        production_encoder = None
        production_model_alpha = None
        production_model_beta = None
        selected_alphas = None

        # Load alphas into a dictionary.
        alpha_dict = {}
        for idx in alphas_list:
            alpha = joblib.load(self.path_alphas + idx)
            alpha = hl.nan_to_zero(alpha)
            alpha_dict[idx] = np.copy(alpha, order='F')

        # Load events into a dictionary
        event_dict = {}
        for idx in events_list:
            event = joblib.load(self.path_events + idx)
            event = hl.nan_to_zero(event)
            event_dict[idx] = np.copy(event, order='F')

        # Define positions matrix
        positions = np.zeros(shape=(numstocks, numdates), dtype=np.float64, order='F')

        # Go through each day.
        # Recreate model on refit day and predict on ordinary day.
        # Fill the positions matrix on today.
        for di, date in enumerate(dates):

            # Break if too early.
            if di < first_index:
                continue

            # Reload the model.
            if di in refit_indices:
                print('  refit day: {}'.format(di))
                # Load vars.
                production_bayes = model_dict[di]['production_bayes']
                encoder_weights = model_dict[di]['encoder_weights']
                encoder_architecture = model_dict[di]['encoder_architecture']
                production_model_alpha = model_dict[di]['production_model_alpha']
                production_model_beta = model_dict[di]['production_model_beta']
                selected_alphas = model_dict[di]['selected_alphas']
                # Setup encoder.
                production_encoder = model_from_json(encoder_architecture)
                production_encoder.set_weights(encoder_weights)

            # ALPHA
            # Define 1-DAY alpha and event matrices used to predict stock prices.
            alpha_reg = np.empty((numstocks, len(alphas_list)), dtype=np.float64, order='F')
            event_reg = np.empty((numstocks, len(events_list)), dtype=np.int32, order='F')
            # Load alphas on given day.
            for i, idx in enumerate(alphas_list):
                alpha = np.copy(alpha_dict[idx][:, di - delay])
                alpha = hl.zero_to_nan(alpha)
                alpha_reg[:, i] = hl.nan_to_zero(alpha)
            # Load events on given day.
            for i, idx in enumerate(events_list):
                event = np.copy(event_dict[idx][:, di - delay])
                event = hl.zero_to_nan(event)
                event_reg[:, i] = hl.nan_to_zero(event)
            # Prepare alpha mat.
            alpha_reg[alpha_reg > max_alpha] = max_alpha
            alpha_reg[alpha_reg < -max_alpha] = -max_alpha
            alpha_reg[np.abs(alpha_reg) < min_alpha] = 0

            # TENSOR
            # Get time series tensor for NN model (Original)
            or_reg = tensor_data_original[:, di - delay]
            or_reg = or_reg * scaler_original
            # Get scalogram tensor for NN model (Zscore)
            sc_reg = tensor_data_wavelet[:, di - delay]

            # PRE-MODEL
            encoded_reg = production_encoder.predict(alpha_reg)                         # Extract deep features
            alpha_reg = np.hstack((alpha_reg, encoded_reg))                             # Deep features adding
            alpha_reg = alpha_reg[:, selected_alphas]                                   # Features selection
            # Unite alphas and events in one matrix.
            alpha_reg = np.hstack((alpha_reg, event_reg))                               # Events stacking
            alpha_reg = hl.nan_to_zero(alpha_reg)  # ......
            # Bayesian model.
            b_mean, b_std = production_bayes.predict(alpha_reg, return_std=True)        # Bayesian ridge model pred.
            p_conf = np.abs(b_mean / b_std)                                             # Confidence measure:
            low_conf_idx = (p_conf < np.percentile(p_conf, 100 - 50))                   # .......
            conf_var = np.ones((alpha_reg.shape[0], 1), dtype=np.float32, order='F')    # .......
            conf_var[low_conf_idx] = 0                                                  # .......
            alpha_reg = np.hstack((alpha_reg, b_mean.reshape(-1, 1)))                   # Stacking predicted mean
            alpha_reg = np.hstack((alpha_reg, conf_var))                                # Stacking pred. confidence

            # MODEL
            # Variables.
            ones_reg = np.ones(shape=(alpha_reg.shape[0], 1))                           # Ones vector
            decoder_reg = np.zeros((alpha_reg.shape[0], 3, 1))                          # Decoder teacher
            regime_reg = hl.nan_to_zero(regime_mat[:, di - delay])                      # Regime var.
            regime_reg = np.reshape(regime_reg, (regime_reg.size, 1))
            regime_reg = regime_reg * 100

            """
            PREDICT
            MC Dropout sampling: 
                - Using random instances via dropout we can 'replicate' bayesian network.
                    - ALPHA
                    - BETA
            """
            #                                          ALPHA model
            # --------------------------------------------------------------------------------------------
            predictions = np.zeros((alpha_reg.shape[0], mc_sampling_n), dtype=np.float32, order='F')
            for i in range(mc_sampling_n):
                prediction = production_model_alpha.predict([alpha_reg, or_reg, regime_reg, ones_reg, decoder_reg])
                # 2-rRNet predictions handling.
                prediction_ffn = prediction[0]
                prediction_ffn = prediction_ffn / scaler_original
                prediction_ffn = hl.nn_handle_multi_day_predictions(prediction_ffn, ret_sizing, ret_adj)
                # S2S-DeepConv-GRU predictions handling.
                prediction_encoder = prediction[1].reshape(prediction[1].shape[0], prediction[1].shape[1])
                prediction_encoder = prediction_encoder / scaler_original
                prediction_encoder = hl.nn_handle_multi_day_predictions(prediction_encoder, ret_sizing, ret_adj)
                # Final prediction.
                predictions[:, i] = hl.cs_zscore(prediction_ffn) * (1 - s2s_weight) \
                                    + hl.cs_zscore(prediction_encoder) * s2s_weight
            predicted_returns_alpha = np.nanmean(predictions, axis=1)
            # --------------------------------------------------------------------------------------------

            #                                          BETA model
            # --------------------------------------------------------------------------------------------
            predictions = np.zeros((alpha_reg.shape[0], mc_sampling_n), dtype=np.float32, order='F')
            # Reshape time series data to match Conv2D req.
            for i in range(mc_sampling_n):
                prediction = production_model_beta.predict([alpha_reg, sc_reg, regime_reg, ones_reg])
                # 2-rRNet-FPN prediction handling.
                predictions[:, i] = hl.cs_zscore(prediction.ravel())
            predicted_returns_beta = np.nanmean(predictions, axis=1)
            # ---------------------------------------------------------------------------------------------

            # Fill positions mat. w/ predicted values.
            positions[:, di] = predicted_returns_alpha * am_weight + predicted_returns_beta * (1 - am_weight)

        # Final check.
        positions = positions[:, first_index:]
        positions = hl.nan_to_zero(positions)

        # Save model to disk.
        joblib.dump(positions, self.path_positions)

        print('Predictions completed')
        return positions


if __name__ == '__main__':
    t = time.time()
    Fit().run_model()
    print(time.time() - t)
