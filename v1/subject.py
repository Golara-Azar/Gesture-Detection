# import packages
import h5py
import os
import sys
import tqdm
import time
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow.compat.v1 as tf
import csv
from tensorflow import keras as K
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Dropout, LSTM, Conv1D, BatchNormalization, PReLU, Bidirectional
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding
from tensorflow.keras import regularizers
from sklearn.preprocessing import OneHotEncoder
import scipy.io as sio
from itertools import combinations, chain
from timeit import default_timer as timer
    
class TimingCallback(K.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

def getResultStats(history, cb_logs): #modified from Erin's code

    best = max(history.history['val_categorical_accuracy'])
    times = []
    for i,t in enumerate(cb_logs):
        if history.history['val_categorical_accuracy'][i] >= best*0.95:
            times.append(t)
    convergence = sum(times)/(len(times)*60)
    return convergence

EXPERIMENT = 'nature_data'


tf.set_random_seed(42)
tf.debugging.set_log_device_placement(False)    # clean training progress bar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # disable AVX support warning
#os.environ['CUDA_VISIBLE_DEVICES'] = "0" #disable cuda related warning
print('# GPUs: ', len(tf.config.list_physical_devices('GPU')))


# global variables
SAMPLE_RATE = 2000    # 2048 Hz but for convenience in windowing
SIGNAL_TYPE = 'transient'
SIGNAL_LEN = 5
TRANSIENT = 1
REP_LEN = int(SAMPLE_RATE*SIGNAL_LEN) if SIGNAL_TYPE == 'complete' else int(SAMPLE_RATE*TRANSIENT)
WIN_LEN = 400    # sliding window length
WIN_INC = 20    # sliding window increment
SUBJECTS = list(range(1, 21))    # single-element list for 1 subject, multi-element list for >1 subjects
TTL_MOVES = int(sys.argv[1])
MOVES = list(range(1, TTL_MOVES+1))
REPS = list(range(1, 6))

TRAIN_REPS = [1, 3, 5]
TEST_REPS = [2, 4]

normalize=True

SAVEDIR = f'./Nature_data/subj/'
DATADIR = "./HD_EMG_Nature/"
#model parameters
MODEL = 'BILSTM' #CNN BILSTM LSTM
print('model: ',MODEL)
HIDDEN_UNIT = 32
EMB_DIM = 32 
IN_DENSE = EMB_DIM 
DILATION_ORDER = 3
DILATION = 2**DILATION_ORDER
LAYERS = 3

RESULTS = os.path.join(SAVEDIR, f'subject_specific_{HIDDEN_UNIT}hidden_order{DILATION_ORDER}dilated_{MODEL}_{LAYERS}Layers_EMB{EMB_DIM}_INDENSE{IN_DENSE}_{len(MOVES)}gestures_train{TRAIN_REPS}_results.csv' )


def loadHDEMG(data_dict, which_moves):
    
    """
    Obtains emg, moves, and reps (including rest) given a group of gestures.
    @param data_dict: Dictionary, data dictionary from a .mat file
    @param which_moves: List, target gestures
    return emgs: 2D Array (# data points x 128 (64 sensors x 2 electrode grids))
           moves: Vector, gesture # of all data points
           reps: Vector, rep # of all data points
    """
    
    # load in moves
    moves = data_dict['class'][()].swapaxes(0, 1).ravel()    # non-adjusted labels to include transient phase
    idxs = np.where(np.isin(moves, MOVES))[0]

    # load in emgs, reps, and force
    moves = moves[idxs]
    emg_extensors = data_dict['emg_extensors'][:, :, idxs].transpose((2, 0, 1)).reshape(-1, 64)
    emg_flexsors = data_dict['emg_flexors'][:, :, idxs].transpose((2, 0, 1)).reshape(-1, 64)
    reps = data_dict['repetition'][:, idxs].swapaxes(0, 1)

    
    #return np.dstack((emg_extensors, emg_flexsors)), moves, reps.ravel()
    return np.hstack((emg_extensors, emg_flexsors)), moves, reps.ravel() #?
    
def get_windows(emgs, moves, reps, which_moves,
                which_reps, rep_len, win_len, win_inc):
    
    """
    Conducts windowing on given data.
    @param emgs: Array, EMG signals in timestamps
    @param moves: Vector, gesture of each timestamp
    @param reps: Vector, repetition of each timestamp
    @param which_moves: Vector, target gestures
    @param which_reps: Vector, target repetitions
    @param rep_len: Integer, length of each repetition
    @param win_len: Integer, sliding window size
    @param win_inc: Integer, sliding window stride
    return X: Array, windows of EMG signals
           y: Vector, gesture label of each EMG window
    """
    
    n_moves = len(which_moves)
    n_reps = len(which_reps)
    n_wins_rep = (rep_len - win_len) // win_inc + 1
    n_wins = n_moves * n_reps * n_wins_rep
    X = np.zeros((n_wins, win_len, 128)) 
    y = np.zeros((n_wins, ))
    cnt = 0
    # loop through each move and rep
    with tqdm.tqdm(total=n_moves*n_reps) as pbar:
        for m in which_moves:
            for r in which_reps:
                # get indices for each move and rep
                idxs = np.where((moves == m) & (reps == r))[0]
                assert len(idxs) != 0, f'missing move {m}, rep {r}'
                # get window start and end
                win_ends = np.array(range(idxs[0]+win_len, idxs[0]+rep_len+1, win_inc))
                for end in win_ends:               
                    start = end - win_len
                    assert moves[start] == moves[end], f'start:{moves[start]}, end:{moves[end]}'
                    X[cnt] = emgs[start:end]
                    y[cnt] = m
                    cnt += 1
                pbar.update(1)
    
    return X, y
    
def normPipeline(emgs, tr_inds, feature_range=(-1, 1), mu=2048):
    
    """
    Applies normalization pipeline (Min-Max, Mu-law, z-score) to sEMG signals.
    @param emgs: Array, sEMG signals.
    @param tr_inds: Vector, training indices.
    @param feature_range: Tuple, desired range of feature values.
    @param mu: Integer, mu value.
    return emgs_normalized: Array, normalized sEMG signals.
    """

    tr_emgs = emgs[tr_inds]
    
    # min-max
    tr_emgs_min = tr_emgs.min(axis=(0, 1))
    tr_emgs_max = tr_emgs.max(axis=(0, 1))
    tr_emgs_std = (tr_emgs - tr_emgs_min) / (tr_emgs_max - tr_emgs_min)
    tr_emgs_minMax = tr_emgs_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    # mu-law
    tr_emgs_mu_temp = np.sign(tr_emgs_minMax) * np.log(1+mu*abs(tr_emgs_minMax)) / np.log(1+mu)
    tr_emgs_mu_temp = (tr_emgs_mu_temp + 1) / 2 * mu + 0.5    # scale to (0, mu)
    tr_emgs_mu = tr_emgs_mu_temp.astype(np.int32)
    
    # z-score
    tr_emgs_mean = np.mean(tr_emgs_mu, axis=(0, 1))
    tr_emgs_std = np.std(tr_emgs_mu, axis=(0, 1))
    
    # normalize entire data set
    emgs_std = (emgs - tr_emgs_min) / (tr_emgs_max - tr_emgs_min)
    emgs_minMax = emgs_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    emgs_mu_temp = np.sign(emgs_minMax) * np.log(1+mu*abs(emgs_minMax)) / np.log(1+mu)
    emgs_mu_temp = (emgs_mu_temp + 1) / 2 * mu + 0.5    # scale to (0, mu)
    emgs_mu = emgs_mu_temp.astype(np.int32)
    emgs_normalized = (emgs_mu - tr_emgs_mean) / tr_emgs_std
    
    return emgs_normalized

def build_model(in_shape):

    #base model
    inputs = Input(shape=in_shape, name='emg_in')
    
    if MODEL == 'CNN':
        
        conv1 = Conv1D(12, 7, 1, name = 'conv1', dilation_rate=2)(inputs)
        batch1 = BatchNormalization(name = 'batch1')(conv1)
        prelu1 = PReLU(name = 'prelu1')(batch1)
        
        conv2 = Conv1D(6, 7, 1, name = 'conv2', dilation_rate=2)(prelu1)
        batch2 = BatchNormalization(name = 'batch2')(conv2)
        prelu2 = PReLU(name = 'prelu2')(batch2)
        
        conv3 = Conv1D(1, 5, 1, name = 'conv3', dilation_rate=2)(prelu2) #, dilation_rate=DILATION
        batch3 = BatchNormalization(name = 'batch3')(conv3)
        prelu3 = PReLU(name = 'prelu3')(batch3)
        
        cnn_out = Flatten(name='cnn_flat')(prelu3) 
        
        dense0 = Dense(EMB_DIM, activation='softmax', name = 'dense0')(cnn_out)
        
        dense0_reshaped = tf.reshape(dense0, shape=(-1,1,EMB_DIM), name='dense0_reshaped')
        
        z2 = dense0_reshaped
        z2_flat = Flatten(name='emb_out_flat')(z2) 
        
        predictions = Dense(len(MOVES), activation='softmax', name = 'dense1')(z2_flat) 
            
    
    elif MODEL=='BILSTM':
        
        
        LSTM1,forward_h, forward_c, backward_h, backward_c= Bidirectional(LSTM(HIDDEN_UNIT, return_sequences=True, return_state=True, dropout=0.2),merge_mode='sum', name = 'bi1')(inputs)#
        LSTM1_d = LSTM1[:,::-DILATION,:][:,::-1,:]
        
        LSTM2,forward_h, forward_c, backward_h, backward_c= Bidirectional(LSTM(HIDDEN_UNIT, return_sequences=True, return_state=True, dropout=0.2),merge_mode='sum', name = 'bi2')(LSTM1_d)#, return_sequences=True, return_state=True
        LSTM2_d = LSTM2[:,::-DILATION,:][:,::-1,:]
        
        LSTM3,forward_h, forward_c, backward_h, backward_c= Bidirectional(LSTM(HIDDEN_UNIT, return_sequences=True, return_state=True, dropout=0.2),merge_mode='sum', name = 'bi3')(LSTM2_d)#, return_sequences=True, return_state=True
        LSTM3_d = LSTM3[:,::-DILATION,:][:,::-1,:]
        
        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        h1 = tf.reshape(state_h, shape=[-1, 1, HIDDEN_UNIT*2], name='lstm_out')
            
            
        h2 = Dense(IN_DENSE,activation='tanh', name = 'dense1')(h1)
        h2_drop = Dropout(0.2)(h2)
        h2_flat = Flatten(name='dense1_flat')(h2_drop) 
        predictions = Dense(len(MOVES), activation='softmax', name = 'dense2')(h2_flat) #, kernel_regularizer=tf.keras.regularizers.L1(0.01), activity_regularizer=tf.keras.regularizers.L2(0.01)
            
        
    model = Model(inputs=inputs,outputs=predictions) #[inputs,users]
    model.summary()
    
    return  model

# run model for each subject separately
with open(RESULTS, 'w', encoding='UTF8' , newline='') as f:
    
    writer = csv.writer(f)
    writer.writerow(['subject','accuracy','conv t'])
    
    for s in SUBJECTS:
        
        data = h5py.File(os.path.join(DATADIR, f's{s}.mat'), 'r')
        print(f'\nloading data - s{s} ...')
        emgs, moves, reps = loadHDEMG(data, MOVES)
        
        # normalization
        if normalize:
            #assert train_reps is not None, 'train reps not given.'
            tr_inds = np.where(np.isin(reps, TRAIN_REPS) == True)[0]
            emgs = normPipeline(emgs, tr_inds, feature_range=(-1, 1), mu=2048)
        
        print('windowing training data ...')
        Xtr_s, ytr_s = get_windows(emgs, moves, reps, MOVES,
                                   TRAIN_REPS, REP_LEN, WIN_LEN, WIN_INC)
        tr_s = np.ones_like(ytr_s) * (s)
        print('windowing testing data ...')
        Xts_s, yts_s = get_windows(emgs, moves, reps, MOVES,
                                   TEST_REPS, REP_LEN, WIN_LEN, WIN_INC)
        ts_s = np.ones_like(yts_s) * (s)
        
        Xtr, ytr, Xts, yts = Xtr_s, ytr_s, Xts_s, yts_s
        tr_subs, ts_subs = tr_s, ts_s
          
        # convert labels to one-hot
        oneHot_encoder = OneHotEncoder(sparse=False)
        ytr = oneHot_encoder.fit_transform(ytr.reshape(-1, 1))
        yts = oneHot_encoder.transform(yts.reshape(-1, 1))
        
        print('data shape:')
        print(Xtr.shape)
        print(ytr.shape)
        print(Xts.shape)
        print(yts.shape)
        
        es = EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=40)
        cb = TimingCallback()
        opt_adam = K.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        #mc = ModelCheckpoint(os.path.join(SAVEPATH, BESTMODEL), monitor='val_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
            
        in_shape = Xtr.shape[1:]
        model = build_model(in_shape)
        model.compile(loss='categorical_crossentropy' , optimizer=opt_adam, metrics=['categorical_accuracy'])
            
        history = model.fit(x=Xtr, y=ytr, epochs=200, shuffle=True, 
                                verbose=1, validation_data = (Xts, yts), callbacks=[es, cb]) 
        conv = getResultStats(history,cb.logs)
            
        results = model.evaluate(Xts, yts)
        print("subject "+str(s)+" test acc:", results[1]*100)
            
        writer.writerow([s,results[1]*100, conv])
    

