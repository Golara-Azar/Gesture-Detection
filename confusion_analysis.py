import h5py
import os
import sys
import tqdm
import time
import numpy as np
import math
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Embedding
from tensorflow.keras import regularizers
from sklearn.preprocessing import OneHotEncoder
import scipy.io as sio
from itertools import combinations, chain
from timeit import default_timer as timer
from sklearn import metrics



EXPERIMENT = 'nature_data'

tf.set_random_seed(42)
tf.debugging.set_log_device_placement(False)    # clean training progress bar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # disable AVX support warning
#os.environ['CUDA_VISIBLE_DEVICES'] = "0" #disable cuda related warning
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
print('# GPUs: ', len(tf.config.list_physical_devices('GPU')))


# global variables
SAMPLE_RATE = 2000    # 2048 Hz but for convenience in windowing
SIGNAL_TYPE = 'transient'
SIGNAL_LEN = 5
TRANSIENT = 0.5#float(sys.argv[1])
REP_LEN = int(SAMPLE_RATE*SIGNAL_LEN) if SIGNAL_TYPE == 'complete' else int(SAMPLE_RATE*TRANSIENT)
WIN_LEN = 400    # sliding window length
WIN_INC = 20    # sliding window increment


TTL_MOVES = 65#int(sys.argv[1])
MOVES = list(range(1, TTL_MOVES+1))
REPS = list(range(1, 6))




TRAIN_REPS = [1, 3, 5]
TEST_REPS = [2, 4]#[4] if 13 in SUBJECTS else [2, 4]



MODE = int(sys.argv[1])
if MODE==1:
    T_REPS =  [1, 3, 5]
elif MODE==2:
    T_REPS = [1, 3]
elif MODE==3:
    T_REPS = [1, 5]
elif MODE==4:
    T_REPS = [3, 5]
elif MODE==5:
    T_REPS = [1]
elif MODE==6:
    T_REPS = [3]
elif MODE==7:
    T_REPS = [5]
else:
     print('wrong MODE')
     T_REPS = TRAIN_REPS
    
    

print('retrain reps: ',T_REPS)
#TRAIN_REPS
V_REPS = TEST_REPS

normalize=True



SAVEDIR = f'./with_embedding_no_freeze/results/'
FIGDIR = f'./with_embedding_no_freeze/figures/'
MODELDIR = f'./with_embedding_no_freeze/models/'
DATADIR = "/scratch/ga2148/NYU_project/HD_EMG_Nature/"
#model parameters
MODEL = 'BILSTM' #CNN BILSTM LSTM
print('model: ',MODEL)
HIDDEN_UNIT = 32
EMB_DIM = 32 
IN_DENSE = EMB_DIM 
DILATION_ORDER = 3
DILATION = 2**DILATION_ORDER
LAYERS = 3

FIGURE = os.path.join(FIGDIR, f'Final_{HIDDEN_UNIT}hidden_order{DILATION_ORDER}dilated_{MODEL}_{LAYERS}Layers_EMB{EMB_DIM}_NOunitnorm_INDENSE{IN_DENSE}_{len(MOVES)}gestures_accuracy_transient{0.5}.png' )


TEST_SUBS = list(range(1,21))
for item in [1,11,10,14,6]:
    TEST_SUBS.remove(item)
print('test subjects:',TEST_SUBS)


def loadHDEMG(data_dict, which_moves):
    
    """
    Obtains emg, moves, and reps (including rest) given a group of gestures.
    @param data_dict: Dictionary, data dictionary from a .mat file
    @param which_moves: List, target gestures
    return emgs: 3D Array (# data points x 64 sensors x 2 electrode grids)
           moves: Vector, gesture # of all data points
           reps: Vector, rep # of all data points
    """
    
    # load in moves
    moves = data_dict['adjusted_class'][()].swapaxes(0, 1).ravel()    # adjusted labels to include transient phase
    idxs = np.where(np.isin(moves, which_moves))[0]

    # load in emgs, reps, and force
    moves = moves[idxs]
    emg_extensors = data_dict['emg_extensors'][:, :, idxs].transpose((2, 0, 1)).reshape(-1, 64)
    emg_flexsors = data_dict['emg_flexors'][:, :, idxs].transpose((2, 0, 1)).reshape(-1, 64)
    reps = data_dict['adjusted_repetition'][:, idxs].swapaxes(0, 1)

    
    return np.hstack((emg_extensors, emg_flexsors)), moves, reps.ravel()

def muLaw(emg, mu=2048):
    
    """
    Applies mu-law transformation on each data point.
    @param emg: Array, sEMG signals.
    @param mu: Integer, mu value.
    return: Array, mu-law transformed sEMG signals.
    """
    
    return np.sign(emg) * np.log(1+mu*abs(emg)) / np.log(1+mu)

def normalize(emgs, tr_inds, feature_range=(-1, 1), mu=2048):
    
    """
    Applies normalization pipeline (Min-Max, Mu-law, z-score) to sEMG signals.
    @param emgs: Array, sEMG signals.
    @param tr_inds: Vector, training indices.
    @param feature_range: Tuple, desired range of feature values.
    @param mu: Integer, mu value.
    return emgs_normalized: Array, normalized sEMG signals.
    """

    tr_emgs = emgs[tr_inds]
    """
    # min-max
    tr_emgs_min = tr_emgs.min(axis=0)
    tr_emgs_max = tr_emgs.max(axis=0)
    tr_emgs_std = (tr_emgs - tr_emgs_min) / (tr_emgs_max - tr_emgs_min)
    tr_emgs_minMax = tr_emgs_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    # mu-law
    tr_emgs_mu_temp = muLaw(tr_emgs_minMax, mu=mu)
    # rescale to (0, mu)
    tr_emgs_mu_temp_min = tr_emgs_mu_temp.min(axis=0)
    tr_emgs_mu_temp_max = tr_emgs_mu_temp.max(axis=0)
    tr_emgs_mu_temp = mu * (tr_emgs_mu_temp-tr_emgs_mu_temp_min) / (tr_emgs_mu_temp_max-tr_emgs_mu_temp_min)
    tr_emgs_mu = tr_emgs_mu_temp.astype(np.int32)
    
    # z-score
    tr_emgs_mean = np.mean(tr_emgs_mu, axis=0)
    tr_emgs_std = np.std(tr_emgs_mu, axis=0)
    
    # normalize entire data set
    emgs_std = (emgs - tr_emgs_min) / (tr_emgs_max - tr_emgs_min)
    emgs_minMax = emgs_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    emgs_mu_temp = muLaw(emgs_minMax, mu=mu)
    emgs_mu_temp = mu * (emgs_mu_temp-tr_emgs_mu_temp_min) / (tr_emgs_mu_temp_max-tr_emgs_mu_temp_min)
    emgs_mu = emgs_mu_temp.astype(np.int32)
    
    emgs_normalized = (emgs_mu - tr_emgs_mean) / tr_emgs_std
    """
    tr_emgs_mean = np.mean(tr_emgs, axis=0)
    tr_emgs_std = np.std(tr_emgs, axis=0)
    emgs_normalized = (emgs - tr_emgs_mean) / tr_emgs_std
    
    return emgs_normalized

   
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
    
    # loop through each move and rep
    cnt = 0
    with tqdm.tqdm(total=n_moves*n_reps) as pbar:
        for m in which_moves:
            for r in which_reps:
                idxs = np.where((moves == m) & (reps == r))[0]
                assert len(idxs) != 0, f'missing move {m}, rep {r}'
                # get window start and end
                win_ends = np.array(range(idxs[0]+win_len, idxs[0]+rep_len+1, win_inc))
                for end in win_ends:               
                    start = end - win_len
                    assert moves[start] == moves[end], f'start:{moves[start]}, end:{moves[end]}'                    
                    emgs_win = emgs[start:end]          
                    X[cnt] = emgs_win
                    y[cnt] = m
                    cnt += 1
                pbar.update(1)
    
    return X, y


def get_idxs(in_array, to_find):
    """Utility function for finding the positions of observations of one array in another an array.

    Args:
        in_array (array): Array in which to locate elements of to_find
        to_find (array): Array of elements to locate in in_array

    Returns:
        TYPE: Indices of all elements of to_find in in_array
    """
    targets = ([np.where(in_array == x) for x in to_find])
    return np.squeeze(np.concatenate(targets, axis=1))
    
# labels=['Siege', 'Initiation', 'Crowd_control', 'Wave_clear', 'Objective_damage']
labels = list(range(1,66))
# markers = [0, 1, 2, 3, 4, 5]
markers = [0,20,40,60,80,100]
# str_markers = ["0", "1", "2", "3", "4", "5"]
str_markers = ["0","20%","40%","60%","80%","100%"]

def make_radar_chart(name, stats, attribute_labels=labels,
                     plot_markers=markers, plot_str_markers=str_markers):

    labels = np.array(attribute_labels)

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats,[stats[0]]))
    angles = np.concatenate((angles,[angles[0]]))

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    plt.yticks(markers)
    ax.set_title(name)
    ax.grid(True)
    
    DIR = os.path.join(FIGDIR, f'Radar_{HIDDEN_UNIT}hidden_order{DILATION_ORDER}dilated_{MODEL}_{LAYERS}Layers_EMB{EMB_DIM}_NOunitnorm_INDENSE{IN_DENSE}_{len(MOVES)}gestures_accuracy_retrain{T_REPS}_transient{0.5}.pdf' )
            
    fig.savefig("static/images/%s.png" % name)

    return #plt.show()

make_radar_chart("Agni", [2,3,4,4,5]) # example


oneHot_encoder = OneHotEncoder(sparse=False)
oneHot_encoder.fit(np.array(MOVES).reshape(-1, 1))    
cm_matrices = []

# for ges in MOVES:
#     print(ges,oneHot_encoder.transform(np.array(ges).reshape(1,-1)))
            

for item in TEST_SUBS:
            
            RETRAIN_MODEL = MODELDIR+f's{item}_{int(WIN_LEN/SAMPLE_RATE*1000)}ms_{HIDDEN_UNIT}hidden_{DILATION}dilated_{MODEL}_{LAYERS}Layers_EMB{EMB_DIM}_NOunitnorm_INDENSE{IN_DENSE}_{len(MOVES)}gestures_retrain{T_REPS}__transient{0.5}.keras'

            
            V_REPS = [4] if item==13 else TEST_REPS
        
            data = h5py.File(os.path.join(DATADIR, f's{item}.mat'), 'r')
            print(f'\nloading data - s{item} ...')
            emgs, moves, reps = loadHDEMG(data, MOVES)
            
            print('train reps:',T_REPS)
            print('test reps:',V_REPS)
            
            # normalize w/ rest reps
            print('normalizing data ...')
            tr_inds = np.where(np.isin(reps, T_REPS) == True)[0]
            emgs = normalize(emgs, tr_inds, feature_range=(-1, 1), mu=2048)
            
            print('windowing training data ...')
            Xtr_s, ytr_s = get_windows(emgs, moves, reps, MOVES,
                                       T_REPS, REP_LEN, WIN_LEN, WIN_INC)
            tr_s = np.ones_like(ytr_s) * (item)
            Xtr, ytr, tr_subs = Xtr_s, ytr_s, tr_s
            
            # convert labels to one-hot
            #oneHot_encoder = OneHotEncoder(sparse=False)
            ytr = oneHot_encoder.transform(ytr.reshape(-1, 1))
            
            
            
            print('windowing testing data ...')
            
            Xts_s, yts_s = get_windows(emgs, moves, reps, MOVES,
                                       V_REPS, REP_LEN, WIN_LEN, WIN_INC)
                                       
                                       
            ts_s = np.ones_like(yts_s) * (item)
            Xts, yts, ts_subs = Xts_s, yts_s, ts_s
            yts = oneHot_encoder.transform(yts.reshape(-1, 1))
            
            
            print('data shape:')
            print(Xtr.shape)
            print(ytr.shape)
            print(Xts.shape)
            print(yts.shape)
            
            train_nums = Xtr.shape[0]
            test_nums = Xts.shape[0]
            
            
            model = K.models.load_model(RETRAIN_MODEL)
            #model.summary()
            
            results = model.evaluate([Xts,ts_subs], yts)
            
            
            print("after training subject "+str(item)+" test acc:", results[1]*100)
            
            y_pred = model.predict([Xts,ts_subs])
            confusion_matrix = metrics.confusion_matrix(np.argmax(yts, axis = 1), np.argmax(y_pred, axis = 1)) 
            cm_matrices.append(confusion_matrix)
            
print(np.shape(cm_matrices))
M = np.sum(cm_matrices,axis=0)  
print(M.shape)
            
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = M, display_labels = MOVES)
            
CONFUSIONFIG = os.path.join(FIGDIR, f'Sum_{HIDDEN_UNIT}hidden_order{DILATION_ORDER}dilated_{MODEL}_{LAYERS}Layers_EMB{EMB_DIM}_NOunitnorm_INDENSE{IN_DENSE}_{len(MOVES)}gestures_accuracy_retrain{T_REPS}_transient{0.5}.pdf' )
            
fig, ax = plt.subplots(figsize=(65,65))
cm_display.plot(ax=ax)
#cm_display.plot()
cm_display.figure_.savefig(CONFUSIONFIG)


            


