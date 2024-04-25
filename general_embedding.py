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

# import tensorflow as tf1
# physical_devices = tf1.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)


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

"""
#top 7 subjects
if TTL_MOVES==65:
    SUBJECTS = [20, 2, 1, 17, 4, 11, 12]  #updated
elif TTL_MOVES==45:
    SUBJECTS = [17, 2, 20, 14, 4, 12, 1]
elif TTL_MOVES==25:
    SUBJECTS = [20, 17, 9, 11, 4, 1, 10]
elif TTL_MOVES==10:
    SUBJECTS = [11, 16, 20, 12, 1, 17, 2]

#worst 7 subjects
if TTL_MOVES==65:
    SUBJECTS = [19, 5, 3, 13, 16, 8, 7] #updated
elif TTL_MOVES==45:
    SUBJECTS = [19, 3, 13, 5, 16, 18, 7]
elif TTL_MOVES==25:
    SUBJECTS = [3, 19, 5, 13, 8, 7, 18]
elif TTL_MOVES==10:
    SUBJECTS = [3, 19, 7, 8, 5, 14, 18]
"""


#SUBJECTS = [19, 5, 3, 13, 16, 8, 7] #worst 7
#SUBJECTS = [20, 2, 1, 17, 4, 11, 12] #top 7
RANDOM = [1,11,10,14,6,16,3]
NUM_SUBS = 5#int(sys.argv[1])
SUBJECTS = RANDOM[:NUM_SUBS]

print('training subjects:',SUBJECTS)    # single-element list for 1 subject, multi-element list for >1 subjects


TRAIN_REPS = [1, 3, 5]
TEST_REPS = [2, 4]#[4] if 13 in SUBJECTS else [2, 4]

RETRAIN = True
TRAIN=False

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
    print('pre-training phase')
    RETRAIN = False
    TRAIN=True
    T_REPS = TRAIN_REPS
    
    

print('retrain reps: ',T_REPS)
#TRAIN_REPS
V_REPS = TEST_REPS

normalize=True
FLEXIBLE=['emb_subj','h2','dense2']#['emb_subj','h2'] 'h2','dense2'
SESSION = 'sensors_v2'#'with_embedding_no_freeze'

SAVEDIR = f'./{SESSION}/results/'
FIGDIR = f'./{SESSION}/figures/'
MODELDIR = f'./{SESSION}/models/'
DATADIR = "/scratch/ga2148/NYU_project/HD_EMG_Nature/"
#model parameters
MODEL = 'BILSTM' #CNN BILSTM LSTM
print('model: ',MODEL)
HIDDEN_UNIT = 32
EMB_DIM = 32 
IN_DENSE = EMB_DIM 
DILATION_ORDER = 3
DILATION = 2**DILATION_ORDER
LAYERS = 4

BESTMODEL = f'init_{SUBJECTS}_{int(WIN_LEN/SAMPLE_RATE*1000)}ms_{HIDDEN_UNIT}hidden_{DILATION}dilated_{MODEL}_{LAYERS}Layers_EMB{EMB_DIM}_NOunitnorm_INDENSE{IN_DENSE}_{len(MOVES)}gestures_transient{0.5}.h5'
FIGURE = os.path.join(FIGDIR, f'init_{SUBJECTS}_{HIDDEN_UNIT}hidden_order{DILATION_ORDER}dilated_{MODEL}_{LAYERS}Layers_EMB{EMB_DIM}_NOunitnorm_INDENSE{IN_DENSE}_{len(MOVES)}gestures_accuracy_transient{0.5}.png' )
RESULTS = os.path.join(SAVEDIR, f'init_{SUBJECTS}_{HIDDEN_UNIT}hidden_order{DILATION_ORDER}dilated_{MODEL}_{LAYERS}Layers_EMB{EMB_DIM}_NOunitnorm_INDENSE{IN_DENSE}_{len(MOVES)}gestures_results_retrain100epochs_retrain{T_REPS}_transient{TRANSIENT}.csv' )
PRETRAIN_RESULTS = os.path.join(SAVEDIR, f'pretraining_{SUBJECTS}_{HIDDEN_UNIT}hidden_order{DILATION_ORDER}dilated_{MODEL}_{LAYERS}Layers_EMB{EMB_DIM}_NOunitnorm_INDENSE{IN_DENSE}_{len(MOVES)}gestures_results_retrain100epochs_retrain{T_REPS}_transient{TRANSIENT}.csv' )

#RESULTS = os.path.join(SAVEDIR,'no_calib.csv')

TEST_SUBS = list(range(1,21))
for item in SUBJECTS:
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
#     force = abs(data_dict['force'][:, idxs].swapaxes(0, 1) - 2.5)
    
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

def step_decay(epoch):
    """
    Applies a step decay on learning rate
    that reduces the rate by half in every 20 epochs.
    """
    global LR, EPOCHS
    initial_lrate = LR
    drop = 0.5
    epochs_drop = EPOCHS // 3
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# Load data dictionary for all training subjects
data_dicts = []
for s in SUBJECTS:
    data_dict = h5py.File(os.path.join(DATADIR, f's{s}.mat'), 'r')
    data_dicts.append(data_dict)
    
# load data for all training subjects
for s, data in enumerate(data_dicts):
    print(f'\nloading data - s{SUBJECTS[s]} ...')
    emgs, moves, reps = loadHDEMG(data, MOVES)
    
    #print('reps:',np.unique(reps))
    #print('moves:',np.unique(moves))
    
    if SUBJECTS[s]==13:
        TEST_REPS = [4]
    
    # normalize w/ rest reps
    print('normalizing data ...')
    tr_inds = np.where(np.isin(reps, TRAIN_REPS) == True)[0]
    emgs = normalize(emgs, tr_inds, feature_range=(-1, 1), mu=2048) #StandardScalar() fit_transform train transform test data
    
    print('windowing training data ...')
    Xtr_s, ytr_s = get_windows(emgs, moves, reps, MOVES,
                               TRAIN_REPS, REP_LEN, WIN_LEN, WIN_INC)
    tr_s = np.ones_like(ytr_s) * (SUBJECTS[s])
    print('windowing testing data ...')
    Xts_s, yts_s = get_windows(emgs, moves, reps, MOVES,
                               TEST_REPS, REP_LEN, WIN_LEN, WIN_INC)
    ts_s = np.ones_like(yts_s) * (SUBJECTS[s])
    if s == 0:
        Xtr, ytr, Xts, yts = Xtr_s, ytr_s, Xts_s, yts_s
        tr_subs, ts_subs = tr_s, ts_s
    else:
        Xtr = np.concatenate((Xtr, Xtr_s))
        Xts = np.concatenate((Xts, Xts_s))
        ytr = np.append(ytr, ytr_s)
        yts = np.append(yts, yts_s)
        tr_subs = np.append(tr_subs, tr_s)
        ts_subs = np.append(ts_subs, ts_s)
        
    TEST_REPS = [2,4]

print('gestures: ',np.unique(ytr))        
# convert labels to one-hot
oneHot_encoder = OneHotEncoder(sparse=False)
ytr = oneHot_encoder.fit_transform(ytr.reshape(-1, 1))
yts = oneHot_encoder.transform(yts.reshape(-1, 1))

print('subjects: ',np.unique(tr_subs))
print('Data shape:')
print(Xtr.shape)
print(ytr.shape)
print(tr_subs.shape)
print(Xts.shape)
print(yts.shape)
print(ts_subs.shape)

in_shape = Xtr.shape[1:]
sub_shape = 1

#base model
inputs = Input(shape=in_shape, name='emg_in')
users = Input(shape=sub_shape, name='sub_in')

if MODEL=='BILSTM':
    #for checking traditional TL, we remove embedding. The retraining will be on only dense layers
    
    LSTM1,forward_h, forward_c, backward_h, backward_c= Bidirectional(LSTM(HIDDEN_UNIT, return_sequences=True, return_state=True, dropout=0.2),merge_mode='sum', name = 'bi1')(inputs)#
    LSTM1_d = LSTM1[:,::-DILATION,:][:,::-1,:]
        
    LSTM2,forward_h, forward_c, backward_h, backward_c= Bidirectional(LSTM(HIDDEN_UNIT, return_sequences=True, return_state=True, dropout=0.2),merge_mode='sum', name = 'bi2')(LSTM1_d)#, return_sequences=True, return_state=True
    LSTM2_d = LSTM2[:,::-DILATION,:][:,::-1,:]
        
    LSTM3,forward_h, forward_c, backward_h, backward_c= Bidirectional(LSTM(HIDDEN_UNIT, return_sequences=True, return_state=True, dropout=0.2),merge_mode='sum', name = 'bi3')(LSTM2_d)#, return_sequences=True, return_state=True
    LSTM3_d = LSTM3[:,::-DILATION,:][:,::-1,:]
    
    LSTM4,forward_h, forward_c, backward_h, backward_c= Bidirectional(LSTM(HIDDEN_UNIT, return_sequences=True, return_state=True, dropout=0.2),merge_mode='sum', name = 'bi4')(LSTM3_d)#, return_sequences=True, return_state=True
    LSTM4_d = LSTM4[:,::-DILATION,:][:,::-1,:]
    
    state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
    h1 = tf.reshape(state_h, shape=[-1, 1, HIDDEN_UNIT*2], name='lstm_out')
    
    #embedding
    h2 = Dense(IN_DENSE,activation='tanh', name = 'h2')(h1) #tanh
    h2_drop = Dropout(0.2)(h2)
    
    EMBED_WEIGHTS = np.zeros((21, EMB_DIM))
    u = Embedding(input_dim=21, output_dim=EMB_DIM, name='emb_subj',  input_length=1, weights=[EMBED_WEIGHTS])(users)
    
    h4 = h2_drop*u
    h4_flat = Flatten(name='h4_flat')(h4) 
    #h4_flat = Flatten(name='h2_flat')(h2_drop) 
    predictions = Dense(len(MOVES), activation='softmax', name = 'dense2')(h4_flat) 
    
    
elif MODEL=='LSTM':
    
    LSTM1, _, _ = LSTM(2*HIDDEN_UNIT, dropout=0.2,return_sequences=True, return_state=True,kernel_initializer=glorot_uniform(seed=42))(inputs)
    LSTM1_d = LSTM1[:,::-DILATION,:][:,::-1,:]
    LSTM2, _, _ = LSTM(2*HIDDEN_UNIT, dropout=0.2,return_sequences=True, return_state=True,kernel_initializer=glorot_uniform(seed=42))(LSTM1_d)
    LSTM2_d = LSTM2[:,::-DILATION,:][:,::-1,:]
    LSTM3, state_h3, _ = LSTM(2*HIDDEN_UNIT, dropout=0.2,return_sequences=True, return_state=True,kernel_initializer=glorot_uniform(seed=42))(LSTM2_d)
    LSTM3_d = LSTM3[:,::-DILATION,:][:,::-1,:]
    
    h1 = tf.reshape(state_h3, shape=[-1, 1, HIDDEN_UNIT*2], name='lstm_out')
    
    h2 = Dense(IN_DENSE, activation='tanh', name='h2')(h1)
    h2_drop = Dropout(0.2)(h2)
    EMBED_WEIGHTS = np.zeros((21, EMB_DIM))
    
    u = Embedding(input_dim=21, output_dim=EMB_DIM, name='emb_subj',  input_length=1, weights=[EMBED_WEIGHTS])(users)
    
    h4 = h2_drop*u
    h4_flat = Flatten(name='h4_flat')(h4) 
    predictions = Dense(len(MOVES), activation='softmax', name = 'dense2')(h4_flat)
    
    
    
model = Model(inputs=[inputs,users],outputs=predictions) #[inputs,users]
model.summary()

LR=1e-3
EPOCHS=200

opt_adam = K.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, weight_decay=0.0)
model.compile(loss='categorical_crossentropy' , optimizer=opt_adam, metrics=['categorical_accuracy'])
es = EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=40)
mc = ModelCheckpoint(os.path.join(MODELDIR, BESTMODEL), monitor='val_categorical_accuracy', mode='max', verbose=1, save_best_only=True)


lrate = LearningRateScheduler(step_decay)
#LR=1e-3
#EPOCHS=200

if TRAIN==True:
    start_time = time.perf_counter()
    history = model.fit(x=[Xtr,tr_subs], y=ytr, epochs=EPOCHS, shuffle=True, 
                    verbose=1, validation_data = ([Xts,ts_subs], yts), callbacks=[es, mc, lrate]) 
    end_time = time.perf_counter()
    pretraining_time = end_time-start_time
    print('Model pretraining time: ',pretraining_time)
    
    # plt.figure()
    # plt.plot(history.history['categorical_accuracy'])
    # plt.plot(history.history['val_categorical_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    # plt.savefig(FIGURE)
    
else:
    print('loading the saved model...')
    
model.load_weights(os.path.join(MODELDIR, BESTMODEL))

avg_results = model.evaluate([Xts,ts_subs], yts)
print('avg accuracy on new repetitions',avg_results[1]*100)

#evaluating on each subject
print('--------evaluation on new reps of seen subjects---------')
model.load_weights(os.path.join(MODELDIR, BESTMODEL))

with open(PRETRAIN_RESULTS, 'w', encoding='UTF8' , newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model pretraining time',pretraining_time,'avg accuracy',avg_results[1]*100])
    writer.writerow(['pretraining subject','inference time','number of samples','accuracy'])   

    for item in SUBJECTS:
        
        idx = get_idxs(ts_subs, [item]) #subject idx
        
        data = Xts[idx,...]
        subs = ts_subs[idx]
        data = np.squeeze(data) 
        y = yts[idx,...]
        
        start_time = time.perf_counter()
        results = model.evaluate([data,subs], y)
        end_time = time.perf_counter()
        infer_time = end_time-start_time
        print(f'Inference time for subject{item}: ', infer_time)
        print(f'Number of data samples for subject{item}', data.shape[0])
        
        print("subject "+str(item)+" test acc:", results[1]*100)
        writer.writerow([item,infer_time,data.shape[0],results[1]*100])
        
if RETRAIN==True:
    
    """
    for layer in model.layers:
        if layer.name not in FLEXIBLE:
            layer.trainable = False
    """
    #model.compile(loss='categorical_crossentropy' , optimizer=opt_adam, metrics=['categorical_accuracy'])
    
    #commented for traditional TL
    weight_matrix = np.array(model.get_layer("emb_subj").weights)
    
    avg = np.zeros((1,EMB_DIM))
    for INIT in SUBJECTS:
        avg = avg + weight_matrix[0][INIT]
        
    
    avg = avg/len(SUBJECTS)
    
    NEW_WEIGHTS = np.tile(avg,(21,1))
    for INIT in SUBJECTS:
        NEW_WEIGHTS[INIT] = weight_matrix[0][INIT]
        
    
    model = Model(inputs=[inputs,users],outputs=predictions)
    
    #comment if not freezing any layers
    # for layer in model.layers:
    #     if layer.name not in FLEXIBLE:
    #         layer.trainable = False    
                   
    model.compile(loss='categorical_crossentropy' , optimizer=opt_adam, metrics=['categorical_accuracy'])
    
            

    print('--------retrain : unseen subjects---------')
    
    print('train reps:',T_REPS)
    print('test reps:',V_REPS)
    
    
    with open(RESULTS, 'w', encoding='UTF8' , newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject','untrained acc','accuracy','conv t','train windowing time','training time','test windowing time','test inference time','#train windows','#test windows'])
        
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
            start_time = time.perf_counter()
            Xtr_s, ytr_s = get_windows(emgs, moves, reps, MOVES,
                                       T_REPS, REP_LEN, WIN_LEN, WIN_INC)
            tr_s = np.ones_like(ytr_s) * (item)
            Xtr, ytr, tr_subs = Xtr_s, ytr_s, tr_s
            
            # convert labels to one-hot
            #oneHot_encoder = OneHotEncoder(sparse=False)
            ytr = oneHot_encoder.transform(ytr.reshape(-1, 1))
            
            end_time = time.perf_counter()
            
            train_window_time = end_time-start_time
            
            print('windowing testing data ...')
            start_time = time.perf_counter()
            Xts_s, yts_s = get_windows(emgs, moves, reps, MOVES,
                                       V_REPS, REP_LEN, WIN_LEN, WIN_INC)
                                       
                                       
            ts_s = np.ones_like(yts_s) * (item)
            Xts, yts, ts_subs = Xts_s, yts_s, ts_s
            yts = oneHot_encoder.transform(yts.reshape(-1, 1))
            end_time = time.perf_counter()
            
            test_window_time = end_time-start_time
            
            print('data shape:')
            print(Xtr.shape)
            print(ytr.shape)
            print(Xts.shape)
            print(yts.shape)
            
            train_nums = Xtr.shape[0]
            test_nums = Xts.shape[0]
            
            es = EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=40)
            cb = TimingCallback()
            #mc = ModelCheckpoint(os.path.join(MODELDIR, RETRAIN_MODEL), monitor='val_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
            
            model = Model(inputs=[inputs,users],outputs=predictions)
            model.load_weights(os.path.join(MODELDIR, BESTMODEL))
            model.compile(loss='categorical_crossentropy' , optimizer=opt_adam, metrics=['categorical_accuracy'])
            model.get_layer("emb_subj").set_weights([NEW_WEIGHTS]) #commented for traditional TL
            
            #print('weights before compiling')
            #print(np.array(model.get_layer("emb_subj").weights))
            
            #model.compile(loss='categorical_crossentropy' , optimizer=opt_adam, metrics=['categorical_accuracy'])
            
            #print('weights after compiling')
            #print(np.array(model.get_layer("emb_subj").weights))
                   
            # for i, layer in enumerate(model.layers):
            #     print(i+1, ' | ', layer.name, ' | ', layer.trainable)
            
            #model.summary()
            
            pre_results = model.evaluate([Xts,ts_subs], yts)
            print("before training subject "+str(item)+" acc:", pre_results[1]*100)
            
            start_time = time.perf_counter()
            history = model.fit(x=[Xtr,tr_subs], y=ytr, epochs=100, shuffle=True, 
                            verbose=0, validation_data = ([Xts,ts_subs], yts), callbacks=[es,cb])
            end_time = time.perf_counter()
            training_time = end_time-start_time
            
            conv = getResultStats(history,cb.logs)
            
            start_time = time.perf_counter()
            results = model.evaluate([Xts,ts_subs], yts)
            end_time = time.perf_counter()
            infer_time = end_time-start_time
            
            model.save(RETRAIN_MODEL)
            
            print("after training subject "+str(item)+" test acc:", results[1]*100)
            
            writer.writerow([item, pre_results[1]*100, results[1]*100, conv, train_window_time,training_time, test_window_time, infer_time, train_nums, test_nums])
            
            y_pred = model.predict([Xts,ts_subs])
            confusion_matrix = metrics.confusion_matrix(np.argmax(yts, axis = 1), np.argmax(y_pred, axis = 1)) 
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = MOVES)
            CONFUSIONFIG = os.path.join(FIGDIR, f'S{item}_{HIDDEN_UNIT}hidden_order{DILATION_ORDER}dilated_{MODEL}_{LAYERS}Layers_EMB{EMB_DIM}_NOunitnorm_INDENSE{IN_DENSE}_{len(MOVES)}gestures_accuracy_retrain{T_REPS}_transient{0.5}.pdf' )
            
            fig, ax = plt.subplots(figsize=(65,65))
            cm_display.plot(ax=ax)
            #cm_display.plot()
            cm_display.figure_.savefig(CONFUSIONFIG)
            
            
