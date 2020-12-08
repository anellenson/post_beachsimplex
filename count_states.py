import scipy.io as sio
import anlTools
import pandas as pd
import matplotlib.pyplot as pl
from itertools import chain
import numpy as np
import seaborn as sns

def smooth_preds(preds_CNN, days):

    average_preds = np.zeros((len(preds_CNN) - days*2))
    for pi,pred in enumerate(preds_CNN[days:-days]):
        average = np.sum(preds_CNN[pi-days:pi+days])/(days*2)
        average_preds[pi] = np.round(average)

        if pred == 3:
            average_preds[pi] = pred

    average_preds = np.concatenate((np.array(preds_CNN[:days]), average_preds, np.array(preds_CNN[-days:])))

    return average_preds


def replace_ref(preds_CNN):

    for li, state in enumerate(preds_CNN.label):
        if state == 0:
            preds_CNN.iloc[li].label = preds_CNN.iloc[li-1].label

    return preds_CNN
### load mat data
def load_mat(matname):
    cnn_preds = sio.loadmat(matname)
    labels = cnn_preds['beachstate_ts']
    data = {}
    for ri, row in enumerate(labels.T):
        data.update({'CNN{}'.format(ri): row})

    labels_ensemble_df = pd.DataFrame(data = data)
    maj_vote_df = labels_ensemble_df.mode(axis = 1)
    wavedata = {}
    for var in ['MWD', 'Tm', 'Hs']:
        var_array = list(chain.from_iterable(cnn_preds[var]))
        varlist = []
        for hh in var_array:
            value = hh[0][0]
            varlist.append(hh[0][0])

        wavedata.update({var:varlist})

    wavedata.update({'label':maj_vote_df[0].values-1})

    preds_df = pd.DataFrame(data = wavedata, index = cnn_preds['datenums'].squeeze())
    preds_df.sort_index(inplace = True)

    return preds_df

def load_simplices(picklename):
    waveparams = pd.read_pickle('data/wavedata_df.pickle')
    cnn_pickle = pd.read_pickle(picklename)
    cnn_out = cnn_pickle['cnn_simplex']
    cnn_simplex_array = np.empty(len(cnn_out), 5)

    for pi,pid in enumerate(waveparams.pid):
        cnn_simplex_array[pi] = pid

    return cnn_simplex_array




#Set up preds_df
cnn_simplex_df = load_simplices()
preds_df_smoothed2= preds_df.copy()
preds_df_smoothed2_removeref = preds_df.copy()
preds_df_smoothed2.label = smooth_preds(preds_df.label, 4)
preds_df_smoothed2_removeref = preds_df_smoothed2.copy()
preds_df_smoothed2_removeref = replace_ref(preds_df_smoothed2_removeref)

anl = anlTools.StateAnl(preds_df)
anl2 = anlTools.StateAnl(preds_df_smoothed2)
anl_noref = anlTools.StateAnl(preds_df_smoothed2_removeref)

anltitles = ['no reflective state', 'smoothed']
for i, anlobj in enumerate([anl_noref, anl2]):
    count_table, res_time_mean, res_time_std, trans_to, P_occ = anlobj.resTime()

    N_trans_LH = np.array([4,30, 62, 20])
    N_trans_LH = N_trans_LH/N_trans_LH.sum()
    trans_to = trans_to/trans_to.sum()
    ntrans_df = pd.DataFrame(index = ['LTT', 'TBR', 'RBB', 'LBT'], data = {'Duck: Lippmann Holman': N_trans_LH, 'Duck: Current Study': trans_to[1:-1]})
    fig, ax  = pl.subplots(1,1)
    fig.suptitle(anltitles[i])
    ntrans_df.plot(kind = 'bar', ax = ax)
    ax.set_ylabel('Number of Transitions')
    ax.set_xlabel('Classes')


    vartable, vardiff = anlobj.waveSwitch('Hs')

    Pocc_LH = [9, 47, 24, 21]
    Pocc_PB = [5, 55, 28, 12]
    pocc_df = pd.DataFrame(index = ['LTT', 'TBR', 'RBB', 'LBT'], data = {'Palm Beach: Ranasinghe (2004)':Pocc_PB, 'Duck: Lippmann Holman':Pocc_LH, 'Duck: Current Study':P_occ[1:-1]*100})
    fig, ax = pl.subplots(1,1)
    pocc_df.plot(kind = 'bar', ax =ax)
    pl.legend(loc = 'upper right')
    ax.set_xlabel('Beach State')
    ax.set_title('Probability of Occurrence')
    ax.set_ylabel('%')


    ###Palm Beach Data
    Hs_PB = [1.21, 1.55, 1.59, 2.25]
    Hs_LH = [0.6, 1, 0.99, 1.5]
    hs_df = pd.DataFrame(index = ['LTT', 'TBR', 'RBB', 'LBT'], data = {'Palm Beach: Ranasinghe (2004)':Hs_PB, 'Duck: Lippman Holman':Hs_LH, 'Duck: Current Study':vartable[1:-1]})
    fig, ax = pl.subplots(1,1)
    fig.suptitle(anltitles[i])
    fig.set_size_inches([3, 6])
    hs_df.plot(kind = 'bar', ax = ax)
    ax.set_xlabel('Beach State')
    ax.set_ylabel('Hs [m]')
    ax.set_title('Significant Wave Height for State')

    res_time_PB = [9.3, 22.8, 11.8, 5.4]
    res_time_LH = [18, 7.2, 4.7, 2.5]
    restime_df = pd.DataFrame(index = ['LTT', 'TBR', 'RBB', 'LBT'], data = {'Palm Beach: Ranasinghe (2004)':res_time_PB, 'Duck: Lippman Holman':res_time_LH, 'Duck: Current Study':res_time_mean[1:-1]})

    res_timestd_PB = [14.6, 18.8, 11.4, 3.8]
    res_timestd_LH = [13.8, 7.6, 4.6, 2.3]
    restimestd_df = pd.DataFrame(index = ['LTT', 'TBR', 'RBB', 'LBT'], data = {'Palm Beach: Ranasinghe (2004)':res_time_PB, 'Duck: Lippman Holman':res_time_LH, 'Duck: Current Study':res_time_std[1:-1]})

    fig, ax = pl.subplots(2,1, sharex = True)
    fig.suptitle(anltitles[i])
    fig.set_size_inches([3, 9])
    restime_df.plot(kind = 'bar', ax = ax[0])
    restimestd_df.plot(kind='bar', ax= ax[1])
    ax[0].set_title('Residence Time')
    ax[1].set_title('Standard Deviation')


count_table_PB = np.zeros((4,4))
count_table_PB[0, 1] = 7
count_table_PB[1,0] = 2
count_table_PB[1,2] = 30
count_table_PB[2, 1] = 3
count_table_PB[2, 3] = 30
count_table_PB[3, 0] = 5
count_table_PB[3, 1] = 22
count_table_PB[3,2] = 3

count_table_LH = np.zeros((4,4))
count_table_LH[0, 1] = 3
count_table_LH[1, 0] = 2
count_table_LH[1, 2] = 16
count_table_LH[1, 3] = 3
count_table_LH[2, 1] = 5
count_table_LH[2, 3] = 25
count_table_LH[3, 0] = 1
count_table_LH[3, 1] = 13
count_table_LH[3, 2] = 14

fig, ax = pl.subplots(3,1, sharex = True)
cb = ax[0].pcolor(count_table_PB, cmap= 'Greys', vmin = 0)
pl.colorbar(cb, ax = ax[0])
cb = ax[1].pcolor(count_table_LH, cmap= 'Greys', vmin = 0)
pl.colorbar(cb, ax = ax[1])
cb = ax[2].pcolor(count_table[1:-1, 1:-1], cmap= 'Greys', vmin = 0)
pl.colorbar(cb, ax = ax[2])


