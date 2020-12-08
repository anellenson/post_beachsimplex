import pandas as pd
import numpy as np 
import matplotlib.pyplot as pl
import matplotlib as mpl
import pickle
import os
import networkx as nx
from collections import Counter 
import datetime as dt
import anlTools
from sklearn import metrics

#Lippman-Holman Data
def load_holman_data(filename):
    dataset = pd.read_excel('DuckBarhop_fixed_bar_3ab.xlsx')
    predsH = dataset['Beach State (LH90) - fixed'].values
    daynoH = dataset.iloc[:,1].values
    dateH = np.array([dt.datetime(1986, 9, 30, 0, 0, 0) + dt.timedelta(days = dd) for dd in daynoH])


    inds = np.argwhere(~np.isnan(predsH))[:,0]
    dateH  = dateH[inds]
    predsH = predsH[inds]
    daynoH = daynoH[inds]

    return predsH, dateH, daynoH

def translate_to_WandS(predsH):
    translation = {2:1, 3:2, 4:2, 5:3, 6:4, 7:4}
    translated_predsH = []
    for pp in predsH:
        translated_predsH.append(translation[pp])

    translated_predsH = np.array(translated_predsH)

    return translated_predsH

def load_cnn(predictionsfile):

    with open(predictionsfile, 'rb') as f:
        cnn_preds = pickle.load(f)

    predicted_imgname = list(cnn_preds.keys())
    preds = list(cnn_preds.values())
    predicted_imgname = [ss.split('/')[-1] for ss in predicted_imgname]
    timestamps = [np.int(ss.split('.')[0]) for ss in predicted_imgname]

    sorted_inds = np.argsort(timestamps)
    timestamps = np.array([timestamps[tt] for tt in sorted_inds])
    predicted_imgname = np.array([predicted_imgname[pp] for pp in sorted_inds])
    preds = np.array([preds[pp] for pp in sorted_inds])
    dates = np.array([dt.datetime.fromtimestamp(tt) for tt in timestamps])
    dayno_cnn = np.array([(dd - dates[0]).days for dd in dates])

    return preds, dates, dayno_cnn

def smooth_preds(preds_CNN):
    days = 2
    average_preds = np.zeros((len(preds_CNN) - days*2))
    for pi,pred in enumerate(preds_CNN[days:-days]):
        average = np.sum(preds_CNN[pi-days:pi+days])/(days*2)
        average_preds[pi] = np.round(average)

        if pred == 3:
            average_preds[pi] = pred

    average_preds = np.concatenate((np.array(preds_CNN[:days]), average_preds, np.array(preds_CNN[-days:])))

    return average_preds



def align_with_H(preds, dates, dayno_cnn, dateH, predsH, daynoH):
    #round to the nearest day
    dateH = np.array([dd.replace(hour = 0) for dd in dateH])
    dates = np.array([dd.replace(hour = 0) for dd in dates])
    dates = np.unique(dates)

    #Now find the intersection
    vals, indsCNN, indsH = np.intersect1d(dates, dateH, return_indices = True)

    #Turn the CNN predictions to numerical predictions
    preds_CNN = np.array([classes.index(pp) for pp in preds])

    #find the common indices
    preds_CNN = np.array(preds_CNN[indsCNN])
    date_CNN = np.array(dates[indsCNN])
    dayno_CNN = np.array(dayno_cnn[indsCNN])

    predsH = predsH[indsH]
    dateH = dateH[indsH]
    daynoH = daynoH[indsH]

    for pi, pp in enumerate(preds_CNN):
        if pp == 0:
            preds_CNN[pi] = preds_CNN[pi-1]
        if pp == 3 and predsH[pi] < 3:
            preds_CNN[pi] = 2

        if pp == 3 and predsH[pi] > 3:
            preds_CNN[pi] = 4




    preds_CNN = preds_CNN - 1
    predsH = predsH -1

    preds_CNN_smooth = smooth_preds(preds_CNN)

    return preds_CNN_smooth, preds_CNN, date_CNN, dayno_CNN, predsH, dateH, daynoH

run = 1

modelname = 'resnet512_five_aug_{}'.format(run)
classes = ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']
#############
proj_dir =  '/home/server/pi/homes/aellenso/Research/DeepBeach/'
plotdir = 'plots/'


predsH, datesH, daynoH = load_holman_data('DuckBarhop_fixed_bar_3ab.xlsx')
preds, dates, dayno_cnn = load_cnn('predictions_{}.pickle'.format(modelname))
predsH = translate_to_WandS(predsH)
preds_CNN_smooth, preds_CNN, date_CNN, dayno_CNN, predsH, datesH, daynoH = align_with_H(preds, dates, dayno_cnn, datesH, predsH, daynoH)


# daynoH = np.array([(dd - dateH[0]).days for dd in dateH])

#####Time series plots
anlTools.plot_ts(date_CNN, datesH, preds_CNN, preds_CNN_smooth, predsH, 'CNN_ts_RBB_bothcorr{}.png'.format(modelname))



confusion_matrix = metrics.confusion_matrix(predsH, preds_CNN)
f1_score = metrics.f1_score(predsH, preds_CNN, average='weighted')
plot_fname = 'conftabel_RBB_bothcorr{}.png'.format(modelname)
anlTools.confusionTable(confusion_matrix, f1_score, plot_fname)

confusion_matrix = metrics.confusion_matrix(predsH, preds_CNN_smooth)
f1_score = metrics.f1_score(predsH, preds_CNN_smooth, average='weighted')
plot_fname = 'conftabel_RBB_bothcorr{}_smooth.png'.format(modelname)
anlTools.confusionTable(confusion_matrix, f1_score, plot_fname)



count_table, res_time_mean, res_time_std, trans_to, P_occ = anlTools.resTime(preds_CNN, dayno_CNN)
plotfname = 'results_RBB_bothcorr{}.png'.format(modelname)
anlTools.plotResTime(res_time_mean, res_time_std, trans_to, P_occ, plotfname)


count_table, res_time_mean, res_time_std, trans_to, P_occ = anlTools.resTime(preds_CNN_smooth, dayno_CNN)
plotfname = 'results_RBB_bothcorr{}_smooth.png'.format(modelname)
anlTools.plotResTime(res_time_mean, res_time_std, trans_to, P_occ, plotfname)

count_table, res_time_mean, res_time_std, trans_to, P_occ = anlTools.resTime(predsH, daynoH)
anlTools.plotResTime(res_time_mean, res_time_std, trans_to, P_occ, 'Original_Holman_WandS.png')


