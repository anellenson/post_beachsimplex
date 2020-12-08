import pandas as pd
import numpy as np 
import matplotlib.pyplot as pl
import pickle
import os 
import networkx as nx
from collections import Counter 
import datetime as dt
import getFRFData as gf
import scipy.io as sio

#############
proj_dir =  '/home/server/pi/homes/aellenso/Research/DeepBeach/python/post_resnet/'
pred_df = sio.loadmat(proj_dir + 'cnn_predictions.mat')

#Longuet Higgins/Holman Data
dataset = pd.read_xcel('DuckBarhop_fixed_bar_3ab.xlsx')
predsH = dataset['Beach State (LH90)-fixed'].values
daynoH = dataset.iloc[:,1].values
predsH = predsH-1 
Hs = dataset.iloc[:,3].values
dateH = np.array([dt.datetime(1986, 10, 1, 6, 0, 0) + dt.timedelta(days = dd) for dd in daynoH])

preds_df = pd.read_pickle(pythondir + 'ResNet/predictions/Predicted_TimeSeries_final_protocol2.pkl')
predicted_imgname = preds_df['img_fnames']
preds = preds_df['preds']
timestamps = [np.int(ss.split('.')[0]) for ss in predicted_imgname]
sorted_inds = np.argsort(timestamps)
timestamps = np.array([timestamps[tt] for tt in sorted_inds])
predicted_imgname = np.array([predicted_imgname[pp] for pp in sorted_inds])
preds = np.array([preds[pp] for pp in sorted_inds])
date = np.array([gf.toDateTime(float(tt)) for tt in timestamps])
dayno = np.array([(dd - date[0]).days for dd in date])

for beachstate in [6,7,8]:
    preds[np.where(preds == beachstate)[0]] = np.nan


idx = np.where((date >= dateH[0]) & (date <= dateH[-1]))[0]
preds = preds[idx]
date = date[idx]
dayno = dayno[idx]

fig, ax = pl.subplots(2,1, tight_layout = True)
fig.set_size_inches(12,4)
ax[0].plot(date, preds, label = 'CNN', color = 'blue')
ax[0].plot(dateH, predsH, label = 'Holman', color = 'orange')
ax[0].set_title('With NaNs')
ax[0].set_ylabel('Beach State')

daynoH = daynoH[~np.isnan(predsH)]
dateH = dateH[~np.isnan(predsH)]
predsH = predsH[~np.isnan(predsH)]


date = date[~np.isnan(preds)]
preds = preds[~np.isnan(preds)]

ax[1].plot(date, preds, label = 'CNN', color = 'blue')
ax[1].plot(dateH, predsH, label = 'Holman', color = 'orange')
ax[1].set_title('Without NaNs')
ax[1].set_ylabel('Beach State')


###Find where it's in line with Holman's data:
classes = ['B','C','D','E','F','G']
count_table = np.zeros((len(classes),len(classes)))
res_time_count = 0
res_time = [[],[],[],[],[],[]]
#firstday = daynoH[0]
#predsH = predsH-1
firstday = dayno[0]
for pi,pp in enumerate(preds[:-1]):
    from_state = int(pp)
    to_state = int(preds[pi + 1])
   
    if to_state == from_state:
        continue
    if to_state != from_state:
        lastday = dayno[pi]
        count_table[to_state, from_state] = count_table[to_state, from_state] + 1
        res_time[from_state].append(lastday - firstday +1)#count inclusively
        firstday = dayno[pi]
        
res_time_mean = [np.nanmean(x) for x in res_time]
res_time_std = [np.nanstd(x) for x in res_time]
    

for vi,var in enumerate(['Hs', 'MWD','Tm']):
    var_table = np.zeros((len(classes),len(classes)))
    vardiff_table = np.zeros((len(classes),len(classes)))
    count_table = np.zeros((len(classes),len(classes)))
    res_time_count = 0
    res_time = [[],[],[],[],[]]

    for ii,ll in enumerate(pred_df.label[:-1]):
      from_state = int(ll)
      to_state = int(pred_df.label.iloc[ii+1])
      var0 = pred_df[var].iloc[ii]
      var1 = pred_df[var].iloc[ii+1]
    
      vardiff = var1 - var0
    
      var_table[to_state, from_state] += var0
      vardiff_table[to_state, from_state] += vardiff
      count_table[to_state, from_state] += 1
      
      if to_state == from_state:
        res_time_count += 1
        
      if to_state != from_state and res_time_count != 0:
        res_time[from_state].append(res_time_count)
        res_time_count = 0
    
    
    var_table = var_table/count_table
    vardiff_table = vardiff_table/count_table
    res_time_mean = [np.mean(x) for x in res_time]
    res_time_std = [np.std(x) for x in res_time]

    pl.figure(vi)
    pl.subplot(211)
    pl.title(var + ' Mean Value')
    pl.table(cellText = np.round(var_table,2), rowLabels = classes, colLabels = classes, loc = 'center' )
    pl.axis('off')
    pl.ylabel('TO')
    pl.text(0.5,0.85,'FROM', fontweight = 'bold')
    pl.text(-0.05,0.5, 'TO', r otation = 90, fontweight = 'bold')

    pl.subplot(212)
    pl.title(var + ' Transition Value')
    pl.table(cellText = np.round(vardiff_table,2), rowLabels = classes, colLabels = classes, loc = 'center' )
    pl.axis('off')
    pl.ylabel('TO')
    pl.text(0.5,0.85,'FROM', fontweight = 'bold')
    pl.text(-0.05,0.5, 'TO', rotation = 90, fontweight = 'bold')


pl.figure(4)
pl.subplot(211)
pl.title('Count')
#pl.table(cellText = np.round(count_table)/7911,2)*100, rowLabels = classes, colLabels = classes, loc = 'center' )
pl.table(cellText = np.round(count_table), rowLabels = classes, colLabels = classes, loc = 'center' )
pl.axis('off')
pl.ylabel('TO')
pl.text(0.5,0.85,'FROM', fontweight = 'bold')
pl.text(-0.05,0.5, 'TO', rotation = 90, fontweight = 'bold')

#Transitions to each state bar graph
trans_to = np.sum(count_table, axis = 1)
y_pos = np.arange(len(classes))
P_occ = trans_to * res_time_mean /(np.sum(trans_to * res_time_mean)) 
fig, ax = pl.subplots(2,1, tight_layout = True, sharex = True)
ax[0].bar(y_pos, P_occ, align='center', fill = False, linewidth = 2)
ax[0].set_ylabel('Probability')
ax[0].set_title('Observations')

ax[1].bar(y_pos, trans_to, align='center', fill = False, linewidth = 2)
ax[1].set_ylabel('Number')
ax[1].set_title('Transitions')
pl.xticks(y_pos, classes)



fig = pl.figure(6)
pl.bar(y_pos, res_time_mean, width = -0.3, align='edge', fill = False, linewidth = 2, label = 'mean')
pl.bar(y_pos, res_time_std, width = 0.3, align = 'edge', fill = False, hatch = '//', label = 'std')
pl.xticks(y_pos, classes)
pl.legend()
pl.ylabel('Number of Days')
plt.title('Residence Time')
pl.savefig('')

#############################Time series plot
fig = pl.figure()
pl.plot(date,preds, label = 'CNN')
pl.plot(dateH,predsH, label = 'Holman')
fig.set_size_inches(12,4)
pl.legend()
pl.ylabel('States')












############# RNN predictions/accuracy

proj_dir =  '/home/server/pi/homes/aellenso/Research/DeepBeach/'
pythondir = '/home/server/pi/homes/aellenso/Research/DeepBeach/python/'
pred_f = 'Predictions_df.pikl' 

CNN_df = pd.read_pickle(pythondir + pred_f)
RNNf = open(pythondir + 'RNN_fcast.pikl','rb')
RNN_pred = pickle.load(RNNf)
RNNf.close()

RNN_pred = RNN_pred['RNN_fcast']
pred_acc = []
for ff in range(len(RNN_pred)):
    oneday_fcast = RNN_pred[0]
    predictions = CNN_df.label.values[ff:] - oneday_fcast[:len(oneday_fcast)-ff]
    acc = len(np.where(predictions == 0)[0])/len(predictions)
    pred_acc.append(acc)
## -- End pasted text --

pl.figure(8)
pl.scatter(np.arange(1,len(pred_acc)+1),pred_acc, color = 'blue')
pl.plot(np.arange(1,len(pred_acc)+1),pred_acc, color = 'blue')
pl.title("Prediction Accuracy By Time Horizon")
pl.xlabel('Days In the Future')
pl.ylabel('Accuracy')
