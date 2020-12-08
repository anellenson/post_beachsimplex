import pandas as pd
import numpy as np 
import matplotlib.pyplot as pl
import pickle
import os 
import networkx as nx
from collections import Counter
import matplotlib.image as mpimg 
import matplotlib as mpl

#############
proj_dir =  '/home/server/pi/homes/aellenso/Research/DeepBeach/'
pythondir = '/home/server/pi/homes/aellenso/Research/DeepBeach/python/'
#pred_f = 'Predictions_df.pikl' 

#pred_df = pd.read_pickle(pythondir + pred_f)
#Longuet Higgins/Holman Data

dataset = pd.read_csv(pythondir + 'HolmanLippmanData.csv')
preds = dataset['Beach State (LH90)'].values
dayno = dataset.iloc[:,1].values
Hs = dataset.iloc[:,3].values
Tm = dataset.iloc[:,4].values

dayno = dayno[~np.isnan(preds)]
Hs = Hs[~np.isnan(preds)]
Tm = Tm[~np.isnan(preds)]
preds = preds[~np.isnan(preds)]
####################################
#Now load resnet predictions and wave parameters from my own dataframe
waveconds = pd.read_pickle("Research/DeepBeach/python/waveconds_imgsnames_df.pickle")
Hs = waveconds['Hs'][:]
Tm = waveconds['Tm'][:]
mwd = waveconds['MWD'][:]
imgname = waveconds['pid'][:] 
preds_df = pd.read_pickle(pythondir + 'ResNet/predictions/Predicted_TimeSeries_final_protocol2.pkl')
predicted_imgname = preds_df['img_fnames']
preds_resnet = preds_df['preds']
preds = []
imgs = []
#line up the predictions with the image names
for ii in imgname:
    idx = predicted_imgname.index(ii)
    prediction = preds_resnet[idx]
    preds.append(prediction)
    imgs.append(predicted_imgname[idx])

for beachstate in [6,7,8]:
    preds_resnet[np.where(preds_resnet == beachstate)[0]] = np.nan



########This is to quantify cycles
alledgelist = []
allimgnames = [] 
edgelist = []
allHslist = []
allTmlist = []
allmwdlist = []
Hslist = []
Tmlist = []
mwdlist = []
basenode = 5
offset = np.where(np.array(preds) == basenode)[0][0]
cycles = []
allcycles = []
start = offset
for ci, c0 in enumerate(preds[offset:-1]):
    ci = ci + offset
    c0 = preds[ci]
    c1 = preds[ci+1]
    if c0 != basenode:
        edge = (c0,c1)
        edgelist.append(edge)
    if c0 == basenode:
        end = ci
        alledgelist.append(edgelist)
        allHslist.append([Hs[start:end+1]])
        allTmlist.append([Tm[start:end+1]])
        allmwdlist.append([mwd[start:end+1]])
        allcycles.append([preds[start:end+1]])
        allimgnames.append([imgs[start:end+1]])
        edgelist = []
        start = ci
        if preds[ci+1] != basenode:
            edge = (preds[ci], preds[ci+1])
            edgelist.append(edge)

#Plot each prediction path
posG = nx.DiGraph()
posG.add_nodes_from(range(0,9))
pos = nx.spring_layout(posG)
for ci,cycle in enumerate(allcycles[10:14]):
    if cycle == []:
        continue
    cycle = cycle[0]
    hours = np.arange(len(cycle))
    switch = np.where(np.diff(cycle) != 0)[0]
    switch = switch + 1
    cleandata = np.where(np.array(allHslist[ci][0]) != 0)[0]
    Hs_ts = np.array(allHslist[ci][0])
    Tm_ts = np.array(allTmlist[ci][0])
    mwd_ts = np.array(allmwdlist[ci][0])
    pl.figure()
    ax1 = pl.subplot2grid((3,2),(0,0), rowspan = 3)
    ax2 = pl.subplot2grid((3,2),(0,1))
    ax3 = pl.subplot2grid((3,2),(1,1))
    ax4 = pl.subplot2grid((3,2),(2,1))
    
    ax2.plot(hours[cleandata], Hs_ts[cleandata])
    for si,ss in enumerate(switch):
        ax2.plot((hours[ss],hours[ss]),(0,3),'k')
        ax2.text(hours[ss] - 1 ,3, str(cycle[ss-1]),fontsize = 8, fontweight = 'bold', color = 'red')
        ax2.text(hours[-1], 3, str(cycle[-1]),fontsize = 8, fontweight = 'bold', color = 'red')

    ax3.plot(hours[cleandata], Tm_ts[cleandata])
    for ss in switch:
        ax3.plot((hours[ss],hours[ss]),(0,15),'k')
        ax3.text(hours[ss] -1, 1, str(cycle[ss-1]),fontsize = 8, fontweight = 'bold', color = 'red')
        ax3.text(hours[-1], 1, str(cycle[-1]),fontsize = 8, fontweight = 'bold', color = 'red')

    ax4.plot(hours[cleandata], mwd_ts[cleandata])
    for ss in switch:
        ax4.plot((hours[ss],hours[ss]),(0,180),'k')
        ax4.text(hours[ss] - 1, 1, str(cycle[ss-1]),fontsize = 8, fontweight = 'bold', color = 'red')
        ax4.text(hours[-1], 1, str(cycle[-1]),fontsize = 8, fontweight = 'bold', color = 'red')


    G = nx.DiGraph()
    G.add_nodes_from(range(0,9))
    G.add_edges_from(alledgelist[ci])
    nx.draw_networkx_nodes(G,pos, ax = ax1)
    nx.draw_networkx_labels(G,pos, ax = ax1)
    nx.draw_networkx_edges(G,pos, ax = ax1)
    ax3.set_title('Tm')
    ax2.set_title('Hs')
    ax4.set_title('mwd')
    #ax3.set_xticks('')
    #ax3.set_yticks('')
    #nx.draw_networkx_edge_labels(G, pos, Hs_labels, ax = ax3)
imgdir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/oblique/test/'
pl.tight_layout()
fig, ax = pl.subplots(4,4)
for ii,img in enumerate(allimgnames[3][0]):
    img = mpimg.imread(imgdir + img)
    ax.ravel('F')[ii].imshow(img, cmap = 'gray')
    ax.ravel('F')[ii].set_title(str(allcycles[3][0][ii]))
    


##########Find mean parameters for each edge:
bulkparam_edgelist = []
for ci, c0 in enumerate(preds[:-1]):
    c0 = preds[ci]
    c1 = preds[ci+1]
    edge = (c0,c1)
    bulkparam_edgelist.append(edge)

wave_edges_df = pd.DataFrame(columns = ['edge','Hs','Tm','mwd','diffHs','diffTm','diffmwd'], index = imgs[:-1])
wave_edges_df['edge'] = bulkparam_edgelist
wave_edges_df['Hs'] = Hs[:-1].values
wave_edges_df['Tm'] = Tm[:-1].values
wave_edges_df['mwd'] = mwd[:-1].values
wave_edges_df['diffHs'] = np.diff(Hs.values)
wave_edges_df['diffTm'] = np.diff(Tm.values)
wave_edges_df['diffmwd'] = np.diff(mwd.values)



unique_G = nx.DiGraph()
#Find the unique edges:
unique_edges = Counter(bulkparam_edgelist)
unique_edges_ = {}
from_edge = []
to_edge = []

for (uu,edgecount) in unique_edges.items():
    if 7.0 in uu or 8.0 in uu or 6.0 in uu:
         continue
    unique_edges_[uu] = edgecount
    
dfcols = ['Hs','Tm','mwd','diffHs','diffTm','diffmwd','edgecount', 'from_edge' , 'to_edge']

unique_df = pd.DataFrame(index = unique_edges_.keys(), columns = dfcols, dtype = float)
for ui, (uu,edgecount) in enumerate(unique_edges_.items()):
    for column in unique_df.columns[:-3]:
        val = wave_edges_df[wave_edges_df['edge'] == uu][column].mean()
        unique_df.loc[uu, column] = val
    unique_df.loc[uu, 'edgecount'] = edgecount
    unique_df.loc[uu, 'from_edge'] = uu[0]
    unique_df.loc[uu, 'to_edge'] = uu[1]

##Remove edges with 7, 8, 9:

G = nx.DiGraph()
G.add_nodes_from(np.arange(0,7))
G.add_edges_from(unique_edges_)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos)
nx.draw_networkx_labels(G,pos)
nx.draw_networkx_edges(G,pos)

unique_df = unique_df.sort_values(by = 'from_edge')
gt = np.where(unique_df.from_edge > unique_df.to_edge)[0]
same = np.where(unique_df.from_edge == unique_df.to_edge)[0]
lt = np.where(unique_df.from_edge < unique_df.to_edge)[0]
inds = np.arange(len(unique_df.index))
xlabels = np.concatenate((unique_df.iloc[gt].index.values, unique_df.iloc[same].index.values, unique_df.iloc[lt].index.values))
xlabels_letters = []
letters = {0.0: 'B',1.0: 'C',2.0: 'D', 3.0: 'E', 4.0: 'F', 5.0: 'G'}
for xx in xlabels:
    xlabels_letters.append((letters[xx[0]],letters[xx[1]]))
    
 
width = 0.35

mpl.rcParams['font.size'] = 20
fig,ax = pl.subplots(3,1, tight_layout = True, sharex = True, figsize = [15,10])
for ci,column in enumerate(['diffHs', 'diffTm','diffmwd']):
    ax.ravel('F')[ci].bar(inds[:len(gt)] - width/2, unique_df.iloc[gt][column],width, label = 'Onshore Bar Migration', color = 'blue')
    ax.ravel('F')[ci].bar(inds[len(gt):len(gt)+len(same)] - width/2, unique_df.iloc[same][column],width, label = 'Same', color = 'purple')
    ax.ravel('F')[ci].bar(inds[len(gt)+len(same):] - width/2, unique_df.iloc[lt][column],width, label = 'Offshore Bar Migration', color = 'red')
    
ax[0].legend(loc = 'upper left', fontsize = 12)
ax[0].set_ylabel('meters')
ax[1].set_ylabel('seconds')
ax[2].set_ylabel('degrees')
ax[0].set_title('Difference in Hs')
ax[1].set_title('Difference in Tm')
ax[2].set_title('Difference in MWD')
ax[2].set_xticks(np.arange(len(inds)))
pl.xticks(rotation = 90)
ax[2].set_xticklabels(xlabels_letters)


mpl.rcParams['font.size'] = 15
fig, ax = pl.subplots(1,3, tight_layout = True, sharex = True, figsize = [8,4])
for ci, column in enumerate(['Hs','Tm','mwd']):
    ax.ravel('F')[ci].bar(np.arange(6),unique_df.groupby(by = 'from_edge')[column].mean(), color = 'black')

ax[0].set_ylabel('meters')
ax[1].set_ylabel('seconds')
ax[2].set_ylabel('degrees')
ax[0].set_title('Hs')
ax[1].set_title('Tm')
ax[2].set_title('MWD')
ax[2].plot((0,5),(78,78), '--k')
ax[2].set_xticks(np.arange(6))
ax[2].set_xticklabels(['B','C','D','E','F','G'])
ax[2].set_ylim(50,100)
ax[1].set_ylim(5,10)


fig, ax = pl.subplots(2,1, tight_layout = True, sharex = True)
unique_df.sort_values(by = 'mwd')['diffmwd'].plot(kind = 'bar', ax = ax[1])
unique_df.sort_values(by = 'mwd')['mwd'].plot(kind = 'bar', ax = ax[0])
ax[0].set_ylabel('deg')
ax[1].set_ylabel('deg')
ax[0].set_title('MWD')
ax[1].set_title('Difference in MWD')
ax[1].set_xlabel('State to State')
ax[0].set_ylim((50, 90))



pos_shifted = {} 
Hs_labels = {}
for ei,(c0,c1) in enumerate(alledgelist[2]): 
    Hs_labels[(c0,c1)] = Hslist[ei]

# Trying to draw it with a di-graph    
fig, ax = pl.subplots(1,1, tight_layout = True)
for pp in iter(pos):
    pos_shifted[pp] = np.array([pos[pp][0] - 0.08, pos[pp][1] - 0.08])
nx.draw_networkx_edge_labels(G, pos_shifted, Hs_labels)    












            
###Find number of unique paths
paths = []
for edgelist in alledgelist:
    edgelist_dict = Counter(edgelist)
    path = [ee for ee in edgelist_dict.keys()]
    paths.append(path) 
    
unique_paths = np.unique(paths)
posG = nx.DiGraph()
posG.add_nodes_from(range(1,6))
pos = nx.spring_layout(posG)

fig, ax = pl.subplots(4,4, tight_layout = {'rect':[0,0,1,0.95]}, figsize = (12,12))
for ui,path in enumerate(unique_paths):
    if uu == []:
        continue
    G = nx.DiGraph()
    G.add_nodes_from(range(2,6))
    G.add_edges_from(path)
   
    nx.draw_networkx_nodes(G,pos,ax = ax.ravel(-1)[ui])
    nx.draw_networkx_labels(G,pos,ax = ax.ravel(-1)[ui])
    nx.draw_networkx_edges(G,pos,ax = ax.ravel(-1)[ui])
fig.suptitle('Unique Paths from 1987-1989')
      
#######Now add mean wave direction, and how long it's been sitting in a node

for edgelist_rep in alledgelist:
    if edgelist == []:
        continue
    edgelist_dict = Counter(edgelist)
    edgelist_ = [ee for ee in edgelist_dict.keys()]
    edgelist_count = [cc[1] for cc in edgelist_dict.items()] #This will set the weight of the edges
    G = nx.DiGraph()
    G.add_nodes_from(range(2,6))
    G.add_edges_from(edgelist_)
    pl.figure()
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width = 3) #Now look up how to make some edges thicker based on the count weights

######Plot side-by-side with wave conditions
base = dt.datetime(1986, 10, 1, 6, 0, 0)
date = [base + dt.timedelta(days = xx - 1.25) for xx in dayno]
waveparams_dt = gf.arrayLinear(date[0],date[-1])

