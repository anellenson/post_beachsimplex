import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl

'''
Created on May 22, 2019

@author: aellenso
'''
class StateAnl:

    def __init__(self, pred_df):
        '''

        :param pred_df: Predictions data frame with columns corresponding to label, hs, mwd, and tm, and index wsith a datetime
        '''
        self.pred_df = pred_df
        self.classes = ['Ref','LTT','TBR','RBB','LBT', 'NoVis']

    def resTime(self):

        count_table = np.zeros((len(self.classes),len(self.classes)))

        res_time_count = 0
        res_time = [[],[],[],[],[],[]]

        firstday = self.pred_df.index[0]

        for pi,pp in enumerate(self.pred_df.label[:-1]):
            if np.isnan(pp) or np.isnan(self.pred_df.iloc[pi+1].label):
                continue
            from_state = int(pp)
            to_state = int(self.pred_df.iloc[pi + 1].label)

            if to_state == from_state:
                continue
            if to_state != from_state:
                lastday = self.pred_df.index[pi]
                count_table[to_state, from_state] = count_table[to_state, from_state] + 1
                res_time[from_state].append(lastday - firstday)#count inclusively
                firstday = self.pred_df.index[pi]


        res_time_mean = [np.nanmean(x) for x in res_time]
        res_time_std = [np.nanstd(x) for x in res_time]
        trans_to = np.sum(count_table, axis = 1)
        P_occ = (trans_to * res_time_mean) /(np.nansum(np.multiply(trans_to,res_time_mean)))

        return count_table, res_time_mean, res_time_std, trans_to, P_occ

    def waveSwitch(self, var):
        '''

        :param var: String of the column corresponding to the wave variable of interest
        :return:
        '''


        meanvar = np.zeros((len(self.classes),))
        vardiff_table = np.zeros((len(self.classes),len(self.classes)))
        count_table = np.zeros((len(self.classes),len(self.classes)))
        count = np.zeros((len(self.classes),))

        for ii,ll in enumerate(self.pred_df.label[:-1]):
            from_state = int(ll)
            to_state = int(self.pred_df.label.iloc[ii+1])
            var0 = self.pred_df[var].iloc[ii]
            var1 = self.pred_df[var].iloc[ii+1]

            if np.any((var0 == 999) or  (var1 == 999) or np.isnan(var0) or np.isnan(var1)):
                continue
            vardiff = var1 - var0

            meanvar[from_state] += var0
            count[from_state] += 1
            vardiff_table[to_state, from_state] += vardiff
            count_table[to_state, from_state] += 1

        vardiff_table = vardiff_table/count_table
        meanvar = meanvar/count

        return meanvar, vardiff_table




class Plotter():

    def __init__(self):
        self.classes = ['Ref', 'LTT', 'TBR', 'LBT', 'NoVis']


    def plotResTime(res_time_mean, res_time_std, trans_to, P_occ, plotfname, original = False):

        if original:
            classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        else:
            classes = ['LTT', 'TBR', 'RBB', 'LBT']

        mpl.rcParams['font.size'] = 12
        mpl.rcParams['font.weight'] = 'bold'
        fig, ax = pl.subplots(3, 1, sharex = True)
        fig.set_size_inches(5, 8)
        y_pos = np.arange(len(classes))
        ax[0].bar(y_pos, P_occ, align='center', fill = False, linewidth = 4)
        ax[0].set_ylabel('Probability')
        ax[0].set_title('Observations')
        if original:
            ax[0].set_ylim(0, 0.5)
        else:
            ax[0].set_ylim(0, 0.8)

        ax[1].bar(y_pos, trans_to, align='center', fill = False, linewidth = 4)
        ax[1].set_ylabel('Number')
        ax[1].set_title('Transitions')
        if original:
            ax[1].set_ylim(0, 40)
        else:
            ax[1].set_ylim(0, 80)


        ax[2].bar(y_pos, res_time_mean, width = -0.3, align='edge', fill = False, linewidth = 2, label = 'mean')
        ax[2].bar(y_pos, res_time_std, width = 0.3, align = 'edge', fill = False, hatch = '//', label = 'std')
        ax[2].set_xticks(y_pos)
        pl.legend()
        ax[2].set_ylabel('Number of Days')
        ax[2].set_title('Residence Time')
        ax[2].set_ylim(0, 20)
        ax[2].set_xticklabels(classes)


        pl.savefig('plots/{}'.format(plotfname))

    def confusionTable(self, confusion_matrix, f1score, plotfname):
        class_names = ['LTT', 'TBR', 'RBB', 'LBT']

        fig, ax  = pl.subplots(1,1)
        sum =np.sum(confusion_matrix, axis = 1)

        confusion_matrix = confusion_matrix/sum[:,None]
        class_acc = confusion_matrix.diagonal()

        im = ax.pcolor(confusion_matrix, cmap = 'Blues', vmin = 0, vmax = 1)

        for row in np.arange(confusion_matrix.shape[0]):
            for col in np.arange(confusion_matrix.shape[1]):
                if row == col:
                    mean = class_acc[row]
                    if mean >= 0.5:
                        color = 'white'
                    else:
                        color = 'black'
                    ax.text(col +0.35, row+0.65, '{0:.2f}'.format(mean), fontsize = 15, fontweight = 'bold', color = color)


        ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
        ax.yaxis.tick_left()
        ax.set_yticks(range(len(class_names)+1))
        ax.set_xticklabels(class_names, fontsize = 10, weight = 'bold')
        ax.set_yticklabels(class_names, fontsize = 10, weight = 'bold')
        ax.set_ylabel('Truth', fontsize = 12, weight = 'bold')
        ax.set_xlabel('CNN', fontsize = 12, weight = 'bold')
        cb = fig.colorbar(im, ax = ax, ticks = [0, 0.2, 0.4, 0.6, 0.8, 1])
        cb.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
        ax.set_title('F1 score: {}'.format(f1score))

        pl.savefig('plots/{}'.format(plotfname))

    def plot_ts(self, date_CNN, datesH, preds_CNN, preds_CNN_smooth, predsH, plotfname):
        classes = ['LTT', 'TBR', 'RBB', 'LBT']
        mpl.rcParams['font.size'] = 13
        fig, ax = pl.subplots(2,1, tight_layout = True, sharex = True)
        fig.set_size_inches(12,4)
        ax[0].plot(date_CNN, preds_CNN, label = 'CNN', color = 'purple', linewidth = 1.5)
        ax[0].plot(datesH, predsH, label = 'Lippman and Holman', color = 'blue', linewidth = 2)
        ax[0].set_title('Beach State Time Series')
        ax[0].set_ylabel('Beach State')
        ax[0].set_yticks(range(len(classes)))
        ax[0].set_yticklabels(classes)

        ax[1].plot(date_CNN, preds_CNN_smooth, label = 'CNN', color = 'purple', linewidth = 1.5)
        ax[1].plot(datesH, predsH, label = 'L&H', color = 'blue', linewidth = 2)
        ax[1].set_ylabel('Beach State')
        ax[1].set_yticks(range(len(classes)))
        ax[1].set_yticklabels(classes)
        ax[1].set_title('Smoothed Time Series')
        pl.legend()
        pl.savefig('plots/{}'.format(plotfname), dpi = 400)


    def plot_vardiff(self, fig, ax, var, vardiff_table):

        ax.set_title(var + ' Transition Value')
        ax.table(cellText = np.round(vardiff_table,2), rowLabels = self.classes, colLabels = self.classes, loc = 'center' )
        pl.axis('off')
        pl.ylabel('TO')
        ax.text(0.5,0.85,'FROM', fontweight = 'bold')
        ax.text(-0.05,0.5, 'TO', rotation = 90, fontweight = 'bold')
