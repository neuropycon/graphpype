
import matplotlib.pyplot as plt
import pylab as pl

import numpy as np


################################################################### matrices #######################################################

def plot_cormat(plot_file, cor_mat, list_labels=[], label_size=2):

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)

    im = ax.matshow(cor_mat, interpolation="none")
    # plt.axis('off')

    [i.set_visible(False) for i in ax.spines.values()]

    # im.set_cmap('binary')
    im.set_cmap('rainbow')
    # add labels

    if len(list_labels) != 0:

        if len(list_labels) == cor_mat.shape[0]:

            plt.xticks(list(range(len(list_labels))), list_labels,
                       rotation='vertical', fontsize=label_size)
            plt.yticks(list(range(len(list_labels))),
                       list_labels, fontsize=label_size)

            plt.subplots_adjust(top=0.8)
        else:
            print("Warning in utils_plot.plot_cormat, incompatible number of labels %d and matrix shape %d" % (
                len(list_labels), cor_mat.shape[0]))

    plt.tick_params(axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    left=False,
                    right=False
                    )
    # ax.set_xticklabels(['']+labels)
    #plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=3)

    fig1.colorbar(im)

    fig1.savefig(plot_file)

    plt.close(fig1)
    # fig1.close()
    del fig1


def plot_ranged_cormat(plot_file, cor_mat, list_labels=[], fix_full_range=[-1.0, 1.0], label_size=2):

    fig1 = plt.figure(frameon=False)
    ax = fig1.add_subplot(1, 1, 1)

    im = ax.matshow(
        cor_mat, vmin=fix_full_range[0], vmax=fix_full_range[1], interpolation="none")
    # plt.axis('off')

    [i.set_visible(False) for i in ax.spines.values()]

    # im.set_cmap('binary')
    im.set_cmap('spectral')
    # im.set_cmap('jet')

    # add labels

    if len(list_labels) != 0:

        if len(list_labels) == cor_mat.shape[0]:

            plt.xticks(list(range(len(list_labels))), list_labels,
                       rotation='vertical', fontsize=label_size)
            plt.yticks(list(range(len(list_labels))),
                       list_labels, fontsize=label_size)

            plt.subplots_adjust(top=0.8)
        else:
            print("Warning in utils_plot.plot_cormat, incompatible number of labels %d and matrix shape %d" % (
                len(list_labels), cor_mat.shape[0]))

    plt.tick_params(axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    left='off',
                    right='off'
                    )
    # ax.set_xticklabels(['']+labels)
    #plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=3)

    fig1.colorbar(im)

    fig1.savefig(plot_file)

    plt.close(fig1)
    # fig1.close()
    del fig1


def plot_int_mat(plot_file, cor_mat, list_labels=[], fix_full_range=[-4, 4], label_size=2):

    fig1 = plt.figure(frameon=False)
    ax = fig1.add_subplot(1, 1, 1)

    #cmap= plt.get_cmap('spectral',9)

    cmap = plt.get_cmap('jet', len(
        np.unique(np.arange(fix_full_range[0], fix_full_range[1]+1))))

    print(len(np.unique(np.arange(fix_full_range[0], fix_full_range[1]+1))))
    #cmap_vals = cmap(np.linspace(0.2,0.8,9))

    im = ax.matshow(
        cor_mat, vmin=fix_full_range[0], vmax=fix_full_range[1], interpolation="none", cmap=cmap)
    # plt.axis('off')

    [i.set_visible(False) for i in ax.spines.values()]

    # im.set_cmap('binary')
    # im.set_cmap('spectral')

    # add labels

    if len(list_labels) != 0:

        if len(list_labels) == cor_mat.shape[0]:

            plt.xticks(list(range(len(list_labels))), list_labels,
                       rotation='vertical', fontsize=label_size)
            plt.yticks(list(range(len(list_labels))),
                       list_labels, fontsize=label_size)

            plt.subplots_adjust(top=0.8)
        else:
            print("Warning in utils_plot.plot_cormat, incompatible number of labels %d and matrix shape %d" % (
                len(list_labels), cor_mat.shape[0]))

    plt.tick_params(axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    left='off',
                    right='off'
                    )
    # ax.set_xticklabels(['']+labels)
    #plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=3)

    #fig1.colorbar(im, ticks = range(-4,5))
    fig1.colorbar(im, ticks=list(range(-4, 5)))

    fig1.savefig(plot_file)

    plt.close(fig1)
    # fig1.close()
    del fig1

    return

################################################################### histogramms #######################################################


def plot_hist(plot_hist_file, data, nb_bins=100):

    #fig2 = figure.Figure()
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    y, x = np.histogram(data, bins=nb_bins)
    ax.plot(x[:-1], y)
    #ax.bar(x[:-1],y, width = y[1]-y[0])
    fig2.savefig(plot_hist_file)

    plt.close(fig2)
    # fig2.close()
    del fig2


################################################################### color bar only #######################################################
import matplotlib as mpl


def plot_colorbar(plot_colorbar_file, colors):

    #fig2 = figure.Figure()
    fig2 = plt.figure(figsize=(8, 3))
    ax2 = fig2.add_axes([0.05, 0.475, 0.9, 0.15])
    #ax = fig2.add_subplot(1,1,1)

    cmap = mpl.colors.ListedColormap(colors)

    cmap.set_over('0.25')
    cmap.set_under('0.75')

    bounds = list(range(len(colors)))

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    print(bounds)

    print(norm)

    cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                    norm=norm,
                                    # to use 'extend', you must
                                    # specify two extra boundaries:
                                    boundaries=bounds,
                                    # extend='both',
                                    # ticks=bounds, # optional
                                    spacing='proportional',
                                    orientation='horizontal')
    cb2.set_label('Discrete intervals, some other units')

    fig2.savefig(plot_colorbar_file)

    plt.close(fig2)
    # fig2.close()
    del fig2

################################################################### signals #######################################################


def plot_signals(plot_signals_file, signals_matrix, colors=[], ylim=[], labels=[], add_zero_line=False):

    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)

    # print signals_matrix.shape[0]
    # print len(signals_matrix.shape)

    if len(ylim) != 0:

        ax.set_ylim(ylim[0], ylim[1])

    if len(signals_matrix.shape) == 1:
        ax.plot(list(range(signals_matrix.shape[0])), signals_matrix[:])

    else:
        for i in range(signals_matrix.shape[0]):

            if len(colors) == signals_matrix.shape[0]:

                if len(labels) == 0 or len(labels) != len(colors):
                    ax.plot(
                        list(range(signals_matrix.shape[1])), signals_matrix[i, :], colors[i])
                else:
                    print("adding label: " + labels[i])
                    ax.plot(list(range(
                        signals_matrix.shape[1])), signals_matrix[i, :], colors[i], label=labels[i])

            elif len(colors) == 1:
                ax.plot(
                    list(range(signals_matrix.shape[1])), signals_matrix[i, :], colors[0])
            else:
                if len(labels) == signals_matrix.shape[0]:
                    print("adding label: " + labels[i])
                    ax.plot(
                        list(range(signals_matrix.shape[1])), signals_matrix[i, :], label=labels[i])
                else:
                    ax.plot(
                        list(range(signals_matrix.shape[1])), signals_matrix[i, :])

    if add_zero_line:
        ax.plot(list(range(signals_matrix.shape[1])), [
                0.0]*signals_matrix.shape[1], color='black', linestyle='--')

    if signals_matrix.shape[0] == len(labels):
        print("adding legend")
        ax.legend(loc=0, prop={'size': 8})

    # ax.plot(,signals_matrix)
    fig2.savefig(plot_signals_file)

    plt.close(fig2)
    # fig2.close()
    del fig2


def plot_sep_signals(plot_signals_file, signals_matrix, colors=[], range_signal=-1):

    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)

    if range_signal == -1:

        range_signal = np.amax(signals_matrix) - np.amin(signals_matrix)

    # print range_signal

    # print signals_matrix.shape

    if len(colors) == signals_matrix.shape[0]:

        for i in range(signals_matrix.shape[0]):
            print(range_signal*i)
            ax.plot(signals_matrix[i, :] + range_signal*i, colors[i])

    else:

        for i in range(signals_matrix.shape[0]):
            # print range_signal*i
            ax.plot(signals_matrix[i, :] + range_signal*i)

    # for i in range(signals_matrix.shape[0]):
        # print range_signal*i

        #sep_signals_matrix[i,:] = signals_matrix[i,:] + range_signal*i

    # print sep_signals_matrix

    # ax.plot(range(signals_matrix.shape[1]),sep_signals_matrix)

    x1, x2, y1, y2 = ax.axis()
    ax.axis((x1, x2, -2.0, y2))

    fig2.savefig(plot_signals_file)

    plt.close(fig2)
    # fig2.close()
    del fig2

    return
