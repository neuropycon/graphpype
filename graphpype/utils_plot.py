import numpy as np


def plot_cormat(plot_file, cor_mat, list_labels=[], fix_full_range=[],
                label_size=2, cmap='rainbow'):
    """plot correlation matrix"""
    import matplotlib.pyplot as plt

    fig1 = plt.figure(frameon=False)
    ax = fig1.add_subplot(1, 1, 1)

    if len(fix_full_range):
        im = ax.matshow(cor_mat, vmin=fix_full_range[0],
                        vmax=fix_full_range[1], interpolation="none")
    else:
        im = ax.matshow(cor_mat, interpolation="none")

    [i.set_visible(False) for i in ax.spines.values()]
    im.set_cmap(cmap)

    # add labels
    if len(list_labels):
        assert len(list_labels) == cor_mat.shape[0], "Error number of labels \
            {} and matrix shape {}".format(len(list_labels), cor_mat.shape[0])

        plt.xticks(list(range(len(list_labels))), list_labels,
                   rotation='vertical', fontsize=label_size)
        plt.yticks(list(range(len(list_labels))), list_labels,
                   fontsize=label_size)
        plt.subplots_adjust(top=0.8)

    # ticks
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                    left=False, right=False)
    # colorbar
    fig1.colorbar(im)
    fig1.savefig(plot_file)
    plt.close(fig1)


def plot_ranged_cormat(plot_file, cor_mat, list_labels=[],
                       fix_full_range=[-1.0, 1.0], label_size=2,
                       cmap='nipy_spectral'):
    """plot ranged correlation matrix"""
    # kept for sake of compatibility with previous version
    plot_cormat(plot_file=plot_file, cor_mat=cor_mat, list_labels=list_labels,
                fix_full_range=fix_full_range, label_size=label_size,
                cmap=cmap)


def plot_int_mat(plot_file, cor_mat, list_labels=[], fix_full_range=[-4, 4],
                 label_size=2, cmap='jet'):
    """plot ranged correlation matrix"""
    # kept for sake of compatibility with previous version
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap, len(
        np.unique(np.arange(fix_full_range[0], fix_full_range[1]+1))))

    plot_cormat(plot_file=plot_file, cor_mat=cor_mat, list_labels=list_labels,
                fix_full_range=fix_full_range, label_size=label_size,
                cmap=cmap)


def plot_hist(plot_hist_file, data, nb_bins=100):
    """ plot histogramms """
    import matplotlib.pyplot as plt
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    y, x = np.histogram(data, bins=nb_bins)
    ax.plot(x[:-1], y)
    fig2.savefig(plot_hist_file)
    plt.close(fig2)


def plot_colorbar(plot_colorbar_file, colors):
    """plot colorbar"""
    # TODO test
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    fig2 = plt.figure(figsize=(8, 3))
    ax2 = fig2.add_axes([0.05, 0.475, 0.9, 0.15])

    cmap = mpl.colors.ListedColormap(colors)
    cmap.set_over('0.25')
    cmap.set_under('0.75')

    bounds = list(range(len(colors)))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
                                    boundaries=bounds, spacing='proportional',
                                    orientation='horizontal')
    cb2.set_label('Discrete intervals, some other units')

    fig2.savefig(plot_colorbar_file)
    plt.close(fig2)


def plot_signals(plot_signals_file, signals_matrix, colors=[],
                 ylim=[], labels=[], add_zero_line=False):
    """ plot signals"""
    import matplotlib.pyplot as plt

    assert len(signals_matrix.shape) <= 2, ("Error, signals_matrix should be \
        at most 2D")
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)

    if len(ylim) == 2:
        ax.set_ylim(ylim[0], ylim[1])

    if len(signals_matrix.shape) == 1:
        ax.plot(list(range(signals_matrix.shape[0])), signals_matrix[:])

    elif len(signals_matrix.shape) == 2:

        nb_signals = signals_matrix.shape[0]
        nb_timings = signals_matrix.shape[1]

        signals_matrix = np.transpose(signals_matrix)
        lines = ax.plot(list(range(signals_matrix.shape[0])), signals_matrix)

        # adding color if available
        if len(colors) == nb_signals:
            [line.set_color(color) for color, line in zip(colors, lines)]
        elif len(colors) == 1:
            [line.set_color(colors[0]) for line in lines]

        # adding labels in available
        if len(labels) == nb_signals:
            [line.set_label(label) for label, line in zip(labels, lines)]
            ax.legend(handles=lines, loc=0, prop={'size': 8})

        # adding zero line
        if add_zero_line:
            ax.plot(list(range(nb_timings)), [0.0]*nb_timings,
                    color='black', linestyle='--')

    fig2.savefig(plot_signals_file)
    plt.close(fig2)


def plot_sep_signals(plot_signals_file, signals_matrix, colors=[], labels=[],
                     range_signal=1):
    """Plotting signals separately"""
    # keeping for sake of compatibility
    assert len(signals_matrix.shape) == 2, ("No interest to use \
        plot_sep_signals, use plot_signals instead")

    range_signal *= (np.amax(signals_matrix) - np.amin(signals_matrix))

    nb_signals = signals_matrix.shape[0]
    nb_timings = signals_matrix.shape[1]

    bias_matrix = np.array([[i*range_signal]*nb_timings
                           for i in range(nb_signals)])
    signals_matrix = signals_matrix+bias_matrix
    ymin = np.amin(signals_matrix)-2
    ymax = np.amax(signals_matrix)+2

    plot_signals(plot_signals_file, signals_matrix, ylim=[ymin, ymax],
                 colors=colors, labels=labels)
