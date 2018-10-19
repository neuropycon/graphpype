"""test over plots (with matplotlib)"""

from graphpype.utils_plot import (plot_cormat, plot_ranged_cormat,
                                  plot_int_mat, plot_hist, plot_signals,
                                  plot_sep_signals)

import os
import shutil
import numpy as np

import matplotlib
matplotlib.use('Agg')

tmp_dir = "/tmp/test_graphpype"
nb_nodes = 10
cor_mat = np.random.rand(nb_nodes, nb_nodes)
int_mat = np.random.randint(low=-4, high=4, size=(nb_nodes, nb_nodes))
nb_timepoints = 1000
signals_matrix = np.random.rand(nb_nodes, nb_timepoints)

if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)

os.makedirs(tmp_dir)


def test_plot_cormat():
    """test plot_cormat"""
    plot_file = os.path.join(tmp_dir, "test_plot_cormat.png")
    plot_cormat(plot_file, cor_mat=cor_mat)

    assert os.path.exists(plot_file)


def test_plot_ranged_cormat():
    """test plot_ranged_cormat"""
    plot_file = os.path.join(tmp_dir, "test_plot_ranged_cormat.png")
    plot_ranged_cormat(plot_file, cor_mat=cor_mat)

    assert os.path.exists(plot_file)


def test_plot_int_mat():
    """test plot_int_mat"""
    plot_file = os.path.join(tmp_dir, "test_plot_int_mat.png")
    plot_int_mat(plot_file, cor_mat=int_mat)

    assert os.path.exists(plot_file)


def test_plot_hist():
    """test plot_hist"""
    plot_file = os.path.join(tmp_dir, "test_plot_hist.png")
    plot_hist(plot_file, data=cor_mat)

    assert os.path.exists(plot_file)


def test_plot_signals():
    """test plot_signals"""
    plot_file = os.path.join(tmp_dir, "test_plot_signals.png")
    plot_signals(plot_file, signals_matrix)

    assert os.path.exists(plot_file)

    # test one color
    one_col_file = os.path.join(tmp_dir, "test_plot_one_col.png")
    plot_signals(one_col_file, signals_matrix, colors=['red'])
    assert os.path.exists(one_col_file)

    # test multiple colors
    mult_col_file = os.path.join(tmp_dir, "test_plot_mult_col.png")
    colors = ['blue', 'red']*int(nb_nodes/2)
    plot_signals(mult_col_file, signals_matrix, colors=colors)

    assert os.path.exists(mult_col_file)

    # test_label
    labels_file = os.path.join(tmp_dir, "test_plot_labels.png")
    labels = [str(i) for i in range(nb_nodes)]
    plot_signals(labels_file, signals_matrix, labels=labels)

    assert os.path.exists(labels_file)


def test_plot_sep_signals():
    """test plot_sep_signals"""
    plot_file = os.path.join(tmp_dir, "test_plot_sep_signals.eps")
    plot_sep_signals(plot_file, signals_matrix, range_signal=10)

    assert os.path.exists(plot_file)
