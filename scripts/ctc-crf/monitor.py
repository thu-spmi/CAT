"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)

Directly execute: (in working directory)
    python3 ctc-crf/monitor.py <path to my checkpoint>
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Union, Tuple

import torch


def draw_time(ax: plt.Axes, scalars: Union[np.array, list], num_steps: int, num_epochs: int, eval_time: Union[np.array, list], prop_box=True):
    batch_per_epoch = num_steps//num_epochs
    accum_time = scalars[:]
    for i in range(1, len(accum_time)):
        accum_time[i] += accum_time[i-1]
        if (i + 1) % batch_per_epoch == 0:
            accum_time[i] += eval_time[(i+1)//batch_per_epoch-1]
    del batch_per_epoch
    accum_time = [x/3600 for x in accum_time]
    ax.plot(accum_time)

    if prop_box:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        speed = accum_time[-1]/num_epochs
        if speed < 1.:
            speed = speed * 60
            if speed < 1.:
                speed = speed * 60
                timestr = "{:.0f}sec/epoch".format(speed)
            else:
                timestr = "{:.1f}min/epoch".format(speed)
        else:
            timestr = "{:.2f}h/epoch".format(speed)
        ax.text(0.05, 0.95, timestr, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', bbox=props)

    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(ls='--')
    ax.set_ylabel('Total time / h')
    return ax


def draw_tr_loss(ax: plt.Axes, scalars: Union[np.array, list], smooth_value: float = 0.9):
    assert smooth_value >= 0. and smooth_value < 1.
    running_mean = [scalars[0]]
    res_smooth = 1 - smooth_value
    for i in range(1, len(scalars)):
        running_mean.append(
            running_mean[i-1]*smooth_value+res_smooth*scalars[i])

    min_loss = min(running_mean)
    if min_loss <= 0.:
        ax.plot(running_mean)
    else:
        ax.semilogy(running_mean)

    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(True, ls='--')
    ax.set_ylabel('Training loss')
    ax.set_xlabel("Step")
    return ax


def draw_dev_loss(ax: plt.Axes, scalars: Union[np.array, list], num_epochs: int, prop_box=True):

    min_loss = min(scalars)
    if min_loss <= 0.:
        ax.plot([i+1 for i in range(num_epochs)], scalars)
    else:
        ax.semilogy([i+1 for i in range(num_epochs)], scalars)

    # ax.axhline(y=min_loss, ls='--', color='black', alpha=0.5)
    if prop_box:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        textstr = '\n'.join([
            "min={:.2f}".format(min_loss),
            f"{num_epochs} epoch"
        ])
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=props)
    ax.grid(True, ls='--')
    ax.set_ylabel('Dev loss')
    ax.set_xlabel('Epoch')
    return ax


def draw_lr(ax: plt.Axes, scalars: Union[np.array, list]):

    ax.semilogy(scalars)

    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(ls='--')
    ax.set_ylabel('learning rate')
    return ax


def read_from_check(check: OrderedDict) -> Tuple[np.array, np.array, int, int]:
    '''
    check: OrderedDict({
            'log_train': ['epoch,loss,loss_real,net_lr,time'],
            'log_eval': ['loss_real,time']
        })
    '''

    df_train = np.array(check['log_train'][1:])
    df_train = {
        'loss': df_train[:, 1],
        'loss_real': df_train[:, 2],
        'lr': df_train[:, 3],
        'time': df_train[:, 4],
    }
    df_eval = np.array(check['log_eval'][1:])
    df_eval = {
        'loss': df_eval[:, 0],
        'time': df_eval[:, 1]
    }
    num_batches = df_train['loss'].shape[0]
    num_epochs = df_eval['loss'].shape[0]
    return df_train, df_eval, num_batches, num_epochs


def plot_monitor(log_path: str = None, log: OrderedDict = None, title: str = None, interactive_show=False, o_path: str = None):
    """Plot the monitor log files

    Args:
        log_path (str, optional): directory of log files
        log (OrderedDict, optional): log files
        title (str, optional): title name (title of ploting)
        interactive_show (bool, optional): specify whether plot in interactive mode. Default False. 
    """

    if log is None:
        # read from file
        if not os.path.isfile(log_path):
            raise FileNotFoundError(f"'{log_path}' doesn't exist!")

        log = torch.load(log_path, map_location='cpu')['log']

    if title is None:
        title = ' '

    df_train, df_eval, num_batches, num_epochs = read_from_check(log)

    _, axes = plt.subplots(2, 2)

    # Time
    draw_time(axes[0][0], df_train['time'],
              num_batches, num_epochs, df_eval['time'])

    # Learning rate
    draw_lr(axes[0][1], df_train['lr'])

    # Training loss and moving average
    draw_tr_loss(axes[1][0], df_train['loss_real'])

    # Dev loss
    draw_dev_loss(axes[1][1], df_eval['loss'], num_epochs)

    # Global settings

    plt.suptitle(title)
    plt.tight_layout()
    if interactive_show:
        plt.show()
    else:
        if o_path is None:
            if log_path is None:
                direc = './'
            elif os.path.isfile(log_path):
                direc = os.path.dirname(log_path)
            elif os.path.isdir(log_path):
                direc = log_path
            else:
                raise ValueError(
                    f"log_path={log_path} is neither a directory nor a file.")

            outpath = os.path.join(direc, 'monitor.png')
        else:
            if os.path.isdir(o_path):
                outpath = os.path.join(o_path, 'monitor.png')
            else:
                assert os.path.isdir(os.path.dirname(o_path))
                outpath = o_path
        plt.savefig(outpath, dpi=300)
        print(f"> Monitor figure saved at {outpath}")
    plt.close()


def cmp(check0: str, check1: str, legends: Union[Tuple[str, str], None] = None, title: str = ' ', o_path=None):
    assert os.path.isfile(check0), f"{check0} is not a file."
    assert os.path.isfile(check1), f"{check1} is not a file."

    check0 = torch.load(check0, map_location='cpu')['log']  # type: OrderedDict
    check1 = torch.load(check1, map_location='cpu')['log']  # type: OrderedDict

    _, axes = plt.subplots(2, 2)

    if legends is None:
        legends = ['1', '2']

    for clog in [check0, check1]:

        df_train, df_eval, num_batches, num_epochs = read_from_check(clog)

        # Time
        draw_time(axes[0][0], df_train['time'],
                  num_batches, num_epochs, df_eval['time'], prop_box=False)

        # Learning rate
        draw_lr(axes[0][1], df_train['lr'])

        # Training loss and moving average
        draw_tr_loss(axes[1][0], df_train['loss_real'])

        # Dev loss
        draw_dev_loss(axes[1][1], df_eval['loss'], num_epochs, prop_box=False)

    axes[0][1].legend(legends, fontsize=8)
    plt.suptitle(title)
    plt.tight_layout()

    legends = [x.replace(' ', '_') for x in legends]
    if o_path is None:
        outpath = os.path.join('.', 'compare-{}-{}.png'.format(*legends))
    else:
        if os.path.isdir(o_path):
            outpath = os.path.join(
                o_path, 'compare-{}-{}.png'.format(*legends))
        else:
            assert os.path.isdir(os.path.dirname(o_path))
            outpath = o_path
    plt.savefig(outpath, dpi=300)
    print(f"> Comparison figure saved at {outpath}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=str, help="Location of log files.")
    parser.add_argument("--title", type=str, default=None,
                        help="Configure the plotting title.")
    parser.add_argument("--cmp", type=str, default=None,
                        help="Same format as log, compared one.")
    parser.add_argument("--cmplegend", type=str, default='1-2',
                        help="Legend for two comparing figures, split by '-'. Default: 1-2")
    parser.add_argument("-o", type=str, default=None, dest="o_path",
                        help="Output figure path.")
    args = parser.parse_args()

    if args.cmp is None:
        plot_monitor(args.log, title=args.title, o_path=args.o_path)
    else:
        legends = args.cmplegend.split('-')
        assert len(legends) == 2
        cmp(args.log, args.cmp, legends=legends,
            title=args.title, o_path=args.o_path)
