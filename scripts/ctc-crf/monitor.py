"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)

Directly execute: (in working directory)
    python3 ctc-crf/monitor.py <path to my exp>
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_monitor(log_path: str = None, train_log: str = 'log_train.csv', dev_log: str = 'log_eval.csv', task: str = None, interactive_show=False):
    """Plot the monitor log files

    Args:
        log_path (str, optional): directory of log files
        train_log (str, optional): location/name of training log file
            if `log_path` is not None, would be `log_path/train_log`
        dev_log (str, optional): location/name of dev log file
            if `log_path` is not None, would be `log_path/dev_log`
        task (str, optional): task name (title of ploting)
        interactive_show (bool, optional): specify whether plot in interactive mode. Default False. 
    """

    if log_path is not None:
        if not os.path.isdir(log_path):
            raise NotADirectoryError(f"{log_path} is not a directory.")
            pass
        train_log = os.path.join(log_path, train_log)
        dev_log = os.path.join(log_path, dev_log)

    if not os.path.isfile(train_log):
        raise FileNotFoundError(f"'{train_log}' doesn't exist!")
    if not os.path.isfile(dev_log):
        raise FileNotFoundError(f"'{dev_log}' doesn't exist!")

    if task is None:
        task = train_log.split('/')[-2]

    direc = os.path.dirname(train_log)

    df_train = pd.read_csv(train_log)
    df_eval = pd.read_csv(dev_log)

    _, axes = plt.subplots(2, 2)

    # Time
    ax = axes[0][0]
    batch_per_epoch = len(df_train)//len(df_eval)
    accum_time = df_train['time'].values
    for i in range(1, len(accum_time)):
        accum_time[i] += accum_time[i-1]
        if (i + 1) % batch_per_epoch == 0:
            accum_time[i] += df_eval['time'].values[(i+1)//batch_per_epoch-1]
    accum_time = [x/3600 for x in accum_time]
    ax.plot(accum_time)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    speed = accum_time[-1]/len(df_eval)
    ax.text(0.05, 0.95, "{:.2f}h/epoch".format(speed), transform=ax.transAxes,
            fontsize=8, verticalalignment='top', bbox=props)

    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(ls='--')
    ax.set_ylabel('Total time / h')
    del ax

    # Learning rate
    ax = axes[0][1]
    lrs = df_train['net_lr'].values
    sim_lrs = [0]
    for i in range(1, len(lrs)):
        if lrs[i] != lrs[i-1]:
            sim_lrs += [i-1, i]
    if sim_lrs[-1] < len(lrs) - 1:
        sim_lrs.append(len(lrs)-1)

    if len(sim_lrs) > 1000 or len(sim_lrs) == 1:
        ax.semilogy(lrs)
    else:
        ax.set_yscale('log')
        for i in range(len(sim_lrs)-1):
            _xs = [sim_lrs[i], sim_lrs[i+1]]
            _ys = [lrs[sim_lrs[i]], lrs[sim_lrs[i+1]]]
            if _ys[0] == _ys[1]:
                ax.plot(_xs, _ys, color="C0")
            else:
                ax.plot(_xs, _ys, ls='--', color='black', alpha=0.5)
        del _xs
        del _ys
    del sim_lrs
    del lrs
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(ls='--')
    ax.set_ylabel('learning rate')
    del ax

    # Training loss and moving average
    ax = axes[1][0]
    train_loss = df_train['loss_real'].values
    running_mean = [train_loss[0]]
    for i in range(1, len(train_loss)):
        running_mean.append(running_mean[i-1]*0.9+0.1*train_loss[i])
    min_loss = min(train_loss)
    if min_loss <= 0.:
        # ax.set_yscale('symlog')
        ax.plot(train_loss, alpha=0.3)
        ax.plot(running_mean, color='orangered')
    else:
        ax.semilogy(train_loss, alpha=0.3)
        ax.semilogy(running_mean, color='orangered')

    del train_loss
    del running_mean
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(True, which="both", ls='--')
    ax.set_ylabel('Train set loss')
    ax.set_xlabel("Step")
    del ax

    # Dev loss
    ax = axes[1][1]
    min_loss = min(df_eval['loss_real'])
    if min_loss <= 0.:
        # ax.set_yscale('symlog')
        ax.plot([i+1 for i in range(len(df_eval))],
                df_eval['loss_real'].values)
    else:
        ax.semilogy([i+1 for i in range(len(df_eval))],
                    df_eval['loss_real'].values)

    ax.axhline(y=min_loss, ls='--', color='black', alpha=0.5)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    textstr = '\n'.join([
        "min={:.2f}".format(min_loss),
        f"{len(df_eval)} epoch"
    ])
    speed = accum_time[-1]/len(df_eval)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=props)
    ax.grid(True, which="both", ls='--')
    ax.set_ylabel('Dev set loss')
    ax.set_xlabel('Epoch')
    del ax

    # Global settings
    titles = [
        task.replace('dev_', '')
    ]
    plt.suptitle('\n'.join(titles))
    plt.tight_layout()
    plt.savefig(os.path.join(direc, 'monitor.png'), dpi=300)
    if interactive_show:
        plt.show()
    else:
        print("Current lr: {:.2e} | Speed: {:.2f} hour / epoch.".format(
            df_train['net_lr'].values[-1], speed))
    plt.close()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=str, help="Location of log files.")
    parser.add_argument("--title", type=str, default=None,
                        help="Configure the plotting title.")
    args = parser.parse_args()

    try:
        plot_monitor(args.log, task=args.title)
    except FileNotFoundError:
        print(
            "Log files not found in {0}, try to find {0}/ckpt".format(args.log))
        plot_monitor(os.path.join(args.log, 'ckpt'), task=args.title)
