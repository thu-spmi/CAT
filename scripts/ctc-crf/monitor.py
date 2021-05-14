"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)

Directly execute: (in working directory)
    python3 ctc-crf/monitor.py <path to my exp>
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_monitor(task: str, interactive_show=False):

    train_log = f'exp/{task}/ckpt/log_train.csv'
    dev_log = f'exp/{task}/ckpt/log_eval.csv'

    assert os.path.isfile(train_log), f'exp/{task}/ckpt/log_train.csv is not a valid file.'
    assert os.path.isfile(dev_log), f'exp/{task}/ckpt/log_eval.csv is not a valid file.'

    df_train = pd.read_csv(train_log)
    df_eval = pd.read_csv(dev_log)

    _, axes = plt.subplots(2, 2)

    # Time
    batch_per_epoch = len(df_train)//len(df_eval)
    accum_time = df_train['time'].values
    for i in range(1, len(accum_time)):
        accum_time[i] += accum_time[i-1]
        if (i + 1) % batch_per_epoch == 0:
            accum_time[i] += df_eval['time'].values[(i+1)//batch_per_epoch-1]
    accum_time = [x/3600 for x in accum_time]
    axes[0][0].plot(accum_time)
    axes[0][0].grid(ls='--')
    axes[0][0].set_ylabel('Total time / h')

    # learning rate
    axes[0][1].semilogy(df_train['net_lr'].values)
    axes[0][1].grid(ls='--')
    axes[0][1].set_ylabel('learning rate')

    # training loss and moving average
    train_loss = df_train['loss_real'].values
    running_mean = [train_loss[0]]
    for i in range(1, len(train_loss)):
        running_mean.append(running_mean[i-1]*0.9+0.1*train_loss[i])
    min_loss = min(train_loss)
    if min_loss <= 0.:
        axes[1][0].set_yscale('symlog')
        axes[1][0].plot(train_loss, alpha=0.3)
        axes[1][0].plot(running_mean, color='orangered')
    else:
        axes[1][0].semilogy(train_loss, alpha=0.3)
        axes[1][0].semilogy(running_mean, color='orangered')
    axes[1][0].grid(True, which="both", ls='--')
    axes[1][0].set_ylabel('Train set loss')
    axes[1][0].set_xlabel('Step')

    # dev loss
    min_loss = min(df_eval['loss_real'])
    if min_loss <= 0.:
        axes[1][1].set_yscale('symlog')
        axes[1][1].plot([i+1 for i in range(len(df_eval))],
                        df_eval['loss_real'].values)
    else:
        axes[1][1].semilogy([i+1 for i in range(len(df_eval))],
                            df_eval['loss_real'].values)

    axes[1][1].axhline(y=min_loss, ls='--', color='black', alpha=0.5)
    axes[1][1].grid(True, which="both", ls='--')
    axes[1][1].set_ylabel('Dev set loss')
    axes[1][1].set_xlabel('Epoch')

    plt.suptitle(f"\'{task.replace('dev_', '')}\' ({len(df_eval)} epochs)" + '\n' +
                 '{:.2f}s/step or {:.2f}h/epoch'.format(3600*accum_time[-1]/len(accum_time), accum_time[-1]/len(df_eval)) +
                 '\n' + "min loss={:.2f}".format(min_loss))
    plt.tight_layout()
    plt.savefig(f'exp/{task}/monitor.png', dpi=200)
    if interactive_show:
        plt.show()
    else:
        print("Current lr: {:.2e} | Speed: {:.2f} hour / epoch.".format(
            df_train['net_lr'].values[-1], accum_time[-1]/len(df_eval)))
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please input the experiment name.")

    task = sys.argv[1].strip('/').split('/')[-1]
    assert os.path.isdir(
        f'exp/{task}'), f"\'exp/{task}\' is not a valid directory!"

    plot_monitor(task)
    print(f"Saved at exp/{task}/monitor.png")
