import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="recognition argument")
parser.add_argument('--csv-file', default='result.csv', type=str,
                    help='CSV file to save the results of epochs')

parser.add_argument('--figure-file', default='result.png', type=str,
                    help='figure image to plot the results of epochs')


def plot_train_figure(csv_file_name, figure_file_name):
    x = []
    y_time = []
    y_lr = []
    y_cv = []

    with open(csv_file_name, 'r') as csv_file:
        plots = csv.reader(csv_file, delimiter=',')
        next(plots)
        for row in plots:
            x.append(int(row[0]))
            y_time.append(float(row[1]))
            y_lr.append(float(row[2]))
            y_cv.append(float(row[3]))
    csv_file.close()

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1,
                        wspace=0.001)
    plt.figure(figsize=(7, 9))
    #   plt.text(a, b, b, ha='center', va='bottom', fontsize=11)
    plt.subplot(311)
    plt.bar(x, y_time, label='times of epoch', width=0.1)
    max_time = max(y_time)
    plt.ylim((0, max_time))
    y_ticks = np.arange(0, max_time*1.2, max_time/10)
    plt.yticks(y_ticks)
    plt.xlabel('epoch')
    plt.ylabel('minutes')
    plt.grid(True, linestyle="--", color="gray", linewidth="0.5", axis='both')
    plt.title('time of epochs')

    plt.subplot(312)
    plt.plot(x, y_lr, label='learning rate', c='r')
    plt.scatter(x, y_lr, marker='*', c='red')
    max_lr = max(y_lr)
    plt.ylim((0, max_lr*1.2))
    y_ticks = np.arange(0, max_lr*1.2, max_lr/10)
    plt.yticks(y_ticks)
    plt.ylabel(r"learning rate")
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.grid(True, linestyle="--", color="gray", linewidth="0.5", axis='both')

    plt.subplot(313)
    plt.plot(x, y_cv, label='held out loss')
    plt.scatter(x, y_cv, marker='*', c='red')
    max_cv = max(y_cv)
    min_cv = min(y_cv)
    plt.ylim((min_cv, max_cv*1.2))
    y_ticks = np.arange(min_cv, max_cv*1.2, (max_cv*1.2-min_cv)/10)
    plt.yticks(y_ticks)
    plt.ylabel('held-out loss')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.grid(True, linestyle="--", color="gray", linewidth="0.5", axis='both')

    plt.savefig(figure_file_name, dpi=100)


def main():
    args = parser.parse_args()
    plot_train_figure(args.csv_file, args.figure_file)


if __name__ == "__main__":
    main()
