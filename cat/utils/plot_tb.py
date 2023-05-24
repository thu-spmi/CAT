# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""Plot tensorboard scalars to images.

Usage:
    in working directory:
    python3 utils/plot_tb.py <path to the logfile>
"""

# TODO (huahuan):
#     EventAccumulator.Reload() is very slow. Maybe we should switch to faster backend.
#     Have a look at https://github.com/tensorflow/tensorboard/issues/4354


import os
import math
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import *

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

FIGURE_HEIGHT = 2.8
FIGURE_WIDTH = 2.7
PLOT_DPI = 150

PathLikedObj = str


def draw_time(ax: plt.Axes, scalar_events: list, prop_box=True):
    steps = [x.step for x in scalar_events]
    d_time = np.asarray([x.wall_time for x in scalar_events])
    d_time -= d_time[0]
    d_time /= 3600.0

    ax.plot(steps, d_time)
    title = "tot. time (hour)"
    speed = d_time[-1] / (steps[-1] - steps[0])

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if prop_box:
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        if speed < 1.0:
            speed = speed * 60
            if speed < 1.0:
                speed = speed * 60
                timestr = f"{speed:.1f} sec/step"
                if speed < 1.0:
                    speed = 1 / speed
                    timestr = f"{speed:.1f} step/sec"
            else:
                timestr = f"{speed:.1f} min/step"
        else:
            timestr = f"{speed:.2f} h/step"
        ax.text(
            0.95,
            0.05,
            timestr,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

    ax.grid(ls="--")
    ax.set_title(title, fontsize=10)
    return ax


def draw_loss(
    ax: plt.Axes,
    scalar_events: list,
    smooth_value: float = 0.9,
    title: str = "loss",
    prop_box=False,
    ax_yformatter=None,
):
    steps = [x.step for x in scalar_events]
    scalars = np.asarray([x.value for x in scalar_events])

    assert smooth_value >= 0.0 and smooth_value < 1.0

    if smooth_value > 0.0:
        res_smooth = 1 - smooth_value
        running_mean = np.zeros_like(scalars)
        running_mean[0] = scalars[0]
        for i in range(1, len(scalars)):
            running_mean[i] = (
                smooth_value * running_mean[i - 1] + res_smooth * scalars[i]
            )
        alpha = 0.25
    else:
        alpha = 1.0

    min_loss = min(scalars)
    if min_loss <= 0.0 or (max(scalars) / min_loss < 10.0):
        ax.plot(steps, scalars, alpha=alpha)
        if smooth_value > 0.0:
            ax.plot(steps, running_mean, color=ax.get_lines()[-1].get_color())
        if ax_yformatter is None:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    else:
        ax.semilogy(steps, scalars, alpha=alpha)
        if smooth_value > 0.0:
            ax.semilogy(steps, running_mean, color=ax.get_lines()[-1].get_color())

    if prop_box:
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        textstr = "\n".join(["min={:.2f}".format(min_loss)])
        ax.text(
            0.95,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.grid(True, ls="--", which="both")
    ax.set_title(title, fontsize=10)
    return ax


def draw_lr(ax: plt.Axes, scalar_events: list):
    ax.plot([x.step for x in scalar_events], [x.value for x in scalar_events])
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.grid(ls="--", which="both")
    ax.set_title("learning rate", fontsize=10)
    return ax


def draw_any(ax: plt.Axes, scalar_events: list, _name: str = ""):
    return draw_loss(
        ax,
        scalar_events,
        smooth_value=0.9,
        title=_name,
        prop_box=False,
        ax_yformatter=ax.yaxis.get_major_formatter(),
    )


def plot_tb(
    tbevent: PathLikedObj,
    o_path: PathLikedObj = None,
    title: str = None,
    interactive_show=False,
):
    """Plot the summary log files

    Args:
        tbevent (str): path to the event file
        title (str, optional): title name (title of ploting)
        interactive_show (bool, optional): specify whether plot in interactive mode. Default False.
    """
    accumulator = EventAccumulator(tbevent).Reload()

    anno_scalars = accumulator.Tags()["scalars"]
    n_plottings = len(anno_scalars)
    assert n_plottings > 0, f"no scalar found in {tbevent}"

    nr = nc = int(math.sqrt(n_plottings))
    while nr * nc < n_plottings:
        nc += 1

    _, axes = plt.subplots(
        nr,
        nc,
        squeeze=False,
        figsize=(FIGURE_WIDTH * nc, FIGURE_HEIGHT * nr),
        constrained_layout=True,
    )
    metas = []
    for i in range(nr * nc):
        r, c = (i // nc), (i % nc)
        if i >= n_plottings:
            # rm empty plotting
            axes[r][c].set_axis_off()
            continue

        scalars = list(accumulator.Scalars(anno_scalars[i]))
        metas.append((anno_scalars[i], scalars))
        if anno_scalars[i].startswith("loss"):
            if "dev" in anno_scalars[i]:
                draw_loss(
                    axes[r][c],
                    scalars,
                    smooth_value=0.0,
                    title=anno_scalars[i],
                    prop_box=True,
                )
            else:
                draw_loss(axes[r][c], scalars, title=anno_scalars[i])
        elif anno_scalars[i] == "lr":
            draw_lr(axes[r][c], scalars)
        else:
            draw_any(axes[r][c], scalars, anno_scalars[i])

    metas = [(tbevent, metas)]
    if title is None:
        title = " "
    # Global settings
    plt.suptitle(title)
    if interactive_show:
        outpath = None
        plt.show()
    else:
        outpath = "./tb_plot.jpg" if o_path is None else o_path
        plt.savefig(outpath, dpi=PLOT_DPI, facecolor="w")
    plt.close()
    return outpath, metas


def cmp(
    tbevents: List[PathLikedObj],
    legends: Union[List[str], None] = None,
    title: str = " ",
    o_path: PathLikedObj = None,
):
    accumulators = [EventAccumulator(_event).Reload() for _event in tbevents]

    annotations = sorted(
        set(sum((accum.Tags()["scalars"] for accum in accumulators), []))
    )
    n_plottings = len(annotations)
    assert n_plottings > 0, f"no scalar found."

    nr = nc = int(math.sqrt(n_plottings))
    while nr * nc < n_plottings:
        nc += 1

    _, axes = plt.subplots(
        nr,
        nc,
        squeeze=False,
        figsize=(FIGURE_WIDTH * nc, FIGURE_HEIGHT * nr),
        constrained_layout=True,
    )

    yaxis_formatter = {}
    metas = []
    for n, _accum in enumerate(accumulators):
        curplot = []
        for i in range(nr * nc):
            r, c = (i // nc), (i % nc)
            if i >= n_plottings:
                # rm empty plotting
                axes[r][c].set_axis_off()
                continue

            if annotations[i] not in _accum.Tags()["scalars"]:
                continue

            scalars = list(_accum.Scalars(annotations[i]))
            curplot.append((annotations[i], scalars))
            if annotations[i].startswith("loss"):
                if i not in yaxis_formatter:
                    yaxis_formatter[i] = None
                if "dev" in annotations[i]:
                    draw_loss(
                        axes[r][c],
                        scalars,
                        ax_yformatter=yaxis_formatter[i],
                        smooth_value=0.0,
                        title=annotations[i],
                    )
                else:
                    draw_loss(
                        axes[r][c],
                        scalars,
                        ax_yformatter=yaxis_formatter[i],
                        title=annotations[i],
                    )
                if n == 0:
                    yaxis_formatter[i] = axes[r][c].yaxis.get_major_formatter()
            elif annotations[i] == "lr":
                draw_lr(axes[r][c], scalars)
            else:
                draw_any(axes[r][c], scalars, annotations[i])
        metas.append((tbevents[n], curplot))

    if legends is not None:
        axes[0][0].legend(legends, fontsize=8)
    plt.suptitle(title)

    outpath = "./cmp.jpg" if o_path is None else o_path
    plt.savefig(outpath, dpi=PLOT_DPI, facecolor="w")
    plt.close()
    return outpath, metas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log", type=str, nargs="+", help="Path to the location of log file(s)."
    )
    parser.add_argument(
        "--title", type=str, default=None, help="Configure the plotting title."
    )
    parser.add_argument(
        "--legend",
        type=str,
        help="Legend for two comparing figures, split by '+'. Default: none",
    )
    parser.add_argument(
        "--store-meta",
        type=str,
        help="Store all meta data with pickle (for further usage, e.g. plot more pretty figures).",
    )
    parser.add_argument(
        "-o",
        "--o-path",
        type=str,
        default=None,
        help="Path of the output figure path. If not specified, saved at the directory of input log file.",
    )
    args = parser.parse_args()

    for fp in [args.store_meta, args.o_path]:
        if fp is not None:
            assert os.path.isdir(os.path.dirname(fp)), f"Directory of {fp} not exist!"

    if len(args.log) == 1:
        opath, metas = plot_tb(args.log[0], title=args.title, o_path=args.o_path)
    else:
        legends = args.legend
        if legends is not None:
            legends = legends.split("+")
            assert len(legends) == len(args.log), (
                "the legends you gave is not equal to the number of log tracks\n"
                f"# legends = {len(legends)}, # tracks = {len(args.log)}"
            )

        opath, metas = cmp(
            args.log, legends=legends, title=args.title, o_path=args.o_path
        )
    if args.store_meta is not None:
        with open(args.store_meta, "wb") as fo:
            pickle.dump(metas, fo)
        print(f"Meta data saved at {args.store_meta}")
