#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Calculate statistics of feature files."""

import argparse
import logging
import os

import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import dlp_mpi

from parallel_wavegan.datasets import MelDataset, MelSCPDataset
from parallel_wavegan.utils import read_hdf5, write_hdf5


def pairwise_mean_var_update(means_: list, vars_: list, n_samples_seen_: list):
    """Global mean and variance computation from sample mean and variance.

    Implementation of (1.5a) and (1.5b) described in:
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247

    >>> rng = np.random.default_rng(0)
    >>> m1 = rng.normal(scale=2, size=(100, 2))
    >>> m2 = rng.normal(loc=1, size=(200, 2))
    >>> scaler = StandardScaler().fit(np.concatenate((m1, m2)))
    >>> scaler.mean_, scaler.scale_
    (array([0.55809324, 0.73998954]), array([1.4801906 , 1.41725947]))
    >>> scaler_m1 = StandardScaler().fit(m1)
    >>> scaler_m1.mean_, scaler_m1.scale_
    (array([-0.14158778,  0.20264034]), array([1.93953458, 1.88941199]))
    >>> scaler_m2 = StandardScaler().fit(m2)
    >>> scaler_m2.mean_, scaler_m2.scale_
    (array([0.90793375, 1.00866414]), array([1.01901126, 1.00570357]))
    >>> pairwise_mean_var_update(
    ...     [scaler_m1.mean_, scaler_m2.mean_],
    ...     [scaler_m1.var_, scaler_m2.var_],
    ...     [scaler_m1.n_samples_seen_, scaler_m2.n_samples_seen_]
    ... )
    (array([0.55809324, 0.73998954]), array([1.48019059, 1.41725947]))
    """
    total_n_samples_seen = n_samples_seen_[0]
    global_sum = means_[0] * total_n_samples_seen
    global_sum_squares = vars_[0] * total_n_samples_seen
    for mean, var, n_samples in zip(means_[1:], vars_[1:], n_samples_seen_[1:]):
        T = mean * n_samples  # sum
        sum_squares = var * n_samples  # sum of squares
        global_sum_squares += (
            sum_squares
            + (
                total_n_samples_seen
                / (n_samples * (total_n_samples_seen + n_samples))
            )
            * (n_samples / (total_n_samples_seen + 1e-5) * global_sum - T) ** 2
        )
        global_sum += T
        total_n_samples_seen += n_samples
    global_mean = global_sum / total_n_samples_seen
    global_var = global_sum_squares / total_n_samples_seen
    global_scale = np.sqrt(global_var)
    return global_mean, global_scale


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean and variance of dumped raw features "
            "(See detail in parallel_wavegan/bin/compute_statistics.py)."
        )
    )
    parser.add_argument(
        "--feats-scp",
        "--scp",
        default=None,
        type=str,
        help=(
            "kaldi-style feats.scp file. "
            "you need to specify either feats-scp or rootdir."
        ),
    )
    parser.add_argument(
        "--rootdir",
        type=str,
        required=True,
        help=(
            "directory including feature files. "
            "you need to specify either feats-scp or rootdir."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--dumpdir",
        default=None,
        type=str,
        required=True,
        help=(
            "directory to save statistics. if not provided, "
            "stats will be saved in the above root directory."
        ),
    )
    parser.add_argument(
        "--target-feats",
        type=str,
        default="feats",
        choices=["feats", "local"],
        help="target name to compute statistics.",
    )
    parser.add_argument(
        "--utt2spk",
        default=None,
        type=str,
        help=(
            "kaldi-style spk2utt file. if given, calculate statistics of each speaker."
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging.",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check arguments
    if (args.feats_scp is not None and args.rootdir is not None) or (
        args.feats_scp is None and args.rootdir is None
    ):
        raise ValueError("Please specify either --rootdir or --feats-scp.")

    # check directory existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir)

    # get dataset
    if args.feats_scp is None:
        if config["format"] == "hdf5":
            mel_query = "*.h5"
            mel_load_fn = lambda x: read_hdf5(x, args.target_feats)  # NOQA
        elif config["format"] == "npy":
            mel_query = f"*-{args.target_feats}.npy"
            mel_load_fn = np.load
        else:
            raise ValueError("support only hdf5 or npy format.")
        dataset = MelDataset(
            args.rootdir,
            mel_query=mel_query,
            mel_load_fn=mel_load_fn,
            return_utt_id=False if args.utt2spk is None else True,
        )
    else:
        if args.target_feats != "feats":
            raise NotImplementedError("Not supported.")
        dataset = MelSCPDataset(
            args.feats_scp,
            return_utt_id=False if args.utt2spk is None else True,
        )
    logging.info(f"The number of files = {len(dataset)}.")

    if args.utt2spk is None:
        # calculate global statistics
        logging.info("Caluculate global statistics.")
        scaler = StandardScaler()
        for mel in tqdm(dataset):
            scaler.partial_fit(mel)

        if config["format"] == "hdf5":
            write_hdf5(
                os.path.join(args.dumpdir, "stats.h5"),
                "mean",
                scaler.mean_.astype(np.float32),
            )
            write_hdf5(
                os.path.join(args.dumpdir, "stats.h5"),
                "scale",
                scaler.scale_.astype(np.float32),
            )
        else:
            stats = np.stack([scaler.mean_, scaler.scale_], axis=0)
            np.save(
                os.path.join(args.dumpdir, "stats.npy"),
                stats.astype(np.float32),
                allow_pickle=False,
            )
    else:
        # calculate statistics of each speaker
        logging.info("Caluculate each speaker statistics.")
        with open(args.utt2spk) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
        utt2spk = {line.split()[0]: line.split()[1] for line in lines}
        spks = list(set(utt2spk.values()))
        spk2scaler = {spk: StandardScaler() for spk in spks}
        for utt_id, mel in tqdm(dataset):
            spk = utt2spk[utt_id]
            spk2scaler[spk].partial_fit(mel)

        for spk, scaler in spk2scaler.items():
            if config["format"] == "hdf5":
                write_hdf5(
                    os.path.join(args.dumpdir, "stats.h5"),
                    f"{spk}/mean",
                    scaler.mean_.astype(np.float32),
                )
                write_hdf5(
                    os.path.join(args.dumpdir, "stats.h5"),
                    f"{spk}/scale",
                    scaler.scale_.astype(np.float32),
                )
            else:
                stats = np.stack([scaler.mean_, scaler.scale_], axis=0)
                np.save(
                    os.path.join(args.dumpdir, f"stats-{spk}.npy"),
                    stats.astype(np.float32),
                    allow_pickle=False,
                )


if __name__ == "__main__":
    main()
