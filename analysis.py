#!/usr/bin/env python3
"""
Analyzes particle decay events by generating signal and background data, computing factors such as Q-factors and sPlot weights, and visualizing the data through plots and fits. The available options enable customization of the dataset size, plot types, and factor calculations.

Usage:
    analysis.py [options]

Options:
    -h --help               Show this screen.
    --num-sig=<nsig>        Number of signal events to generate. [default: 10000]
    --num-bkg=<nbkg>        Number of background events to generate. [default: 10000]
    --parallel              Use parallel processing for event generation.
    --knn=<knn>             Number of nearest neighbors for kNN calculations. [default: 100]
    --density-knn           Compute kNN calculations based off on local density for each event
    --radius-knn=<radius>   Use radius-based neighbors calculations with specified radius. [default: None]
    --t-dep                 Use t-dependence in mass variable
"""

# Import necessary libraries
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import scipy.optimize as opt
from docopt import docopt
from iminuit import Minuit, cost
from iminuit.util import MError
from matplotlib.colors import LinearSegmentedColormap
from rich.console import Console
from rich.progress import Progress, track
from rich.table import Table
from scipy.integrate import quad
from scipy.sparse.csgraph import connected_components
from scipy.special import voigt_profile
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

# Set matplotlib and random number generator settings
# mpl.style.use("frappe")
plt.rc('axes', labelsize=16)
rng = np.random.default_rng(1)
console = Console()

# Define constants to generate MC according to https://arxiv.org/abs/0804.3382
m_min, m_max = 0.68, 0.88
b_true = 0.3
m_omega = 0.78256  # GeV/c2
G_omega = 0.00844  # GeV,
sigma = 0.005  # GeV
p00_true = 0.65
p1n1_true = 0.05
p10_true = 0.10
t_true = 0.11
t_false = 0.43
t_min = 0
t_max = 2
g_true = 0.13
g_false = 0.56
g_min = -1.8
g_max = 1.8
voigt_norm = quad(lambda x: voigt_profile((x - m_omega), sigma, m_omega * G_omega / 2), m_min, m_max)
RED = '#CC3311'
BLUE = '#0077BB'
PURPLE = '#AA3377'
BLACK = '#000000'
PALE_GRAY = '#DDDDDD'
DARK_GRAY = '#555555'
ERROR_RED = '#CC3311'
# colors = [(136, 46, 114), (25, 101, 176), (82, 137, 199), (123, 175, 222), (78, 178, 101), (144, 201, 135), (202, 224, 171), (247, 240, 86), (246, 193, 65), (241, 147, 45), (232, 96, 28), (220, 5, 12)]
# colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
# CMAP = LinearSegmentedColormap.from_list("analysis", colors, N=1000)
CMAP = 'viridis'


# Define an Event namedtuple for easy handling of data
class Event(NamedTuple):
    mass: float
    costheta: float
    phi: float
    t: float
    g: float


# print(f"Norm of voigtian over ({m_min}, {m_max}): {voigt_norm[0]}±{voigt_norm[1]}")


# Define model functions for signal mass, background mass, signal angular distribution, etc.
def m_sig(m: float | np.ndarray) -> float | np.ndarray:
    """Signal mass distribution modeled by a normalized Voigtian"""
    return voigt_profile((m - m_omega), sigma, m_omega * G_omega / 2) / voigt_norm[0]


m_sig_max = m_sig(m_omega)


def m_bkg(m: float | np.ndarray, b: float = b_true) -> float | np.ndarray:
    """Background mass distribution modeled as a linear function"""
    return 2 * (m_min * (b - 1) + m_max * b + m - 2 * b * m) / (m_min - m_max) ** 2


m_bkg_max = m_bkg(m_max, b_true)


def w_sig(
    costheta: float | np.ndarray,
    phi: float | np.ndarray,
    p00: float = p00_true,
    p1n1: float = p1n1_true,
    p10: float = p10_true,
) -> float | np.ndarray:
    """Signal angular distribution"""
    theta = np.arccos(costheta)
    return (3 / (4 * np.pi)) * (
        0.5 * (1 - p00)
        + 0.5 * (3 * p00 - 1) * np.cos(theta) ** 2
        - p1n1 * np.sin(theta) ** 2 * np.cos(2 * phi)
        - np.sqrt(2) * p10 * np.sin(2 * theta) * np.cos(phi)
    )


w_sig_max = 1.61558


def w_bkg(costheta: float | np.ndarray, phi: float | np.ndarray) -> float | np.ndarray:
    """Background angular distribution"""
    theta = np.arccos(costheta)
    return (1 + np.abs(np.sin(theta) * np.cos(phi))) / (6 * np.pi)


w_bkg_max = 1 / (3 * np.pi)


def t_sig(t: float | np.ndarray, tau: float = t_true) -> float | np.ndarray:
    """Signal t distribution"""
    return np.exp(-t / tau) / tau


t_sig_max = t_sig(t_min, t_true)


def t_bkg(t: float | np.ndarray, tau: float = t_false) -> float | np.ndarray:
    """Background t distribution"""
    return np.exp(-t / tau) / tau


t_bkg_max = t_bkg(t_min, t_false)


def g_sig(g: float | np.ndarray, sigma: float = g_true) -> float | np.ndarray:
    """Signal g distribution"""
    return np.exp(-0.5 * g**2 / sigma**2) / (np.sqrt(2 * np.pi) * sigma)


g_sig_max = g_sig(0, g_true)


def g_bkg(g: float | np.ndarray, sigma: float = t_false) -> float | np.ndarray:
    """Background g distribution"""
    return np.exp(-0.5 * g**2 / sigma**2) / (np.sqrt(2 * np.pi) * sigma)


g_bkg_max = g_bkg(0, g_false)


# Functions to generate signal and background events
def gen_sig(n: int = 10_000) -> list:
    """Generate signal events"""
    with Progress(transient=True) as progress:
        m_task = progress.add_task('Generating Signal (mass)', total=n)
        w_task = progress.add_task('Generating Signal (costheta, phi)', total=n)
        t_task = progress.add_task('Generating Signal (t)', total=n)
        g_task = progress.add_task('Generating Signal (g)', total=n)
        ms = []
        while len(ms) < n:
            m_star = rng.uniform(m_min, m_max)
            if m_sig(m_star) >= rng.uniform(0, m_sig_max):
                ms.append(m_star)
                progress.advance(m_task)
        costhetas = []
        phis = []
        while len(costhetas) < n:
            costheta_star = rng.uniform(-1, 1)
            phi_star = rng.uniform(-np.pi, np.pi)
            if w_sig(costheta_star, phi_star) >= rng.uniform(0, w_sig_max):
                costhetas.append(costheta_star)
                phis.append(phi_star)
                progress.advance(w_task)
        ts = []
        while len(ts) < n:
            t_star = rng.uniform(t_min, t_max)
            if t_sig(t_star) >= rng.uniform(0, t_sig_max):
                ts.append(t_star)
                progress.advance(t_task)
        gs = []
        while len(gs) < n:
            g_star = rng.uniform(g_min, g_max)
            if g_sig(g_star) >= rng.uniform(0, g_sig_max):
                gs.append(g_star)
                progress.advance(g_task)

        return [Event(m, costheta, phi, t, g) for m, costheta, phi, t, g in zip(ms, costhetas, phis, ts, gs)]


def gen_bkg(n: int = 10_000) -> list:
    """Generate background events"""
    with Progress(transient=True) as progress:
        m_task = progress.add_task('Generating Background (mass)', total=n)
        w_task = progress.add_task('Generating Background (costheta, phi)', total=n)
        t_task = progress.add_task('Generating Background (t)', total=n)
        g_task = progress.add_task('Generating Background (g)', total=n)
        ms = []
        while len(ms) < n:
            m_star = rng.uniform(m_min, m_max)
            if m_bkg(m_star) >= rng.uniform(0, m_bkg_max):
                ms.append(m_star)
                progress.advance(m_task)
        costhetas = []
        phis = []
        while len(costhetas) < n:
            costheta_star = rng.uniform(-1, 1)
            phi_star = rng.uniform(-np.pi, np.pi)
            if w_bkg(costheta_star, phi_star) >= rng.uniform(0, w_bkg_max):
                costhetas.append(costheta_star)
                phis.append(phi_star)
                progress.advance(w_task)
        ts = []
        while len(ts) < n:
            t_star = rng.uniform(t_min, t_max)
            if t_bkg(t_star) >= rng.uniform(0, t_bkg_max):
                ts.append(t_star)
                progress.advance(t_task)
        gs = []
        while len(gs) < n:
            g_star = rng.uniform(g_min, g_max)
            if g_bkg(g_star) >= rng.uniform(0, g_bkg_max):
                gs.append(g_star)
                progress.advance(g_task)
        return [Event(m, costheta, phi, t, g) for m, costheta, phi, t, g in zip(ms, costhetas, phis, ts, gs)]


# Functions to parallelize the generation of signal and background events if producing a large sample
def gen_event_partial(n, seed):
    rng = np.random.default_rng(seed)  # Initialize random seed for each process

    events = []
    for _ in range(n):
        ms, costhetas, phis, ts, gs = [], [], [], [], []

        # Generate mass
        while len(ms) < 1:
            m_star = rng.uniform(m_min, m_max)
            if m_sig(m_star) >= rng.uniform(0, m_sig_max):
                ms.append(m_star)

        # Generate costheta and phi
        while len(costhetas) < 1:
            costheta_star = rng.uniform(-1, 1)
            phi_star = rng.uniform(-np.pi, np.pi)
            if w_sig(costheta_star, phi_star) >= rng.uniform(0, w_sig_max):
                costhetas.append(costheta_star)
                phis.append(phi_star)

        # Generate t
        while len(ts) < 1:
            t_star = rng.uniform(t_min, t_max)
            if t_sig(t_star) >= rng.uniform(0, t_sig_max):
                ts.append(t_star)

        # Generate g
        while len(gs) < 1:
            g_star = rng.uniform(g_min, g_max)
            if g_sig(g_star) >= rng.uniform(0, g_sig_max):
                gs.append(g_star)

        events.append(Event(ms[0], costhetas[0], phis[0], ts[0], gs[0]))

    return events


def gen_bkg_event_partial(n, seed):
    rng = np.random.default_rng(seed)

    events = []
    for _ in range(n):
        ms, costhetas, phis, ts, gs = [], [], [], [], []

        # Generate mass for background
        while len(ms) < 1:
            m_star = rng.uniform(m_min, m_max)
            if m_bkg(m_star, b_true) >= rng.uniform(0, m_bkg_max):
                ms.append(m_star)

        # Generate costheta and phi for background
        while len(costhetas) < 1:
            costheta_star = rng.uniform(-1, 1)
            phi_star = rng.uniform(-np.pi, np.pi)
            if w_bkg(costheta_star, phi_star) >= rng.uniform(0, w_bkg_max):
                costhetas.append(costheta_star)
                phis.append(phi_star)

        # Generate t for background
        while len(ts) < 1:
            t_star = rng.uniform(t_min, t_max)
            if t_bkg(t_star, t_false) >= rng.uniform(0, t_bkg_max):
                ts.append(t_star)

        # Generate g for background
        while len(gs) < 1:
            g_star = rng.uniform(g_min, g_max)
            if g_bkg(g_star, g_false) >= rng.uniform(0, g_bkg_max):
                gs.append(g_star)

        events.append(Event(ms[0], costhetas[0], phis[0], ts[0], gs[0]))

    return events


def parallel_event_generation(gen_function, n=10000, num_workers=4):
    events_per_worker = n // num_workers
    seeds = list(range(num_workers))  # Unique seeds for each worker

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(gen_function, events_per_worker, seed) for seed in seeds]
        results = []
        for future in futures:
            results.extend(future.result())
    return results


# Calculate K-nearest neighbors for a given set of points
def k_nearest_neighbors(x, k):
    neighbors = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(x)
    _, indices = neighbors.kneighbors(x)
    return indices  # includes the point itself + k nearest neighbors


def calculate_local_density_knn(events, phase_space, metric='euclidean'):
    """
    Calculate the KNN based on local density for each event.
    The function returns indices of events in the neighborhood for each event.
    """
    # Calculate pairwise distances
    nbrs = NearestNeighbors(n_neighbors=len(events), algorithm='auto', metric=metric).fit(phase_space)
    distances, indices = nbrs.kneighbors(phase_space)

    # Estimate local density
    k_density = 5  # Can adjust this based on dataset
    densities = 1 / (distances[:, k_density] + 1e-5)  # Small constant to avoid division by zero

    # Sort densities to identify dense areas
    sorted_density_indices = np.argsort(densities)[::-1]

    # Compute variable K based on density ranking
    variable_k = np.linspace(50, 200, len(events)).astype(int)  # Linearly increasing K from 10 to 100
    sorted_k = variable_k[np.argsort(sorted_density_indices)]  # Assign K based on density ranking

    # Calculate variable KNN
    variable_knn_indices = [indices[i, :k] for i, k in enumerate(sorted_k)]

    return variable_knn_indices


def calculate_radius_neighbors(events, phase_space, radius, metric='euclidean'):
    """
    Calculate neighbors within a specified radius for each event.
    """
    nbrs = NearestNeighbors(radius=radius, algorithm='auto', metric=metric).fit(phase_space)
    distances, indices = nbrs.radius_neighbors(phase_space)

    # Convert sparse matrix to list of lists for indices
    indices_list = [list(ind) for ind in indices]
    return indices_list


# Define a class for weighted unbinned negative log-likelihood calculation
class WeightedUnbinnedNLL:
    @staticmethod
    def _safe_log(y: np.ndarray) -> np.ndarray:
        return np.log(y + 1e-323)

    @staticmethod
    def _unbinned_nll_weighted(y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Calculate the weighted unbinned negative log-likelihood"""
        return -np.sum(w * WeightedUnbinnedNLL._safe_log(y))

    def __init__(self, data: np.ndarray, model, weights: np.ndarray | None = None):
        self.weights = weights
        if weights is None:
            self.weights = np.ones(data.shape[0])
        self.data = data
        self.model = model

    def __call__(self, params, *args) -> float:
        """Evaluate the weighted unbinned NLL for given parameters"""
        y = self.model(self.data, *params, *args)
        if np.any(y < 0):
            return 1e20  # temporary fix...
        return WeightedUnbinnedNLL._unbinned_nll_weighted(y, self.weights)

    def fit(self, p0: list[float], *args, **kwargs):
        """Perform minimization to find the best-fit parameters"""
        return opt.minimize(lambda x, *args: self.__call__(x, *args), p0, **kwargs)


# Functions to plot event distributions and fits
def plot_events(
    events: list[Event],
    signal_events: list[Event],
    weights: np.ndarray | None = None,
    filename='events.png',
    directory='study',
):
    ms = [e.mass for e in events]
    costhetas = [e.costheta for e in events]
    phis = [e.phi for e in events]
    ts = [e.t for e in events]
    gs = [e.g for e in events]
    ms_sig = [e.mass for e in signal_events]
    ts_sig = [e.t for e in signal_events]
    gs_sig = [e.g for e in signal_events]

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(6, 6))
    # Plotting code...
    weights_label = 'Unweighted' if weights is None else 'Weighted'

    nw, bw, _ = ax[0, 0].hist(ms, bins=100, range=(m_min, m_max), weights=weights, label=weights_label, color=PALE_GRAY)
    ax[0, 0].hist(ms, bins=100, range=(m_min, m_max), weights=weights, histtype='step', color=ERROR_RED)
    nt, bt, _ = ax[0, 0].hist(ms_sig, bins=100, range=(m_min, m_max), histtype='step', label='Truth', color='black')
    # ax[0, 0].bar(x=bt[:-1], height=np.abs(nw - nt), bottom=np.minimum(nw, nt), width=np.diff(bt), align='edge', lw=0, color=RED, alpha=0.3)
    ax[0, 0].set_xlabel(r'$M_{3\pi}$ (GeV/$c^2$)')
    ax[0, 0].set_ylabel(r'Counts / 0.002')
    ax[0, 0].legend(loc='upper right')
    ax[0, 1].hist2d(costhetas, phis, bins=(50, 70), range=[(-1, 1), (-np.pi, np.pi)], weights=weights, cmap=CMAP)
    ax[0, 1].set_xlabel(r'$\cos(\theta)$')
    ax[0, 1].set_ylabel(r'$\phi$')
    nw, bw, _ = ax[1, 0].hist(ts, bins=100, range=(t_min, t_max), weights=weights, label=weights_label, color=PALE_GRAY)
    ax[1, 0].hist(ts, bins=100, range=(t_min, t_max), weights=weights, histtype='step', color=ERROR_RED)
    nt, bt, _ = ax[1, 0].hist(ts_sig, bins=100, range=(t_min, t_max), histtype='step', label='Truth', color='black')
    # ax[1, 0].bar(x=bt[:-1], height=np.abs(nw - nt), bottom=np.minimum(nw, nt), width=np.diff(bt), align='edge', lw=0, color=RED, alpha=0.3)
    ax[1, 0].set_xlabel('$t$ (arb)')
    ax[1, 0].set_ylabel(r'Counts / 0.02')
    ax[1, 0].legend(loc='upper right')
    nw, bw, _ = ax[1, 1].hist(gs, bins=100, range=(g_min, g_max), weights=weights, label=weights_label, color=PALE_GRAY)
    ax[1, 1].hist(gs, bins=100, range=(g_min, g_max), weights=weights, histtype='step', color=ERROR_RED)
    nt, bt, _ = ax[1, 1].hist(gs_sig, bins=100, range=(g_min, g_max), histtype='step', label='Truth', color='black')
    # ax[1, 1].bar(x=bt[:-1], height=np.abs(nw - nt), bottom=np.minimum(nw, nt), width=np.diff(bt), align='edge', lw=0, color=RED, alpha=0.3)
    ax[1, 1].set_xlabel('$g$ (arb)')
    ax[1, 1].set_ylabel(r'Counts / 0.036')
    ax[1, 1].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(Path(directory).resolve() / filename, dpi=300)
    plt.close()


def plot_all_events(events_sig: list[Event], events_bkg: list[Event], filename='generated_data.png', directory='study'):
    ms_sig = [e.mass for e in events_sig]
    costhetas_sig = [e.costheta for e in events_sig]
    phis_sig = [e.phi for e in events_sig]
    ts_sig = [e.t for e in events_sig]
    gs_sig = [e.g for e in events_sig]
    ms_bkg = [e.mass for e in events_bkg]
    costhetas_bkg = [e.costheta for e in events_bkg]
    phis_bkg = [e.phi for e in events_bkg]
    ts_bkg = [e.t for e in events_bkg]
    gs_bkg = [e.g for e in events_bkg]
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(12, 9), sharey='col')

    # signal plots
    ax[0, 0].hist(ms_sig, bins=100, range=(m_min, m_max), label='signal', color=BLUE)
    ax[0, 0].set_xlabel(r'$M_{3\pi}$ (GeV/$c^2$)')
    ax[0, 0].set_ylabel(r'Counts / 0.002')
    ax[0, 0].legend(loc='upper right')
    ax[0, 1].hist2d(costhetas_sig, phis_sig, bins=(50, 70), range=[(-1, 1), (-np.pi, np.pi)], label='signal', cmap=CMAP)
    ax[0, 1].set_xlabel(r'$\cos(\theta)$')
    ax[0, 1].set_ylabel(r'$\phi$')
    ax[0, 2].hist(ts_sig, bins=100, range=(t_min, t_max), label='signal', color=BLUE)
    ax[0, 2].set_xlabel('$t$ (arb)')
    ax[0, 2].set_ylabel(r'Counts / 0.02')
    ax[0, 2].legend(loc='upper right')
    ax[0, 3].hist(gs_sig, bins=100, range=(g_min, g_max), label='signal', color=BLUE)
    ax[0, 3].set_xlabel('$g$ (arb)')
    ax[0, 3].set_ylabel(r'Counts / 0.036')
    ax[0, 3].legend(loc='upper right')

    # background plots
    ax[1, 0].hist(ms_bkg, bins=100, range=(m_min, m_max), label='background', color=RED)
    ax[1, 0].set_xlabel(r'$M_{3\pi}$ (GeV/$c^2$)')
    ax[1, 0].set_ylabel(r'Counts / 0.002')
    ax[1, 0].legend(loc='upper right')
    ax[1, 1].hist2d(
        costhetas_bkg, phis_bkg, bins=(50, 70), range=[(-1, 1), (-np.pi, np.pi)], label='background', cmap=CMAP
    )
    ax[1, 1].set_xlabel(r'$\cos(\theta)$')
    ax[1, 1].set_ylabel(r'$\phi$')
    ax[1, 2].hist(ts_bkg, bins=100, range=(t_min, t_max), label='background', color=RED)
    ax[1, 2].set_xlabel('$t$ (arb)')
    ax[1, 2].set_ylabel(r'Counts / 0.02')
    ax[1, 2].legend(loc='upper right')
    ax[1, 3].hist(gs_bkg, bins=100, range=(g_min, g_max), label='background', color=RED)
    ax[1, 3].set_xlabel('$g$ (arb)')
    ax[1, 3].set_ylabel(r'Counts / 0.036')
    ax[1, 3].legend(loc='upper right')

    # combined plots
    ax[2, 0].hist(
        [ms_bkg, ms_sig],
        bins=100,
        range=(m_min, m_max),
        stacked=True,
        color=[RED, BLUE],
        label=['background', 'signal'],
    )
    ax[2, 0].set_xlabel(r'$M_{3\pi}$ (GeV/$c^2$)')
    ax[2, 0].set_ylabel(r'Counts / 0.002')
    ax[2, 0].legend(loc='upper right')
    ax[2, 1].hist2d(
        costhetas_bkg + costhetas_sig, phis_bkg + phis_sig, bins=(50, 70), range=[(-1, 1), (-np.pi, np.pi)], cmap=CMAP
    )
    ax[2, 1].set_xlabel(r'$\cos(\theta)$')
    ax[2, 1].set_ylabel(r'$\phi$')
    ax[2, 2].hist(
        [ts_bkg, ts_sig],
        bins=100,
        range=(t_min, t_max),
        stacked=True,
        color=[RED, BLUE],
        label=['background', 'signal'],
    )
    ax[2, 2].set_xlabel('$t$ (arb)')
    ax[2, 2].set_ylabel(r'Counts / 0.02')
    ax[2, 2].legend(loc='upper right')
    ax[2, 3].hist(
        [gs_bkg, gs_sig],
        bins=100,
        range=(g_min, g_max),
        stacked=True,
        color=[RED, BLUE],
        label=['background', 'signal'],
    )
    ax[2, 3].set_xlabel('$g$ (arb)')
    ax[2, 3].set_ylabel(r'Counts / 0.036')
    ax[2, 3].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(Path(directory).resolve() / filename, dpi=300)
    plt.close()


def calculate_sideband_weights(events: list[Event]) -> np.ndarray:
    ms = np.array([e.mass for e in events])
    left_cut = m_omega - 3 * G_omega
    right_cut = m_omega + 3 * G_omega

    def model(m: np.ndarray, z, b) -> np.ndarray:
        return z * m_sig(m) + (1 - z) * m_bkg(m, b)

    c = cost.UnbinnedNLL(ms, model)
    # 100% signal starting condition
    m_1 = Minuit(c, z=1.0, b=b_true)
    m_1.limits['z'] = (0, 1)
    m_1.migrad()
    # 100% background starting condition
    m_2 = Minuit(c, z=0.0, b=b_true)
    m_2.limits['z'] = (0, 1)
    m_2.migrad()
    # 50% signal / 50% background starting condition
    m_3 = Minuit(c, z=0.5, b=b_true)
    m_3.limits['z'] = (0, 1)
    m_3.migrad()
    fits = [m_1, m_2, m_3]
    nlls = np.array([m.fval for m in fits])
    best_fit = fits[np.argmin(nlls)]

    left_area = quad(lambda x: m_bkg(x, best_fit.values[1]), m_min, left_cut)[0]
    center_area = quad(lambda x: m_bkg(x, best_fit.values[1]), left_cut, right_cut)[0]
    right_area = quad(lambda x: m_bkg(x, best_fit.values[1]), right_cut, m_max)[0]

    weights = np.ones_like(ms)
    mask_sidebands = (ms < left_cut) | (ms > right_cut)
    weights[mask_sidebands] = -center_area / (left_area + right_area)
    return weights


def calculate_inplot(events: list[Event]) -> np.ndarray:
    ms = np.array([e.mass for e in events])

    def model(m: np.ndarray, z, b) -> np.ndarray:
        return z * m_sig(m) + (1 - z) * m_bkg(m, b)

    def inplot(m: np.ndarray, z, b) -> np.ndarray:
        return (z * m_sig(m)) / (z * m_sig(m) + (1 - z) * m_bkg(m, b))

    inplot_weights = []
    c = cost.UnbinnedNLL(ms, model)
    # 100% signal starting condition
    m_1 = Minuit(c, z=1.0, b=b_true)
    m_1.limits['z'] = (0, 1)
    m_1.migrad()
    # 100% background starting condition
    m_2 = Minuit(c, z=0.0, b=b_true)
    m_2.limits['z'] = (0, 1)
    m_2.migrad()
    # 50% signal / 50% background starting condition
    m_3 = Minuit(c, z=0.5, b=b_true)
    m_3.limits['z'] = (0, 1)
    m_3.migrad()
    fits = [m_1, m_2, m_3]
    nlls = np.array([m.fval for m in fits])
    best_fit = fits[np.argmin(nlls)]
    inplot_weights = inplot(ms, *best_fit.values)
    return np.array(inplot_weights)


def calculate_q_factors(
    events: list[Event],
    phase_space: np.ndarray,
    name: str,
    num_knn: int,
    use_density_knn=False,
    use_radius_knn=None,
    plot_indices: list[int] | None = None,
) -> np.ndarray:
    ms = np.array([e.mass for e in events])

    knn_indices = []

    if use_density_knn:
        tag = '_density'
        # Calculate KNN based on local density
        with console.status('Calculating K-Nearest Neighbors Based on Local Density'):
            knn_indices = calculate_local_density_knn(events, phase_space)
    elif use_radius_knn:
        tag = '_radius'
        radius = float(use_radius_knn)
        with console.status('Calculating Radius Neighbors'):
            knn_indices = calculate_radius_neighbors(events, phase_space, radius)
    else:
        tag = ''
        # Standard KNN calculation
        with console.status('Calculating K-Nearest Neighbors'):
            indices = k_nearest_neighbors(phase_space, num_knn)
            # Exclude the first index for each event since it is the event itself
            knn_indices = [index_set[1:] for index_set in indices]

    def model(m: np.ndarray, z, b) -> np.ndarray:
        return z * m_sig(m) + (1 - z) * m_bkg(m, b)

    def inplot(m, z, b) -> float:
        return (z * m_sig(m)) / (z * m_sig(m) + (1 - z) * m_bkg(m, b))

    q_factors = []
    sq_factors = []
    for i in track(range(len(events)), description='Calculating Q-Factors'):
        indices = knn_indices[i]
        c = cost.UnbinnedNLL(ms[indices], model)
        # 100% signal starting condition
        m_1 = Minuit(c, z=1.0, b=b_true)
        m_1.limits['z'] = (0, 1)
        m_1.migrad()
        # 100% background starting condition
        m_2 = Minuit(c, z=0.0, b=b_true)
        m_2.limits['z'] = (0, 1)
        m_2.migrad()
        # 50% signal / 50% background starting condition
        m_3 = Minuit(c, z=0.5, b=b_true)
        m_3.limits['z'] = (0, 1)
        m_3.migrad()
        fits = [m_1, m_2, m_3]
        nlls = np.array([m.fval for m in fits])
        best_fit = fits[np.argmin(nlls)]
        n_sig = len(ms[indices]) * best_fit.values[0]
        n_bkg = len(ms[indices]) * (1 - best_fit.values[0])
        b = best_fit.values[1]
        V_ss_inv = np.sum(
            np.array([m_sig(m) * m_sig(m) / (n_sig * m_sig(m) + n_bkg * m_bkg(m, b)) ** 2 for m in ms[indices]]), axis=0
        )
        V_sb_inv = np.sum(
            np.array([m_sig(m) * m_bkg(m, b) / (n_sig * m_sig(m) + n_bkg * m_bkg(m, b)) ** 2 for m in ms[indices]]),
            axis=0,
        )
        V_bb_inv = np.sum(
            np.array([m_bkg(m, b) * m_bkg(m, b) / (n_sig * m_sig(m) + n_bkg * m_bkg(m, b)) ** 2 for m in ms[indices]]),
            axis=0,
        )
        Vmat_inv = np.array([[V_ss_inv, V_sb_inv], [V_sb_inv, V_bb_inv]])
        # Fix issue if matrix inversion is not possible
        try:
            V = np.linalg.inv(Vmat_inv)
        except np.linalg.LinAlgError:
            print('Encountered a singular matrix, applying regularization.')
            epsilon = 1e-5  # Small regularization term
            Vmat_inv_reg = Vmat_inv + epsilon * np.eye(Vmat_inv.shape[0])
            V = np.linalg.inv(Vmat_inv_reg)

        sq_factors.append(
            (V[0, 0] * m_sig(ms[i]) + V[0, 1] * m_bkg(ms[i], b)) / (n_sig * m_sig(ms[i]) + n_bkg * m_bkg(ms[i], b))
        )
        q_factors.append(inplot(ms[i], *best_fit.values))
        if plot_indices and i in plot_indices:
            plot_qfactor_fit(
                ms[i],
                ms[indices],
                z_fit=best_fit.values[0],
                b_fit=best_fit.values[1],
                event_index=i,
                qfactor_type=f'{name}{tag}',
            )
    return np.array(q_factors), np.array(sq_factors)


def calculate_splot_weights(events: list[Event], sig_frac_init=0.5, b_init=0.5) -> np.ndarray:
    """Calculate sPlot weights for distinguishing signal from background"""
    ms = np.array([e.mass for e in events])  # Extracting the mass values from events

    def model(m: np.ndarray, sig_frac, b) -> np.ndarray:
        return sig_frac * m_sig(m) + (1 - sig_frac) * m_bkg(m, b)

    # Performing the fit
    c = cost.UnbinnedNLL(ms, model)
    mi = Minuit(c, sig_frac=sig_frac_init, b=b_init)
    mi.limits['sig_frac'] = (0, 1)  # Ensuring physical bounds
    mi.limits['b'] = (0, 1)
    mi.migrad()

    # Extract fit results for signal and background contributions
    n_sig = len(events) * mi.values['sig_frac']
    n_bkg = len(events) * (1 - mi.values['sig_frac'])
    b = mi.values['b']

    # Calculate inverse variance matrix elements
    V_ss_inv = np.sum([m_sig(m) ** 2 / (n_sig * m_sig(m) + n_bkg * m_bkg(m, b)) ** 2 for m in ms])
    V_sb_inv = np.sum([m_sig(m) * m_bkg(m, b) / (n_sig * m_sig(m) + n_bkg * m_bkg(m, b)) ** 2 for m in ms])
    V_bb_inv = np.sum([m_bkg(m, b) ** 2 / (n_sig * m_sig(m) + n_bkg * m_bkg(m, b)) ** 2 for m in ms])
    Vmat_inv = np.array([[V_ss_inv, V_sb_inv], [V_sb_inv, V_bb_inv]])
    V = np.linalg.inv(Vmat_inv)

    # Calculate sWeights and bWeights for each event
    sweights = [(V[0, 0] * m_sig(m) + V[0, 1] * m_bkg(m, b)) / (n_sig * m_sig(m) + n_bkg * m_bkg(m, b)) for m in ms]
    bweights = [(V[1, 0] * m_sig(m) + V[1, 1] * m_bkg(m, b)) / (n_sig * m_sig(m) + n_bkg * m_bkg(m, b)) for m in ms]

    # Combine sweights and bweights into a two-dimensional array
    return np.vstack((sweights, bweights)).T  # Transpose to get the correct shape


def fit_angles(
    events: list[Event],
    weights: np.ndarray | None = None,
):
    """
    Perform a weighted fit to the angular distribution of events to estimate the physics parameters
    p00, p1n1, and p10
    """

    def model(angles: np.ndarray, p00: float, p1n1: float, p10: float) -> np.ndarray:
        return w_sig(angles[:, 0], angles[:, 1], p00, p1n1, p10)

    angles = np.array([[e.costheta, e.phi] for e in events])
    wunll = WeightedUnbinnedNLL(angles, model, weights=weights)
    # return wunll.fit([p00_true, p1n1_true, p10_true])

    def _cost(p00: float, p1n1: float, p10: float) -> float:
        return wunll([p00, p1n1, p10])

    m = Minuit(_cost, p00=p00_true, p1n1=p1n1_true, p10=p10_true)
    m.migrad()
    m.minos(cl=1)
    return m


def fit_t(events: list[Event], weights: np.ndarray | None = None):
    """Perform a weighted fit to the t distribution of events to estimate the t parameter"""
    ts = np.array([e.t for e in events])
    wunll = WeightedUnbinnedNLL(ts, t_sig, weights=weights)
    # return wunll.fit([t_true])

    def _cost(tau: float) -> float:
        return wunll([tau])

    m = Minuit(_cost, tau=t_true)
    m.migrad()
    m.minos(cl=1)
    return m


def fit_g(events: list[Event], weights: np.ndarray | None = None):
    """Perform a weighted fit to the g distribution of events to estimate the g parameter"""
    gs = np.array([e.g for e in events])
    wunll = WeightedUnbinnedNLL(gs, g_sig, weights=weights)
    # return wunll.fit([g_true])

    def _cost(g: float) -> float:
        return wunll([g])

    m = Minuit(_cost, g=g_true)
    m.migrad()
    m.minos(cl=1)
    return m


def plot_qfactor_fit(mstar, ms, z_fit: float, b_fit: float, event_index: int, qfactor_type: str, directory='study'):
    # Combined model for the fit
    def model(m, z, b):
        return z * m_sig(m) + (1 - z) * m_bkg(m, b)

    plt.figure(figsize=(10, 6))
    # Scatter plot of selected masses
    plt.hist(ms, bins=30, label='Selected Events Masses', density=True, color=PALE_GRAY)
    plt.hist(ms, bins=30, density=True, histtype='step', color=DARK_GRAY)

    # Plot fit components
    m_range = np.linspace(m_min, m_max, 1000)
    plt.axvline(mstar, ls=':', lw=2, color=BLACK, label='Event')
    plt.plot(m_range, [z_fit * m_sig(m_val) for m_val in m_range], ls='-', lw=2, color=BLUE, label='Signal Fit')
    plt.plot(
        m_range,
        [(1 - z_fit) * m_bkg(m_val, b_fit) for m_val in m_range],
        ls='--',
        lw=2,
        color=RED,
        label='Background Fit',
    )
    plt.plot(
        m_range, [model(m_val, z_fit, b_fit) for m_val in m_range], ls='-', lw=2.5, color=PURPLE, label='Total Fit'
    )

    plt.xlabel('Mass')
    plt.ylabel('Density')
    plt.title(f'Fit for Event {event_index} and its Nearest Neighbors')
    plt.legend()
    plt.savefig(Path(directory).resolve() / f'qfactors_{qfactor_type}_{event_index}.png', dpi=300)
    plt.close()


def plot_radius_knn_visualization(events, selected_event_index, radius_knn, directory='study'):
    # Extract coordinates of events
    x_coords = [event.costheta for event in events]  # Example, adjust according to your actual spatial representation
    y_coords = [event.phi for event in events]  # Example

    # Coordinates of the selected event
    selected_event_x = x_coords[selected_event_index]
    selected_event_y = y_coords[selected_event_index]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x_coords, y_coords, label='Events')
    circle = plt.Circle(
        (selected_event_x, selected_event_y), radius_knn, color='r', fill=False, linewidth=2, label='Radius KNN'
    )
    plt.gca().add_patch(circle)

    # Highlight the selected event
    plt.scatter([selected_event_x], [selected_event_y], color='red', label='Selected Event')

    plt.xlabel('cosTheta')
    plt.ylabel('phi')
    plt.title(f'Visualization of Radius KNN ({radius_knn}) Neighborhood')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(Path(directory).resolve() / f'radius_vis_{radius_knn}.png', dpi=300)


def calculate_theoretical_q_factors(events, b_true):
    """
    Compute theoretical Q-factors based on the true underlying model.
    """
    # Extract the masses from all events as a NumPy array
    masses = np.array([event.mass for event in events])

    # Calculate signal and background densities using the vectorized functions
    signal_densities = m_sig(masses)
    console.print(signal_densities)
    background_densities = m_bkg(masses, b_true)
    console.print(background_densities)

    # Calculate total densities
    total_densities = signal_densities + background_densities
    console.print(total_densities)

    # Calculate Q-factors, handling division by zero by using np.where
    q_factors_theoretical = np.where(total_densities > 0, signal_densities / total_densities, 0)

    return q_factors_theoretical


def compare_q_factors(
    q_factors_calculated,
    q_factors_theoretical,
    title='Q-Factors Comparison',
    q_factor_type='standard',
    directory='study',
):
    """
    Compare calculated Q-factors to the theoretical Q-factor distribution.
    """
    # Visual comparison using histograms
    plt.figure(figsize=(12, 6))
    plt.scatter(q_factors_theoretical, q_factors_calculated, alpha=0.5, label='Data', color=BLUE)
    plt.plot(
        [0, 1], [0, 1], ls='-', color=RED, label=r'$Q_{\text{calc}} = Q_{\text{gen}}$'
    )  # Red line for Qcalc = Qgen
    plt.xlabel(r'Generated Q-factors ($Q_{\text{gen}}$)')
    plt.ylabel(r'Calculated Q-factors ($Q_{\text{calc}}$)')
    plt.title('Calculated vs. Generated Q-factors')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(-1, 2)
    plt.savefig(Path(directory).resolve() / f'theory_comparison_{q_factor_type}.png', dpi=300)
    plt.close()

    q_factors_difference = q_factors_calculated - q_factors_theoretical

    plt.figure(figsize=(12, 6))
    plt.hist(q_factors_difference, bins=50, range=(-1.0, 1.0), color=BLUE)
    plt.xlabel(r'$Q_{\text{calc}} - Q_{\text{gen}}$')
    plt.ylabel('Frequency')
    plt.title('Difference Between Calculated and Generated Q-factors')
    plt.savefig(Path(directory).resolve() / f'theory_comparison_subtract_{q_factor_type}.png', dpi=300)
    plt.close()

    # Quantitative comparison using Kolmogorov-Smirnov test
    ks_stat, ks_p_value = ks_2samp(q_factors_calculated, q_factors_theoretical)
    console.print(f'KS Statistic: {ks_stat}, P-value: {ks_p_value}')
    if ks_p_value < 0.05:
        console.print(
            'The calculated Q-factors distribution is [red]significantly different[/] from the theoretical distribution.'
        )
    else:
        console.print(
            'The calculated Q-factors distribution is [blue]not significantly different[/] from the theoretical distribution.'
        )


def main():
    args = docopt(__doc__)

    num_sig = int(args['--num-sig'])
    num_bkg = int(args['--num-bkg'])
    num_knn = int(args['--knn'])
    use_density_knn = args['--density-knn']
    use_radius_knn = args['--radius-knn']

    directory = 'study'
    if args['--t-dep']:
        directory += '_t_dep'

    if use_radius_knn != 'None':
        try:
            use_radius_knn = float(use_radius_knn)
        except ValueError:
            raise ValueError(f'Invalid value for --radius_knn: {use_radius_knn}')
    else:
        use_radius_knn = None
    tag = ''
    if use_density_knn:
        tag = '_density'
    elif use_radius_knn:
        tag = '_radius'

    parallel = args['--parallel']

    if parallel:
        # Generate events in parallel.
        console.print('Generating signal and background events in parallel ...')
        events_sig = parallel_event_generation(gen_event_partial, n=num_sig, num_workers=4)
        events_bkg = parallel_event_generation(gen_bkg_event_partial, n=num_bkg, num_workers=4)
    else:
        # Default to sequential generation.
        console.print('Generating signal and background events sequentially ...')
        events_sig = gen_sig(n=num_sig)
        events_bkg = gen_bkg(n=num_bkg)

    with console.status('Plotting events'):
        plot_all_events(events_sig, events_bkg, filename='all_events.png')
    events_all = events_sig + events_bkg

    t = Table(title='Fit Results')
    t.add_column('Weighting Method')
    t.add_column('ρ⁰₀₀')
    t.add_column('ρ⁰₁,₋₁')
    t.add_column('Re[ρ⁰₁₀]')
    t.add_column('τ')
    t.add_column('σ')
    t.add_row(
        'Truth',
        f'{p00_true:.3f}',
        f'{p1n1_true:.3f}',
        f'{p10_true:.3f}',
        f'{t_true:.3f}',
        f'{g_true:.3f}',
        end_section=True,
    )
    latex_table = rf"""
\begin{{table}}
\centering
\begin{{tabular}}{{lccccc}}\toprule
Weighting Method & $\rho^0_{{00}}$ & $\rho^0_{{1,-1}}$ & $\Re[\rho^0_{{10}}]$ & $\tau$ & $\sigma$ \\ \midrule
\textbf{{Truth}} & \textbf{{{p00_true:.3f}}} & \textbf{{{p1n1_true:.3f}}} & \textbf{{{p10_true:.3f}}} & \textbf{{{t_true:.3f}}} & \textbf{{{g_true:.3f}}} \\ \midrule
"""

    # console.print(latex_table)

    def colorize_by_number(
        fit: float,
        error: float | MError,
        true: float,
        sep='±',
        prefix='[{color}]',
        suffix='[/]',
        good='default',
        bad='yellow',
        worst='red',
    ) -> str:
        if isinstance(error, MError):
            if (fit < true + error.lower * 5) or (fit > true + error.upper * 5):
                return f'{prefix.format(color=worst)}{fit:.3f}^{{{error.upper:+.3f}}}_{{{error.lower:+.3f}}}{suffix}'
            if (fit < true + error.lower * 3) or (fit > true + error.upper * 3):
                return f'{prefix.format(color=bad)}{fit:.3f}^{{{error.upper:+.3f}}}_{{{error.lower:+.3f}}}{suffix}'
            return f'{prefix.format(color=good)}{fit:.3f}^{{{error.upper:+.3f}}}_{{{error.lower:+.3f}}}{suffix}'
        if abs(fit - true) > error * 5:
            return f'{prefix.format(color=worst)}{fit:.3f}{sep}{error:.3f}{suffix}'
        if abs(fit - true) > error * 3:
            return f'{prefix.format(color=bad)}{fit:.3f}{sep}{error:.3f}{suffix}'
        return f'{prefix.format(color=good)}{fit:.3f}{sep}{error:.3f}{suffix}'

    def get_results(events, weights=None, latex=False):
        m_angles = fit_angles(events, weights=weights)
        res_t = fit_t(events, weights=weights)
        res_g = fit_g(events, weights=weights)
        if latex:
            p00_fit = colorize_by_number(
                m_angles.values['p00'],
                m_angles.errors['p00'],
                p00_true,
                sep=r'\pm',
                prefix=r'{{\color{{{color}}}$',
                suffix=r'$}',
                good='black',
                bad='red',
                worst='red',
            )
            p1n1_fit = colorize_by_number(
                m_angles.values['p1n1'],
                m_angles.errors['p1n1'],
                p1n1_true,
                sep=r'\pm',
                prefix=r'{{\color{{{color}}}$',
                suffix=r'$}',
                good='black',
                bad='red',
                worst='red',
            )
            p10_fit = colorize_by_number(
                m_angles.values['p10'],
                m_angles.errors['p10'],
                p10_true,
                sep=r'\pm',
                prefix=r'{{\color{{{color}}}$',
                suffix=r'$}',
                good='black',
                bad='red',
                worst='red',
            )
            t_fit = colorize_by_number(
                res_t.values['tau'],
                res_t.errors['tau'],
                t_true,
                sep=r'\pm',
                prefix=r'{{\color{{{color}}}$',
                suffix=r'$}',
                good='black',
                bad='red',
                worst='red',
            )
            g_fit = colorize_by_number(
                res_g.values['g'],
                res_g.errors['g'],
                g_true,
                sep=r'\pm',
                prefix=r'{{\color{{{color}}}$',
                suffix=r'$}',
                good='black',
                bad='red',
                worst='red',
            )
            return [p00_fit, p1n1_fit, p10_fit, t_fit, g_fit]

        p00_fit = colorize_by_number(
            m_angles.values['p00'],
            m_angles.errors['p00'],
            p00_true,
        )
        p1n1_fit = colorize_by_number(
            m_angles.values['p1n1'],
            m_angles.errors['p1n1'],
            p1n1_true,
        )
        p10_fit = colorize_by_number(
            m_angles.values['p10'],
            m_angles.errors['p10'],
            p10_true,
        )
        t_fit = colorize_by_number(res_t.values['tau'], res_t.errors['tau'], t_true)
        g_fit = colorize_by_number(res_g.values['g'], res_g.errors['g'], g_true)
        return [p00_fit, p1n1_fit, p10_fit, t_fit, g_fit]

    plot_events(events_bkg, events_sig, weights=None, filename='bkg_no_weights.png', directory=directory)
    plot_events(events_sig, events_sig, weights=None, filename='sig_no_weights.png', directory=directory)
    plot_events(events_all, events_sig, weights=None, filename='all_no_weights.png', directory=directory)
    t.add_row('None', *get_results(events_all, weights=None))
    latex_table += (
        rf"No Weights & {' & '.join(get_results(events_all, weights=None, latex=True))} \\ \cmidrule(lr){{2-6}}" + '\n'
    )
    # console.print(t)
    # console.print(latex_table)

    sideband_weights = calculate_sideband_weights(events_all)
    plot_events(
        events_bkg, events_sig, weights=sideband_weights[num_sig:], filename='bkg_sideband.png', directory=directory
    )
    plot_events(
        events_sig, events_sig, weights=sideband_weights[:num_sig], filename='sig_sideband.png', directory=directory
    )
    plot_events(events_all, events_sig, weights=sideband_weights, filename='all_sideband.png', directory=directory)
    t.add_row('Sideband Subtraction', *get_results(events_all, weights=sideband_weights))
    latex_table += (
        rf"Sideband Subtraction & {' & '.join(get_results(events_all, weights=sideband_weights, latex=True))} \\ \cmidrule(lr){{2-6}}"
        + '\n'
    )
    # console.print(t)
    # console.print(latex_table)

    inplot_weights = calculate_inplot(events_all)
    plot_events(
        events_bkg, events_sig, weights=inplot_weights[num_sig:], filename='bkg_inplot.png', directory=directory
    )
    plot_events(
        events_sig, events_sig, weights=inplot_weights[:num_sig], filename='sig_inplot.png', directory=directory
    )
    plot_events(events_all, events_sig, weights=inplot_weights, filename='all_inplot.png', directory=directory)
    t.add_row('inPlot', *get_results(events_all, weights=inplot_weights))
    latex_table += (
        rf"inPlot (Q-factors with $k=N$) & {' & '.join(get_results(events_all, weights=inplot_weights, latex=True))} \\"
        + '\n'
    )
    # console.print(t)
    # console.print(latex_table)

    splot_weights = calculate_splot_weights(events_all)[:, 0]
    plot_events(events_bkg, events_sig, weights=splot_weights[num_sig:], filename='bkg_splot.png', directory=directory)
    plot_events(events_sig, events_sig, weights=splot_weights[:num_sig], filename='sig_splot.png', directory=directory)
    plot_events(events_all, events_sig, weights=splot_weights, filename='all_splot.png', directory=directory)
    t.add_row('sPlot', *get_results(events_all, weights=splot_weights))
    latex_table += (
        rf"sPlot & {' & '.join(get_results(events_all, weights=splot_weights, latex=True))} \\ \cmidrule(lr){{2-6}}"
        + '\n'
    )
    # console.print(t)
    # console.print(latex_table)

    q_factors_weights, sq_factors_weights = calculate_q_factors(
        events_all,
        phase_space=np.array([[e.costheta / (2 / 3), e.phi / (2 * np.pi**3 / 3)] for e in events_all]),
        name='angles',
        num_knn=num_knn,
        use_density_knn=use_density_knn,
        use_radius_knn=use_radius_knn,
        plot_indices=[0, 1, 2, num_sig, num_sig + 1, num_sig + 2],
    )
    plot_events(
        events_bkg,
        events_sig,
        weights=q_factors_weights[num_sig:],
        filename=f'bkg_q_factor{tag}.png',
        directory=directory,
    )
    plot_events(
        events_sig,
        events_sig,
        weights=q_factors_weights[:num_sig],
        filename=f'sig_q_factor{tag}.png',
        directory=directory,
    )
    plot_events(
        events_all, events_sig, weights=q_factors_weights, filename=f'all_q_factor{tag}.png', directory=directory
    )
    t.add_row('Q-Factors', *get_results(events_all, weights=q_factors_weights))
    latex_table += (
        rf"Q-Factors ($k=100$) & {' & '.join(get_results(events_all, weights=q_factors_weights, latex=True))} \\" + '\n'
    )
    plot_events(
        events_bkg,
        events_sig,
        weights=sq_factors_weights[num_sig:],
        filename=f'bkg_sq_factor{tag}.png',
        directory=directory,
    )
    plot_events(
        events_sig,
        events_sig,
        weights=sq_factors_weights[:num_sig],
        filename=f'sig_sq_factor{tag}.png',
        directory=directory,
    )
    plot_events(
        events_all, events_sig, weights=sq_factors_weights, filename=f'all_sq_factor{tag}.png', directory=directory
    )
    t.add_row('sQ-Factors', *get_results(events_all, weights=sq_factors_weights))
    latex_table += (
        rf"sQ-Factors ($k=100$) & {' & '.join(get_results(events_all, weights=sq_factors_weights, latex=True))} \\ \cmidrule(lr){{2-6}}"
        + '\n'
    )
    # console.print(t)
    # console.print(latex_table)

    q_factors_t_weights, sq_factors_t_weights = calculate_q_factors(
        events_all,
        phase_space=np.array(
            [[e.costheta / (2 / 3), e.phi / (2 * np.pi**3 / 3), e.t / ((t_max**3 - t_min**3) / 3)] for e in events_all]
        ),
        name='angles_t',
        num_knn=num_knn,
        use_density_knn=use_density_knn,
        use_radius_knn=use_radius_knn,
        plot_indices=[0, 1, 2, num_sig, num_sig + 1, num_sig + 2],
    )
    plot_events(
        events_bkg,
        events_sig,
        weights=q_factors_t_weights[num_sig:],
        filename=f'bkg_q_factor_t{tag}.png',
        directory=directory,
    )
    plot_events(
        events_sig,
        events_sig,
        weights=q_factors_t_weights[:num_sig],
        filename=f'sig_q_factor_t{tag}.png',
        directory=directory,
    )
    plot_events(
        events_all, events_sig, weights=q_factors_t_weights, filename=f'all_q_factor_t{tag}.png', directory=directory
    )
    t.add_row('Q-Factors (with t)', *get_results(events_all, weights=q_factors_t_weights))
    latex_table += (
        rf"Q-Factors (with t) & {' & '.join(get_results(events_all, weights=q_factors_t_weights, latex=True))} \\"
        + '\n'
    )
    plot_events(
        events_bkg,
        events_sig,
        weights=sq_factors_t_weights[num_sig:],
        filename=f'bkg_sq_factor_t{tag}.png',
        directory=directory,
    )
    plot_events(
        events_sig,
        events_sig,
        weights=sq_factors_t_weights[:num_sig],
        filename=f'sig_sq_factor_t{tag}.png',
        directory=directory,
    )
    plot_events(
        events_all, events_sig, weights=sq_factors_t_weights, filename=f'all_sq_factor_t{tag}.png', directory=directory
    )
    t.add_row('sQ-Factors (with t)', *get_results(events_all, weights=sq_factors_t_weights))
    latex_table += (
        rf"sQ-Factors (with t) & {' & '.join(get_results(events_all, weights=sq_factors_t_weights, latex=True))} \\ \cmidrule(lr){{2-6}}"
        + '\n'
    )
    # console.print(t)
    # console.print(latex_table)

    q_factors_g_weights, sq_factors_g_weights = calculate_q_factors(
        events_all,
        phase_space=np.array(
            [[e.costheta / (2 / 3), e.phi / (2 * np.pi**3 / 3), e.g / ((g_max**3 - g_min**3) / 3)] for e in events_all]
        ),
        name='angles_g',
        num_knn=num_knn,
        use_density_knn=use_density_knn,
        use_radius_knn=use_radius_knn,
        plot_indices=[0, 1, 2, num_sig, num_sig + 1, num_sig + 2],
    )
    plot_events(
        events_bkg,
        events_sig,
        weights=q_factors_g_weights[num_sig:],
        filename=f'bkg_q_factor_g{tag}.png',
        directory=directory,
    )
    plot_events(
        events_sig,
        events_sig,
        weights=q_factors_g_weights[:num_sig],
        filename=f'sig_q_factor_g{tag}.png',
        directory=directory,
    )
    plot_events(
        events_all, events_sig, weights=q_factors_g_weights, filename=f'all_q_factor_g{tag}.png', directory=directory
    )
    t.add_row('Q-Factors (with g)', *get_results(events_all, weights=q_factors_g_weights))
    latex_table += (
        rf"Q-Factors (with g) & {' & '.join(get_results(events_all, weights=q_factors_g_weights, latex=True))} \\"
        + '\n'
    )
    plot_events(
        events_bkg,
        events_sig,
        weights=sq_factors_g_weights[num_sig:],
        filename=f'bkg_sq_factor_g{tag}.png',
        directory=directory,
    )
    plot_events(
        events_sig,
        events_sig,
        weights=sq_factors_g_weights[:num_sig],
        filename=f'sig_sq_factor_g{tag}.png',
        directory=directory,
    )
    plot_events(
        events_all, events_sig, weights=sq_factors_g_weights, filename=f'all_sq_factor_g{tag}.png', directory=directory
    )
    t.add_row('sQ-Factors (with g)', *get_results(events_all, weights=sq_factors_g_weights))
    latex_table += (
        rf"sQ-Factors (with g) & {' & '.join(get_results(events_all, weights=sq_factors_g_weights, latex=True))} \\ \cmidrule(lr){{2-6}}"
        + '\n'
    )
    # console.print(t)
    # console.print(latex_table)

    q_factors_t_g_weights, sq_factors_t_g_weights = calculate_q_factors(
        events_all,
        phase_space=np.array(
            [
                [
                    e.costheta / (2 / 3),
                    e.phi / (2 * np.pi**3 / 3),
                    e.t / ((t_max**3 - t_min**3) / 3),
                    e.g / ((g_max**3 - g_min**3) / 3),
                ]
                for e in events_all
            ]
        ),
        name='angles_t_g',
        num_knn=num_knn,
        use_density_knn=use_density_knn,
        use_radius_knn=use_radius_knn,
        plot_indices=[0, 1, 2, num_sig, num_sig + 1, num_sig + 2],
    )
    plot_events(
        events_bkg,
        events_sig,
        weights=q_factors_t_g_weights[num_sig:],
        filename=f'bkg_q_factor_t_g{tag}.png',
        directory=directory,
    )
    plot_events(
        events_sig,
        events_sig,
        weights=q_factors_t_g_weights[:num_sig],
        filename=f'sig_q_factor_t_g{tag}.png',
        directory=directory,
    )
    plot_events(
        events_all,
        events_sig,
        weights=q_factors_t_g_weights,
        filename=f'all_q_factor_t_g{tag}.png',
        directory=directory,
    )
    t.add_row('Q-Factors (with t and g)', *get_results(events_all, weights=q_factors_t_g_weights))
    latex_table += (
        rf"Q-Factors (with t and g) & {' & '.join(get_results(events_all, weights=q_factors_t_g_weights, latex=True))} \\"
        + '\n'
    )
    plot_events(
        events_bkg,
        events_sig,
        weights=sq_factors_t_g_weights[num_sig:],
        filename=f'bkg_sq_factor_t_g{tag}.png',
        directory=directory,
    )
    plot_events(
        events_sig,
        events_sig,
        weights=sq_factors_t_g_weights[:num_sig],
        filename=f'sig_sq_factor_t_g{tag}.png',
        directory=directory,
    )
    plot_events(
        events_all,
        events_sig,
        weights=sq_factors_t_g_weights,
        filename=f'all_sq_factor_t_g{tag}.png',
        directory=directory,
    )
    t.add_row('sQ-Factors (with t and g)', *get_results(events_all, weights=sq_factors_t_g_weights))
    latex_table += (
        rf"sQ-Factors (with t and g) & {' & '.join(get_results(events_all, weights=sq_factors_t_g_weights, latex=True))} \\ \bottomrule"
        + '\n'
    )
    # console.print(t)
    # console.print(latex_table)

    latex_table += r"""\end{tabular}
\caption{Fit results from each weighting method. Results which deviate more than $5\sigma$ are highlighted red.}
\label{table:fit_results}
\end{table}
    """

    console.print(latex_table)
    (Path(directory).resolve() / 'latex_table.txt').write_text(latex_table)

    if use_radius_knn:
        selected_event_index = 0  # Index of the event you want to inspect
        plot_radius_knn_visualization(events_all, selected_event_index, use_radius_knn, directory=directory)

    # Theoretical model remains constant across variants
    q_factors_theoretical = calculate_theoretical_q_factors(events_all, b_true)
    # q_factors_theoretical = inplot_weights

    compare_q_factors(
        q_factors_weights,
        q_factors_theoretical,
        title='Standard Q-Factors Comparison',
        q_factor_type='standard',
        directory=directory,
    )
    compare_q_factors(
        q_factors_t_weights,
        q_factors_theoretical,
        title='Q-Factors with t Comparison',
        q_factor_type='with_t',
        directory=directory,
    )
    compare_q_factors(
        q_factors_g_weights,
        q_factors_theoretical,
        title='Q-Factors with g Comparison',
        q_factor_type='with_g',
        directory=directory,
    )
    compare_q_factors(
        q_factors_t_g_weights,
        q_factors_theoretical,
        title='Q-Factors with t and g Comparison',
        q_factor_type='with_t_and_g',
        directory=directory,
    )
    compare_q_factors(
        sq_factors_weights,
        q_factors_theoretical,
        title='sQ-Factors Comparison',
        q_factor_type='sq_factors',
        directory=directory,
    )

    console.print(t)


if __name__ == '__main__':
    main()
