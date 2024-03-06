#!/usr/bin/env python3
from __future__ import annotations

from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplcatppuccin
import numpy as np
import scipy.optimize as opt
from iminuit import Minuit, cost
from rich.console import Console
from rich.progress import Progress, track
from rich.table import Table
from scipy.integrate import quad
from scipy.special import voigt_profile
from sklearn.neighbors import NearestNeighbors

# mpl.style.use("frappe")
plt.rc('axes', labelsize=16)
rng = np.random.default_rng(1)
console = Console()

# Generate MC according to https://arxiv.org/abs/0804.3382
m_min, m_max = 0.68, 0.88
b_true = 0.3
m_omega = 0.78256 # GeV/c2
G_omega = 0.00844 # GeV,
sigma = 0.005 # GeV
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
voigt_norm = quad(lambda x: voigt_profile((x - m_omega), sigma, m_omega * G_omega/2), m_min, m_max)

class Event(NamedTuple):
    mass: float
    costheta: float
    phi: float
    t: float
    g: float

# print(f"Norm of voigtian over ({m_min}, {m_max}): {voigt_norm[0]}±{voigt_norm[1]}")

def m_sig(m: float | np.ndarray) -> float | np.ndarray:
    return voigt_profile((m - m_omega), sigma, m_omega * G_omega/2) / voigt_norm[0]

m_sig_max = m_sig(m_omega)

def m_bkg(m: float | np.ndarray, b: float = b_true) -> float | np.ndarray:
    return 2 * (m_min * (b - 1) + m_max * b + m - 2 * b * m) / (m_min - m_max)**2

m_bkg_max = m_bkg(m_max, b_true)

def w_sig(costheta: float | np.ndarray, phi: float | np.ndarray,
          p00: float = p00_true, p1n1: float = p1n1_true, p10: float = p10_true) -> float | np.ndarray:
    theta = np.arccos(costheta)
    return (3 / (4 * np.pi)) * (0.5 * (1 - p00)
                                + 0.5 * (3 * p00 - 1) * np.cos(theta)**2
                                - p1n1 * np.sin(theta)**2 * np.cos(2 * phi)
                                - np.sqrt(2) * p10 * np.sin(2 * theta) * np.cos(phi))

w_sig_max = 1.61558

def w_bkg(costheta: float | np.ndarray, phi: float | np.ndarray) -> float | np.ndarray:
    theta = np.arccos(costheta)
    return (1 + np.abs(np.sin(theta) * np.cos(phi))) / (6 * np.pi)

w_bkg_max = 1 / (3 * np.pi)

def t_sig(t: float | np.ndarray, tau: float=t_true) -> float | np.ndarray:
    return np.exp(- t / tau) / tau

t_sig_max = t_sig(t_min, t_true)

def t_bkg(t: float | np.ndarray, tau: float=t_false) -> float | np.ndarray:
    return np.exp(- t / tau) / tau

t_bkg_max = t_bkg(t_min, t_false)

def g_sig(g: float | np.ndarray, sigma: float=g_true) -> float | np.ndarray:
    return np.exp(-0.5 * g**2 / sigma**2) / (np.sqrt(2 * np.pi) * sigma)

g_sig_max = g_sig(0, g_true)

def g_bkg(g: float | np.ndarray, sigma: float=t_false) -> float | np.ndarray:
    return np.exp(-0.5 * g**2 / sigma**2) / (np.sqrt(2 * np.pi) * sigma)

g_bkg_max = g_bkg(0, g_false)



def gen_sig(n: int = 10_000) -> list:
    with Progress(transient=True) as progress:
        m_task = progress.add_task("Generating Signal (mass)", total=n)
        w_task = progress.add_task("Generating Signal (costheta, phi)", total=n)
        t_task = progress.add_task("Generating Signal (t)", total=n)
        g_task = progress.add_task("Generating Signal (g)", total=n)
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
    with Progress(transient=True) as progress:
        m_task = progress.add_task("Generating Background (mass)", total=n)
        w_task = progress.add_task("Generating Background (costheta, phi)", total=n)
        t_task = progress.add_task("Generating Background (t)", total=n)
        g_task = progress.add_task("Generating Background (g)", total=n)
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

def k_nearest_neighbors(x, k=100):
    neighbors = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(x)
    _, indices = neighbors.kneighbors(x)
    return indices # includes the point itself + 100 nearest neighbors

class WeightedUnbinnedNLL:
    @staticmethod
    def _safe_log(y: np.ndarray) -> np.ndarray:
        return np.log(y + 1e-323)

    @staticmethod
    def _unbinned_nll_weighted(y: np.ndarray, w: np.ndarray) -> np.ndarray:
        return -np.sum(w * WeightedUnbinnedNLL._safe_log(y))

    def __init__(self, data: np.ndarray, model, weights: np.ndarray | None=None):
        self.weights = weights
        if weights is None:
            self.weights = np.ones(data.shape[0])
        self.data = data
        self.model = model

    def __call__(self, params, *args) -> float:
        y = self.model(self.data, *params, *args)
        if np.any(y < 0):
            return 1e20 # temporary fix...
        return WeightedUnbinnedNLL._unbinned_nll_weighted(y, self.weights)

    def fit(self, p0: list[float], *args, **kwargs):
        return opt.minimize(lambda x, *args: self.__call__(x, *args), p0, **kwargs)

def plot_events(events: list[Event], signal_events: list[Event], weights: np.ndarray | None = None, filename='events.png'):
    ms = [e.mass for e in events]
    costhetas = [e.costheta for e in events]
    phis = [e.phi for e in events]
    ts = [e.t for e in events]
    gs = [e.g for e in events]
    ms_sig = [e.mass for e in signal_events]
    ts_sig = [e.t for e in signal_events]
    gs_sig = [e.g for e in signal_events]
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(6, 6))
    ax[0, 0].hist(ms, bins=100, range=(m_min, m_max), weights=weights, label='Weighted')
    ax[0, 0].hist(ms_sig, bins=100, range=(m_min, m_max), histtype='step', label='Truth')
    ax[0, 0].set_xlabel(r"$M_{3\pi}$ (GeV/$c^2$)")
    ax[0, 0].set_ylabel(r"Counts / 0.002")
    ax[0, 0].legend(loc='upper right')
    ax[0, 1].hist2d(costhetas, phis, bins=(50, 70), range=[(-1, 1), (-np.pi, np.pi)], weights=weights)
    ax[0, 1].set_xlabel(r"$\cos(\theta)$")
    ax[0, 1].set_ylabel(r"$\phi$")
    ax[1, 0].hist(ts, bins=100, range=(t_min, t_max), weights=weights, label='Weighted')
    ax[1, 0].hist(ts_sig, bins=100, range=(t_min, t_max), histtype='step', label='Truth')
    ax[1, 0].set_xlabel("$t$ (arb)")
    ax[1, 0].set_ylabel(r"Counts / 0.02")
    ax[1, 0].legend(loc='upper right')
    ax[1, 1].hist(gs, bins=100, range=(g_min, g_max), weights=weights, label='Weighted')
    ax[1, 1].hist(gs_sig, bins=100, range=(g_min, g_max), histtype='step', label='Truth')
    ax[1, 1].set_xlabel("$g$ (arb)")
    ax[1, 1].set_ylabel(r"Counts / 0.036")
    ax[1, 1].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_all_events(events_sig: list[Event], events_bkg: list[Event], filename='generated_data.png'):
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
    ax[0, 0].hist(ms_sig, bins=100, range=(m_min, m_max), label="signal")
    ax[0, 0].set_xlabel(r"$M_{3\pi}$ (GeV/$c^2$)")
    ax[0, 0].set_ylabel(r"Counts / 0.002")
    ax[0, 0].legend(loc="upper right")
    ax[0, 1].hist2d(costhetas_sig, phis_sig, bins=(50, 70), range=[(-1, 1), (-np.pi, np.pi)], label="signal")
    ax[0, 1].set_xlabel(r"$\cos(\theta)$")
    ax[0, 1].set_ylabel(r"$\phi$")
    ax[0, 2].hist(ts_sig, bins=100, range=(t_min, t_max), label="signal")
    ax[0, 2].set_xlabel("$t$ (arb)")
    ax[0, 2].set_ylabel(r"Counts / 0.02")
    ax[0, 2].legend(loc="upper right")
    ax[0, 3].hist(gs_sig, bins=100, range=(g_min, g_max), label="signal")
    ax[0, 3].set_xlabel("$g$ (arb)")
    ax[0, 3].set_ylabel(r"Counts / 0.036")
    ax[0, 3].legend(loc="upper right")

    # background plots
    ax[1, 0].hist(ms_bkg, bins=100, range=(m_min, m_max), color='C1', label="background")
    ax[1, 0].set_xlabel(r"$M_{3\pi}$ (GeV/$c^2$)")
    ax[1, 0].set_ylabel(r"Counts / 0.002")
    ax[1, 0].legend(loc="upper right")
    ax[1, 1].hist2d(costhetas_bkg, phis_bkg, bins=(50, 70), range=[(-1, 1), (-np.pi, np.pi)], label="background")
    ax[1, 1].set_xlabel(r"$\cos(\theta)$")
    ax[1, 1].set_ylabel(r"$\phi$")
    ax[1, 2].hist(ts_bkg, bins=100, range=(t_min, t_max), color='C1', label="background")
    ax[1, 2].set_xlabel("$t$ (arb)")
    ax[1, 2].set_ylabel(r"Counts / 0.02")
    ax[1, 2].legend(loc="upper right")
    ax[1, 3].hist(gs_bkg, bins=100, range=(g_min, g_max), color='C1', label="background")
    ax[1, 3].set_xlabel("$g$ (arb)")
    ax[1, 3].set_ylabel(r"Counts / 0.036")
    ax[1, 3].legend(loc="upper right")

    # combined plots
    ax[2, 0].hist([ms_bkg, ms_sig], bins=100, range=(m_min, m_max), stacked=True, color=['C1', 'C0'], label=["background", "signal"])
    ax[2, 0].set_xlabel(r"$M_{3\pi}$ (GeV/$c^2$)")
    ax[2, 0].set_ylabel(r"Counts / 0.002")
    ax[2, 0].legend(loc="upper right")
    ax[2, 1].hist2d(costhetas_bkg + costhetas_sig, phis_bkg + phis_sig, bins=(50, 70), range=[(-1, 1), (-np.pi, np.pi)], label=["background", "signal"])
    ax[2, 1].set_xlabel(r"$\cos(\theta)$")
    ax[2, 1].set_ylabel(r"$\phi$")
    ax[2, 2].hist([ts_bkg, ts_sig], bins=100, range=(t_min, t_max), stacked=True, color=['C1', 'C0'], label=["background", "signal"])
    ax[2, 2].set_xlabel("$t$ (arb)")
    ax[2, 2].set_ylabel(r"Counts / 0.02")
    ax[2, 2].legend(loc="upper right")
    ax[2, 3].hist([gs_bkg, gs_sig], bins=100, range=(g_min, g_max), stacked=True, color=['C1', 'C0'], label=["background", "signal"])
    ax[2, 3].set_xlabel("$g$ (arb)")
    ax[2, 3].set_ylabel(r"Counts / 0.036")
    ax[2, 3].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
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
    mask_sidebands = (ms <  left_cut) | (ms >  right_cut)
    weights[mask_sidebands] = - center_area / (left_area + right_area)
    return weights

def calculate_q_factors(events: list[Event]) -> np.ndarray:
    ms = np.array([e.mass for e in events])
    phase_space = np.array([[e.costheta, e.phi] for e in events])

    with console.status("Calculating K-Nearest Neighbors"):
        knn = k_nearest_neighbors(phase_space)

    def model(m: np.ndarray, z, b) -> np.ndarray:
        return z * m_sig(m) + (1 - z) * m_bkg(m, b)

    def inplot(m, z, b) -> float:
        return (z * m_sig(m)) / (z * m_sig(m) + (1 - z) * m_bkg(m, b))

    q_factors = []
    for i in track(range(len(events)), description="Calculating Q-Factors"):
        c = cost.UnbinnedNLL(ms[knn[i]], model)
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
        q_factors.append(inplot(ms[i], *best_fit.values))
    return np.array(q_factors)

def calculate_q_factors_with_t(events: list[Event]) -> np.ndarray:
    ms = np.array([e.mass for e in events])
    phase_space = np.array([[e.costheta, e.phi, e.t] for e in events])

    with console.status("Calculating K-Nearest Neighbors"):
        knn = k_nearest_neighbors(phase_space)

    def model(m: np.ndarray, z, b) -> np.ndarray:
        return z * m_sig(m) + (1 - z) * m_bkg(m, b)

    def inplot(m, z, b) -> float:
        return (z * m_sig(m)) / (z * m_sig(m) + (1 - z) * m_bkg(m, b))

    q_factors = []
    for i in track(range(len(events)), description="Calculating Q-Factors"):
        c = cost.UnbinnedNLL(ms[knn[i]], model)
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
        q_factors.append(inplot(ms[i], *best_fit.values))
    return np.array(q_factors)

def calculate_q_factors_with_g(events: list[Event]) -> np.ndarray:
    ms = np.array([e.mass for e in events])
    phase_space = np.array([[e.costheta, e.phi, e.g] for e in events])

    with console.status("Calculating K-Nearest Neighbors"):
        knn = k_nearest_neighbors(phase_space)

    def model(m: np.ndarray, z, b) -> np.ndarray:
        return z * m_sig(m) + (1 - z) * m_bkg(m, b)

    def inplot(m, z, b) -> float:
        return (z * m_sig(m)) / (z * m_sig(m) + (1 - z) * m_bkg(m, b))

    q_factors = []
    for i in track(range(len(events)), description="Calculating Q-Factors"):
        c = cost.UnbinnedNLL(ms[knn[i]], model)
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
        q_factors.append(inplot(ms[i], *best_fit.values))
    return np.array(q_factors)

def calculate_q_factors_with_t_g(events: list[Event]) -> np.ndarray:
    ms = np.array([e.mass for e in events])
    phase_space = np.array([[e.costheta, e.phi, e.t, e.g] for e in events])

    with console.status("Calculating K-Nearest Neighbors"):
        knn = k_nearest_neighbors(phase_space)

    def model(m: np.ndarray, z, b) -> np.ndarray:
        return z * m_sig(m) + (1 - z) * m_bkg(m, b)

    def inplot(m, z, b) -> float:
        return (z * m_sig(m)) / (z * m_sig(m) + (1 - z) * m_bkg(m, b))

    q_factors = []
    for i in track(range(len(events)), description="Calculating Q-Factors"):
        c = cost.UnbinnedNLL(ms[knn[i]], model)
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
        q_factors.append(inplot(ms[i], *best_fit.values))
    return np.array(q_factors)


def calculate_splot_weights(events: list[Event]) -> np.ndarray:
    ms = np.array([e.mass for e in events])
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
    n_sig = len(ms) * best_fit.values[0]
    n_bkg = len(ms) * (1 - best_fit.values[0])
    b = best_fit.values[1]
    V_ss_inv = np.sum(np.array([m_sig(m) * m_sig(m) / (n_sig * m_sig(m) + n_bkg * m_bkg(m, b))**2 for m in ms]), axis=0)
    V_sb_inv = np.sum(np.array([m_sig(m) * m_bkg(m, b) / (n_sig * m_sig(m) + n_bkg * m_bkg(m, b))**2 for m in ms]), axis=0)
    V_bb_inv = np.sum(np.array([m_bkg(m, b) * m_bkg(m, b) / (n_sig * m_sig(m) + n_bkg * m_bkg(m, b))**2 for m in ms]), axis=0)
    Vmat_inv = np.array([[V_ss_inv, V_sb_inv], [V_sb_inv, V_bb_inv]])
    V = np.linalg.inv(Vmat_inv)
    return np.array([(V[0, 0] * m_sig(m) + V[0, 1] * m_bkg(m, b)) / (n_sig * m_sig(m) + n_bkg * m_bkg(m, b)) for m in ms])


def calculate_sq_factors(events: list[Event]) -> np.ndarray:
    ms = np.array([e.mass for e in events])
    phase_space = np.array([[e.costheta, e.phi] for e in events])

    with console.status("Calculating K-Nearest Neighbors"):
        knn = k_nearest_neighbors(phase_space)

    def model(m: np.ndarray, z, b) -> np.ndarray:
        return z * m_sig(m) + (1 - z) * m_bkg(m, b)

    def inplot(m, z, b) -> float:
        return (z * m_sig(m)) / (z * m_sig(m) + (1 - z) * m_bkg(m, b))

    sq_factors = []
    for i in track(range(len(events)), description="Calculating sQ-Factors"):
        c = cost.UnbinnedNLL(ms[knn[i]], model)
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
        n_sig = len(ms[knn[i]]) * best_fit.values[0]
        n_bkg = len(ms[knn[i]]) * (1 - best_fit.values[0])
        b = best_fit.values[1]
        V_ss_inv = np.sum(np.array([m_sig(m) * m_sig(m) / (n_sig * m_sig(m) + n_bkg * m_bkg(m, b))**2 for m in ms[knn[i]]]), axis=0)
        V_sb_inv = np.sum(np.array([m_sig(m) * m_bkg(m, b) / (n_sig * m_sig(m) + n_bkg * m_bkg(m, b))**2 for m in ms[knn[i]]]), axis=0)
        V_bb_inv = np.sum(np.array([m_bkg(m, b) * m_bkg(m, b) / (n_sig * m_sig(m) + n_bkg * m_bkg(m, b))**2 for m in ms[knn[i]]]), axis=0)
        Vmat_inv = np.array([[V_ss_inv, V_sb_inv], [V_sb_inv, V_bb_inv]])
        V = np.linalg.inv(Vmat_inv)
        sq_factors.append((V[0, 0] * m_sig(ms[i]) + V[0, 1] * m_bkg(ms[i], b)) / (n_sig * m_sig(ms[i]) + n_bkg * m_bkg(ms[i], b)))
    return np.array(sq_factors)

def fit_angles(events: list[Event], weights: np.ndarray | None = None, p00_init: float=p00_true, p1n1_init: float=p1n1_true, p10_init: float=p10_true):
    def model(angles: np.ndarray, p00: float, p1n1: float, p10: float) -> np.ndarray:
        return w_sig(angles[:, 0], angles[:, 1], p00, p1n1, p10)

    angles = np.array([[e.costheta, e.phi] for e in events])
    wunll = WeightedUnbinnedNLL(angles, model, weights=weights)
    return wunll.fit([p00_true, p1n1_true, p10_true])

def fit_t(events: list[Event], weights: np.ndarray | None = None, t_init: float=t_true):
    ts = np.array([e.t for e in events])
    wunll_t = WeightedUnbinnedNLL(ts, t_sig, weights=weights)
    return wunll_t.fit([t_true])

def fit_g(events: list[Event], weights: np.ndarray | None = None, g_init: float=g_true):
    gs = np.array([e.g for e in events])
    wunll_g = WeightedUnbinnedNLL(gs, g_sig, weights=weights)
    return wunll_g.fit([g_true])


def main():
    events_sig = gen_sig()
    events_bkg = gen_bkg()
    with console.status("Plotting events"):
        plot_all_events(events_sig, events_bkg, filename="all_events.png")
    events_all = events_sig + events_bkg

    sideband_weights = calculate_sideband_weights(events_all)
    q_factors = calculate_q_factors(events_all)
    q_factors_t = calculate_q_factors_with_t(events_all)
    q_factors_g = calculate_q_factors_with_g(events_all)
    q_factors_t_g = calculate_q_factors_with_t_g(events_all)
    sweights = calculate_splot_weights(events_all)
    sq_factors = calculate_sq_factors(events_all)

    t = Table(title="Fit Results")
    t.add_column("Weighting Method")
    t.add_column("ρ⁰₀₀")
    t.add_column("ρ⁰₁,₋₁")
    t.add_column("Re[ρ⁰₁₀]")
    t.add_column("τ")
    t.add_column("σ")
    t.add_row("Truth", f"{p00_true:.2f}", f"{p1n1_true:.2f}", f"{p10_true:.2f}", f"{t_true:.2f}", f"{g_true:.2f}", end_section=True)

    def colorize_by_number(fit: float, true: float, threshold_yellow: float=0.02, threshold_red: float=0.05) -> str:
        if abs(fit - true) > threshold_red:
            return f"[red]{fit:.2f}[/]"
        elif abs(fit - true) > threshold_yellow:
            return f"[yellow]{fit:.2f}[/]"
        else:
            return f"[blue]{fit:.2f}[/]"

    def get_results(events, weights=None):
        res_angles = fit_angles(events, weights=weights)
        res_t = fit_t(events, weights=weights)
        res_g = fit_g(events, weights=weights)
        p00_fit = colorize_by_number(res_angles.x[0], p00_true)
        p1n1_fit = colorize_by_number(res_angles.x[1], p1n1_true)
        p10_fit = colorize_by_number(res_angles.x[2], p10_true)
        t_fit = colorize_by_number(res_t.x[0], t_true)
        g_fit = colorize_by_number(res_g.x[0], g_true)
        return [p00_fit, p1n1_fit, p10_fit, t_fit, g_fit]

    t.add_row("None", *get_results(events_all, weights=None))
    t.add_row("Sideband Subtraction", *get_results(events_all, weights=sideband_weights))
    t.add_row("Q-Factors", *get_results(events_all, weights=q_factors))
    t.add_row("Q-Factors (with t)", *get_results(events_all, weights=q_factors_t))
    t.add_row("Q-Factors (with g)", *get_results(events_all, weights=q_factors_g))
    t.add_row("Q-Factors (with t and g)", *get_results(events_all, weights=q_factors_t_g))
    t.add_row("sWeights", *get_results(events_all, weights=sweights))
    t.add_row("sQ-Factors", *get_results(events_all, weights=sq_factors))

    console.print(t)

    plot_events(events_bkg, events_sig, weights=None, filename="bkg_no_weights.png")
    plot_events(events_bkg, events_sig, weights=sideband_weights[10_000:], filename="bkg_sideband.png")
    plot_events(events_bkg, events_sig, weights=q_factors[10_000:], filename="bkg_q_factor.png")
    plot_events(events_bkg, events_sig, weights=q_factors_t[10_000:], filename="bkg_q_factor_t.png")
    plot_events(events_bkg, events_sig, weights=q_factors_g[10_000:], filename="bkg_q_factor_g.png")
    plot_events(events_bkg, events_sig, weights=q_factors_t_g[10_000:], filename="bkg_q_factor_t_g.png")
    plot_events(events_bkg, events_sig, weights=sweights[10_000:], filename="bkg_sweight.png")
    plot_events(events_bkg, events_sig, weights=sq_factors[10_000:], filename="bkg_sq_factor.png")

    plot_events(events_sig, events_sig, weights=None, filename="sig_no_weights.png")
    plot_events(events_sig, events_sig, weights=sideband_weights[:10_000], filename="sig_sideband.png")
    plot_events(events_sig, events_sig, weights=q_factors[:10_000], filename="sig_q_factor.png")
    plot_events(events_sig, events_sig, weights=q_factors_t[:10_000], filename="sig_q_factor_t.png")
    plot_events(events_sig, events_sig, weights=q_factors_g[:10_000], filename="sig_q_factor_g.png")
    plot_events(events_sig, events_sig, weights=q_factors_t_g[:10_000], filename="sig_q_factor_t_g.png")
    plot_events(events_sig, events_sig, weights=sweights[:10_000], filename="sig_sweight.png")
    plot_events(events_sig, events_sig, weights=sq_factors[:10_000], filename="sig_sq_factor.png")

    plot_events(events_all, events_sig, weights=None, filename="all_no_weights.png")
    plot_events(events_all, events_sig, weights=sideband_weights, filename="all_sideband.png")
    plot_events(events_all, events_sig, weights=q_factors, filename="all_q_factor.png")
    plot_events(events_all, events_sig, weights=q_factors_t, filename="all_q_factor_t.png")
    plot_events(events_all, events_sig, weights=q_factors_g, filename="all_q_factor_g.png")
    plot_events(events_all, events_sig, weights=q_factors_t_g, filename="all_q_factor_t_g.png")
    plot_events(events_all, events_sig, weights=sweights, filename="all_sweight.png")
    plot_events(events_all, events_sig, weights=sq_factors, filename="all_sq_factor.png")


if __name__ == '__main__':
    main()
