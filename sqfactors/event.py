from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

import numpy as np
from rich.progress import Progress
from scipy.special import voigt_profile

from sqfactors import bounds, r, truths


class Event(NamedTuple):
    mass: float
    costheta: float
    phi: float
    t: float
    g: float


# Define model functions for signal mass, background mass, signal angular distribution, etc.
# voigt_norm = quad(lambda x: voigt_profile((x - truths["m_omega"]), truths["sigma"], truths["m_omega"] * truths["G_omega"] / 2), m_min, bounds["m_max"])
def m_sig(m: float | np.ndarray) -> float | np.ndarray:
    """Signal mass distribution modeled by a normalized Voigtian"""
    return voigt_profile(
        (m - truths['m_omega']), truths['sigma'], truths['m_omega'] * truths['G_omega'] / 2
    )


m_sig_max = m_sig(truths['m_omega'])


def m_bkg(m: float | np.ndarray, b: float = truths['b']) -> float | np.ndarray:
    """Background mass distribution modeled as a linear function"""
    x1 = bounds['m_min']
    x2 = bounds['m_max']
    y1 = b
    y2 = (2.0 - (x2 - x1) * y1) / (x2 - x1)
    return (y2 - y1) * (m - x1) / (x2 - x1) + y1


m_bkg_max = m_bkg(bounds['m_max'], truths['b'])


def w_sig(
    costheta: float | np.ndarray,
    phi: float | np.ndarray,
    p00: float = truths['p00'],
    p1n1: float = truths['p1n1'],
    p10: float = truths['p10'],
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


def t_sig(t: float | np.ndarray, tau: float = truths['tau_sig']) -> float | np.ndarray:
    """Signal t distribution"""
    return np.exp(-t / tau) / tau


t_sig_max = t_sig(bounds['t_min'], truths['tau_sig'])


def t_bkg(t: float | np.ndarray, tau: float = truths['tau_bkg']) -> float | np.ndarray:
    """Background t distribution"""
    return np.exp(-t / tau) / tau


t_bkg_max = t_bkg(bounds['t_min'], truths['tau_bkg'])


def g_sig(g: float | np.ndarray, sigma: float = truths['sigma_sig']) -> float | np.ndarray:
    """Signal g distribution"""
    return np.exp(-0.5 * g**2 / sigma**2) / (np.sqrt(2 * np.pi) * sigma)


g_sig_max = g_sig(0, truths['sigma_sig'])


def g_bkg(g: float | np.ndarray, sigma: float = truths['tau_bkg']) -> float | np.ndarray:
    """Background g distribution"""
    return np.exp(-0.5 * g**2 / sigma**2) / (np.sqrt(2 * np.pi) * sigma)


g_bkg_max = g_bkg(0, truths['sigma_bkg'])


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
            m_star = r().uniform(bounds['m_min'], bounds['m_max'])
            if m_sig(m_star) >= r().uniform(0, m_sig_max):
                ms.append(m_star)
                progress.advance(m_task)
        costhetas = []
        phis = []
        while len(costhetas) < n:
            costheta_star = r().uniform(-1, 1)
            phi_star = r().uniform(-np.pi, np.pi)
            if w_sig(costheta_star, phi_star) >= r().uniform(0, w_sig_max):
                costhetas.append(costheta_star)
                phis.append(phi_star)
                progress.advance(w_task)
        ts = []
        while len(ts) < n:
            t_star = r().uniform(bounds['t_min'], bounds['t_max'])
            if t_sig(t_star) >= r().uniform(0, t_sig_max):
                ts.append(t_star)
                progress.advance(t_task)
        gs = []
        while len(gs) < n:
            g_star = r().uniform(bounds['g_min'], bounds['g_max'])
            if g_sig(g_star) >= r().uniform(0, g_sig_max):
                gs.append(g_star)
                progress.advance(g_task)

        return [
            Event(m, costheta, phi, t, g)
            for m, costheta, phi, t, g in zip(ms, costhetas, phis, ts, gs)
        ]


def gen_bkg(n: int = 10_000) -> list:
    """Generate background events"""
    with Progress(transient=True) as progress:
        m_task = progress.add_task('Generating Background (mass)', total=n)
        w_task = progress.add_task('Generating Background (costheta, phi)', total=n)
        t_task = progress.add_task('Generating Background (t)', total=n)
        g_task = progress.add_task('Generating Background (g)', total=n)
        ms = []
        while len(ms) < n:
            m_star = r().uniform(bounds['m_min'], bounds['m_max'])
            if m_bkg(m_star) >= r().uniform(0, m_bkg_max):
                ms.append(m_star)
                progress.advance(m_task)
        costhetas = []
        phis = []
        while len(costhetas) < n:
            costheta_star = r().uniform(-1, 1)
            phi_star = r().uniform(-np.pi, np.pi)
            if w_bkg(costheta_star, phi_star) >= r().uniform(0, w_bkg_max):
                costhetas.append(costheta_star)
                phis.append(phi_star)
                progress.advance(w_task)
        ts = []
        while len(ts) < n:
            t_star = r().uniform(bounds['t_min'], bounds['t_max'])
            if t_bkg(t_star) >= r().uniform(0, t_bkg_max):
                ts.append(t_star)
                progress.advance(t_task)
        gs = []
        while len(gs) < n:
            g_star = r().uniform(bounds['g_min'], bounds['g_max'])
            if g_bkg(g_star) >= r().uniform(0, g_bkg_max):
                gs.append(g_star)
                progress.advance(g_task)
        return [
            Event(m, costheta, phi, t, g)
            for m, costheta, phi, t, g in zip(ms, costhetas, phis, ts, gs)
        ]


# Functions to parallelize the generation of signal and background events if producing a large sample
def gen_event_partial(n, seed):
    rng = np.random.default_rng(seed)  # Initialize random seed for each process

    events = []
    for _ in range(n):
        ms, costhetas, phis, ts, gs = [], [], [], [], []

        # Generate mass
        while len(ms) < 1:
            m_star = rng.uniform(bounds['m_min'], bounds['m_max'])
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
            t_star = rng.uniform(bounds['t_min'], bounds['t_max'])
            if t_sig(t_star) >= rng.uniform(0, t_sig_max):
                ts.append(t_star)

        # Generate g
        while len(gs) < 1:
            g_star = rng.uniform(bounds['g_min'], bounds['g_max'])
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
            m_star = rng.uniform(bounds['m_min'], bounds['m_max'])
            if m_bkg(m_star, truths['b']) >= rng.uniform(0, m_bkg_max):
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
            t_star = rng.uniform(bounds['t_min'], bounds['t_max'])
            if t_bkg(t_star, truths['tau_bkg']) >= rng.uniform(0, t_bkg_max):
                ts.append(t_star)

        # Generate g for background
        while len(gs) < 1:
            g_star = rng.uniform(bounds['g_min'], bounds['g_max'])
            if g_bkg(g_star, truths['sigma_bkg']) >= rng.uniform(0, g_bkg_max):
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
