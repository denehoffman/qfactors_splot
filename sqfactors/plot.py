from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.stats import ks_2samp

from sqfactors import bounds, console
from sqfactors.event import Event, m_bkg, m_sig

mpl.use('Agg')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
mpl.rcParams['axes.labelsize'] = 16

# Define colorscheme
RED = '#CC3311'
BLUE = '#0077BB'
PURPLE = '#AA3377'
BLACK = '#000000'
PALE_GRAY = '#DDDDDD'
DARK_GRAY = '#555555'
ERROR_RED = '#CC3311'
CMAP = 'viridis'


def plot_events(
    events: list[Event],
    signal_events: list[Event],
    weights: np.ndarray | None = None,
    filename='events.png',
    directory: str | Path = 'study',
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

    nw, bw, _ = ax[0, 0].hist(
        ms,
        bins=100,
        range=(bounds['m_min'], bounds['m_max']),
        weights=weights,
        label=weights_label,
        color=PALE_GRAY,
    )
    ax[0, 0].hist(
        ms,
        bins=100,
        range=(bounds['m_min'], bounds['m_max']),
        weights=weights,
        histtype='step',
        color=ERROR_RED,
    )
    nt, bt, _ = ax[0, 0].hist(
        ms_sig,
        bins=100,
        range=(bounds['m_min'], bounds['m_max']),
        histtype='step',
        label='Truth',
        color='black',
    )
    # ax[0, 0].bar(x=bt[:-1], height=np.abs(nw - nt), bottom=np.minimum(nw, nt), width=np.diff(bt), align='edge', lw=0, color=RED, alpha=0.3)
    ax[0, 0].set_xlabel(r'$M_{3\pi}$ (GeV/$c^2$)')
    ax[0, 0].set_ylabel(r'Counts / 0.002')
    ax[0, 0].legend(loc='upper right')
    ax[0, 1].hist2d(
        costhetas, phis, bins=(50, 70), range=[(-1, 1), (-np.pi, np.pi)], weights=weights, cmap=CMAP
    )
    ax[0, 1].set_xlabel(r'$\cos(\theta)$')
    ax[0, 1].set_ylabel(r'$\phi$')
    nw, bw, _ = ax[1, 0].hist(
        ts,
        bins=100,
        range=(bounds['t_min'], bounds['t_max']),
        weights=weights,
        label=weights_label,
        color=PALE_GRAY,
    )
    ax[1, 0].hist(
        ts,
        bins=100,
        range=(bounds['t_min'], bounds['t_max']),
        weights=weights,
        histtype='step',
        color=ERROR_RED,
    )
    nt, bt, _ = ax[1, 0].hist(
        ts_sig,
        bins=100,
        range=(bounds['t_min'], bounds['t_max']),
        histtype='step',
        label='Truth',
        color='black',
    )
    # ax[1, 0].bar(x=bt[:-1], height=np.abs(nw - nt), bottom=np.minimum(nw, nt), width=np.diff(bt), align='edge', lw=0, color=RED, alpha=0.3)
    ax[1, 0].set_xlabel('$t$ (arb)')
    ax[1, 0].set_ylabel(r'Counts / 0.02')
    ax[1, 0].legend(loc='upper right')
    nw, bw, _ = ax[1, 1].hist(
        gs,
        bins=100,
        range=(bounds['g_min'], bounds['g_max']),
        weights=weights,
        label=weights_label,
        color=PALE_GRAY,
    )
    ax[1, 1].hist(
        gs,
        bins=100,
        range=(bounds['g_min'], bounds['g_max']),
        weights=weights,
        histtype='step',
        color=ERROR_RED,
    )
    nt, bt, _ = ax[1, 1].hist(
        gs_sig,
        bins=100,
        range=(bounds['g_min'], bounds['g_max']),
        histtype='step',
        label='Truth',
        color='black',
    )
    # ax[1, 1].bar(x=bt[:-1], height=np.abs(nw - nt), bottom=np.minimum(nw, nt), width=np.diff(bt), align='edge', lw=0, color=RED, alpha=0.3)
    ax[1, 1].set_xlabel('$g$ (arb)')
    ax[1, 1].set_ylabel(r'Counts / 0.036')
    ax[1, 1].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(Path(directory).resolve() / filename, dpi=300)
    plt.close()


def plot_all_events(
    events_sig: list[Event],
    events_bkg: list[Event],
    filename='generated_data.png',
    directory: str | Path = 'study',
):
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
    _, ax = plt.subplots(nrows=3, ncols=4, figsize=(12, 9), sharey='col')

    # signal plots
    ax[0, 0].hist(
        ms_sig, bins=100, range=(bounds['m_min'], bounds['m_max']), label='signal', color=BLUE
    )
    ax[0, 0].set_xlabel(r'$M_{3\pi}$ (GeV/$c^2$)')
    ax[0, 0].set_ylabel(r'Counts / 0.002')
    ax[0, 0].legend(loc='upper right')
    ax[0, 1].hist2d(
        costhetas_sig,
        phis_sig,
        bins=(50, 70),
        range=[(-1, 1), (-np.pi, np.pi)],
        label='signal',
        cmap=CMAP,
    )
    ax[0, 1].set_xlabel(r'$\cos(\theta)$')
    ax[0, 1].set_ylabel(r'$\phi$')
    ax[0, 2].hist(
        ts_sig, bins=100, range=(bounds['t_min'], bounds['t_max']), label='signal', color=BLUE
    )
    ax[0, 2].set_xlabel('$t$ (arb)')
    ax[0, 2].set_ylabel(r'Counts / 0.02')
    ax[0, 2].legend(loc='upper right')
    ax[0, 3].hist(
        gs_sig, bins=100, range=(bounds['g_min'], bounds['g_max']), label='signal', color=BLUE
    )
    ax[0, 3].set_xlabel('$g$ (arb)')
    ax[0, 3].set_ylabel(r'Counts / 0.036')
    ax[0, 3].legend(loc='upper right')

    # background plots
    ax[1, 0].hist(
        ms_bkg, bins=100, range=(bounds['m_min'], bounds['m_max']), label='background', color=RED
    )
    ax[1, 0].set_xlabel(r'$M_{3\pi}$ (GeV/$c^2$)')
    ax[1, 0].set_ylabel(r'Counts / 0.002')
    ax[1, 0].legend(loc='upper right')
    ax[1, 1].hist2d(
        costhetas_bkg,
        phis_bkg,
        bins=(50, 70),
        range=[(-1, 1), (-np.pi, np.pi)],
        label='background',
        cmap=CMAP,
    )
    ax[1, 1].set_xlabel(r'$\cos(\theta)$')
    ax[1, 1].set_ylabel(r'$\phi$')
    ax[1, 2].hist(
        ts_bkg, bins=100, range=(bounds['t_min'], bounds['t_max']), label='background', color=RED
    )
    ax[1, 2].set_xlabel('$t$ (arb)')
    ax[1, 2].set_ylabel(r'Counts / 0.02')
    ax[1, 2].legend(loc='upper right')
    ax[1, 3].hist(
        gs_bkg, bins=100, range=(bounds['g_min'], bounds['g_max']), label='background', color=RED
    )
    ax[1, 3].set_xlabel('$g$ (arb)')
    ax[1, 3].set_ylabel(r'Counts / 0.036')
    ax[1, 3].legend(loc='upper right')

    # combined plots
    ax[2, 0].hist(
        [ms_bkg, ms_sig],
        bins=100,
        range=(bounds['m_min'], bounds['m_max']),
        stacked=True,
        color=[RED, BLUE],
        label=['background', 'signal'],
    )
    ax[2, 0].set_xlabel(r'$M_{3\pi}$ (GeV/$c^2$)')
    ax[2, 0].set_ylabel(r'Counts / 0.002')
    ax[2, 0].legend(loc='upper right')
    ax[2, 1].hist2d(
        costhetas_bkg + costhetas_sig,
        phis_bkg + phis_sig,
        bins=(50, 70),
        range=[(-1, 1), (-np.pi, np.pi)],
        cmap=CMAP,
    )
    ax[2, 1].set_xlabel(r'$\cos(\theta)$')
    ax[2, 1].set_ylabel(r'$\phi$')
    ax[2, 2].hist(
        [ts_bkg, ts_sig],
        bins=100,
        range=(bounds['t_min'], bounds['t_max']),
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
        range=(bounds['g_min'], bounds['g_max']),
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


def plot_qfactor_fit(
    mstar,
    ms,
    z_fit: float,
    b_fit: float,
    event_index: int,
    qfactor_type: str,
    directory: str | Path = 'study',
):
    # Combined model for the fit
    def model(m, z, b) -> float | np.ndarray:
        return z * m_sig(m) + (1 - z) * m_bkg(m, b)

    plt.figure(figsize=(10, 6))
    # Scatter plot of selected masses
    plt.hist(ms, bins=30, label='Selected Events Masses', density=True, color=PALE_GRAY)
    plt.hist(ms, bins=30, density=True, histtype='step', color=DARK_GRAY)

    # Plot fit components
    m_range = np.linspace(bounds['m_min'], bounds['m_max'], 1000)
    plt.axvline(mstar, ls=':', lw=2, color=BLACK, label='Event')
    plt.plot(
        m_range,
        np.array([z_fit * m_sig(m_val) for m_val in m_range]),
        ls='-',
        lw=2,
        color=BLUE,
        label='Signal Fit',
    )
    plt.plot(
        m_range,
        np.array([(1 - z_fit) * m_bkg(m_val, b_fit) for m_val in m_range]),
        ls='--',
        lw=2,
        color=RED,
        label='Background Fit',
    )
    plt.plot(
        m_range,
        np.array([model(m_val, z_fit, b_fit) for m_val in m_range]),
        ls='-',
        lw=2.5,
        color=PURPLE,
        label='Total Fit',
    )

    plt.xlabel('Mass')
    plt.ylabel('Density')
    plt.title(f'Fit for Event {event_index} and its Nearest Neighbors')
    plt.legend()
    plt.savefig(Path(directory).resolve() / f'qfactors_{qfactor_type}_{event_index}.png', dpi=300)
    plt.close()


def plot_radius_knn_visualization(
    events, selected_event_index, radius_knn, directory: str | Path = 'study'
):
    # Extract coordinates of events
    x_coords = [
        event.costheta for event in events
    ]  # Example, adjust according to your actual spatial representation
    y_coords = [event.phi for event in events]  # Example

    # Coordinates of the selected event
    selected_event_x = x_coords[selected_event_index]
    selected_event_y = y_coords[selected_event_index]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x_coords, y_coords, label='Events')
    circle = Circle(
        (selected_event_x, selected_event_y),
        radius_knn,
        color='r',
        fill=False,
        linewidth=2,
        label='Radius KNN',
    )
    plt.gca().add_patch(circle)

    # Highlight the selected event
    plt.scatter([selected_event_x], [selected_event_y], color='red', label='Selected Event')

    plt.xlabel('cosTheta')
    plt.ylabel('phi')
    plt.title(f'Visualization of Radius KNN ({radius_knn}) Neighborhood')
    plt.legend()
    plt.grid(True)  # noqa: FBT003
    plt.axis('equal')
    plt.savefig(Path(directory).resolve() / f'radius_vis_{radius_knn}.png', dpi=300)


def compare_q_factors(
    q_factors_calculated,
    q_factors_theoretical,
    title='Q-Factors Comparison',
    q_factor_type='standard',
    directory: str | Path = 'study',
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
    plt.grid(True)  # noqa: FBT003
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
    plt.savefig(
        Path(directory).resolve() / f'theory_comparison_subtract_{q_factor_type}.png', dpi=300
    )
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
