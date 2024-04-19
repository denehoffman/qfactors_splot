from __future__ import annotations

import numpy as np
import scipy.optimize as opt
from iminuit import Minuit, cost
from rich.progress import track
from scipy.integrate import quad
from sklearn.neighbors import NearestNeighbors

from sqfactors import bounds, console, truths
from sqfactors.event import (
    Event,
    g_sig,
    m_bkg,
    m_sig,
    t_sig,
    w_sig,
)
from sqfactors.plot import plot_qfactor_fit
from sqfactors.utils import Result

# voigt_norm = quad(lambda x: voigt_profile((x - truths["m_omega"]), truths["sigma"], truths["m_omega"] * truths["G_omega"] / 2), m_min, bounds["m_max"])


# Calculate K-nearest neighbors for a given set of points
def k_nearest_neighbors(x, k):
    neighbors = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(x)
    _, indices = neighbors.kneighbors(x)
    return indices  # includes the point itself + k nearest neighbors


def calculate_local_density_knn(events, phase_space, metric="euclidean"):
    """
    Calculate the KNN based on local density for each event.
    The function returns indices of events in the neighborhood for each event.
    """
    # Calculate pairwise distances
    nbrs = NearestNeighbors(n_neighbors=len(events), algorithm="auto", metric=metric).fit(phase_space)
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
    return [indices[i, :k] for i, k in enumerate(sorted_k)]


def calculate_radius_neighbors(_events, phase_space, radius, metric="euclidean"):
    """
    Calculate neighbors within a specified radius for each event.
    """
    nbrs = NearestNeighbors(radius=radius, algorithm="auto", metric=metric).fit(phase_space)
    _distances, indices = nbrs.radius_neighbors(phase_space)

    # Convert sparse matrix to list of lists for indices
    return [list(ind) for ind in indices]


# Define a class for weighted unbinned negative log-likelihood calculation
class WeightedUnbinnedNLL:
    @staticmethod
    def _safe_log(y: np.ndarray) -> np.ndarray:
        return np.log(y + 1e-323)

    @staticmethod
    def _unbinned_nll_weighted(y: np.ndarray, w: np.ndarray) -> float:
        """Calculate the weighted unbinned negative log-likelihood"""
        return -np.sum(w * WeightedUnbinnedNLL._safe_log(y), dtype=float)

    def __init__(self, data: np.ndarray, model, weights: np.ndarray | None = None):
        self.weights = np.ones(data.shape[0]) if weights is None else weights
        self.data = data
        self.model = model

    def __call__(self, params, *args) -> float:
        """Evaluate the weighted unbinned NLL for given parameters"""
        y = self.model(self.data, *params, *args)
        if np.any(y < 0):
            return 1e20  # temporary fix...
        return WeightedUnbinnedNLL._unbinned_nll_weighted(y, self.weights)

    def fit(self, p0: list[float], *args, **kwargs):  # noqa: ARG002
        """Perform minimization to find the best-fit parameters"""
        return opt.minimize(lambda x, *args: self.__call__(x, *args), p0, **kwargs)


# Functions to plot event distributions and fits


def calculate_sideband_weights(events: list[Event]) -> np.ndarray:
    ms = np.array([e.mass for e in events])
    left_cut = truths["m_omega"] - 3 * truths["G_omega"]
    right_cut = truths["m_omega"] + 3 * truths["G_omega"]

    def model(x: np.ndarray, z, b, *args) -> np.ndarray:  # noqa: ARG001
        return z * m_sig(x) + (1 - z) * m_bkg(x, b)

    c = cost.UnbinnedNLL(ms, model)
    # 100% signal starting condition
    m_1 = Minuit(c, z=1.0, b=truths["b"])
    m_1.limits["z"] = (0, 1)
    m_1.migrad()
    # 100% background starting condition
    m_2 = Minuit(c, z=0.0, b=truths["b"])
    m_2.limits["z"] = (0, 1)
    m_2.migrad()
    # 50% signal / 50% background starting condition
    m_3 = Minuit(c, z=0.5, b=truths["b"])
    m_3.limits["z"] = (0, 1)
    m_3.migrad()
    fits = [m_1, m_2, m_3]
    nlls = np.array([m.fval for m in fits])
    best_fit = fits[np.argmin(nlls)]

    left_area = quad(lambda x: m_bkg(x, best_fit.values[1]), bounds["m_min"], left_cut)[0]  # noqa: PD011
    center_area = quad(lambda x: m_bkg(x, best_fit.values[1]), left_cut, right_cut)[0]  # noqa: PD011
    right_area = quad(lambda x: m_bkg(x, best_fit.values[1]), right_cut, bounds["m_max"])[0]  # noqa: PD011

    weights = np.ones_like(ms)
    mask_sidebands = (ms < left_cut) | (ms > right_cut)
    weights[mask_sidebands] = -center_area / (left_area + right_area)
    return weights


def calculate_inplot(events: list[Event]) -> np.ndarray:
    ms = np.array([e.mass for e in events])

    def model(x: np.ndarray, z, b, *args) -> np.ndarray:  # noqa: ARG001
        return z * m_sig(x) + (1 - z) * m_bkg(x, b)

    def inplot(x: np.ndarray, z, b, *args) -> np.ndarray:  # noqa: ARG001
        return (z * m_sig(x)) / (z * m_sig(x) + (1 - z) * m_bkg(x, b))

    inplot_weights = []
    c = cost.UnbinnedNLL(ms, model)
    # 100% signal starting condition
    m_1 = Minuit(c, z=1.0, b=truths["b"])
    m_1.limits["z"] = (0, 1)
    m_1.migrad()
    # 100% background starting condition
    m_2 = Minuit(c, z=0.0, b=truths["b"])
    m_2.limits["z"] = (0, 1)
    m_2.migrad()
    # 50% signal / 50% background starting condition
    m_3 = Minuit(c, z=0.5, b=truths["b"])
    m_3.limits["z"] = (0, 1)
    m_3.migrad()
    fits = [m_1, m_2, m_3]
    nlls = np.array([m.fval for m in fits])
    best_fit = fits[np.argmin(nlls)]
    inplot_weights = inplot(ms, *best_fit.values)  # noqa: PD011
    return np.array(inplot_weights)


def calculate_q_factors(
    events: list[Event],
    phase_space: np.ndarray,
    name: str,
    num_knn: int,
    use_density_knn=False,
    use_radius_knn=None,
    plot_indices: list[int] | None = None,
    directory="study",
) -> tuple[np.ndarray, np.ndarray]:
    ms = np.array([e.mass for e in events])

    knn_indices = []

    if use_density_knn:
        tag = "_density"
        # Calculate KNN based on local density
        with console.status("Calculating K-Nearest Neighbors Based on Local Density"):
            knn_indices = calculate_local_density_knn(events, phase_space)
    elif use_radius_knn:
        tag = "_radius"
        radius = float(use_radius_knn)
        with console.status("Calculating Radius Neighbors"):
            knn_indices = calculate_radius_neighbors(events, phase_space, radius)
    else:
        tag = ""
        # Standard KNN calculation
        with console.status("Calculating K-Nearest Neighbors"):
            indices = k_nearest_neighbors(phase_space, num_knn)
            # Exclude the first index for each event since it is the event itself
            knn_indices = [index_set[1:] for index_set in indices]

    def model(x: np.ndarray, z, b, *args) -> np.ndarray:  # noqa: ARG001
        return z * m_sig(x) + (1 - z) * m_bkg(x, b)

    def inplot(x, z, b, *args) -> float:  # noqa: ARG001
        return (z * m_sig(x)) / (z * m_sig(x) + (1 - z) * m_bkg(x, b))

    q_factors = []
    sq_factors = []
    for i in track(range(len(events)), description="Calculating Q-Factors"):
        indices = knn_indices[i]
        c = cost.UnbinnedNLL(ms[indices], model)
        # 100% signal starting condition
        m_1 = Minuit(c, z=1.0, b=truths["b"])
        m_1.limits["z"] = (0, 1)
        m_1.migrad()
        # 100% background starting condition
        m_2 = Minuit(c, z=0.0, b=truths["b"])
        m_2.limits["z"] = (0, 1)
        m_2.migrad()
        # 50% signal / 50% background starting condition
        m_3 = Minuit(c, z=0.5, b=truths["b"])
        m_3.limits["z"] = (0, 1)
        m_3.migrad()
        fits = [m_1, m_2, m_3]
        nlls = np.array([m.fval for m in fits])
        best_fit = fits[np.argmin(nlls)]
        n_sig = len(ms[indices]) * best_fit.values[0]  # noqa: PD011
        n_bkg = len(ms[indices]) * (1 - best_fit.values[0])  # noqa: PD011
        b = best_fit.values[1]  # noqa: PD011
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
            console.print("Encountered a singular matrix, applying regularization.")
            epsilon = 1e-5  # Small regularization term
            Vmat_inv_reg = Vmat_inv + epsilon * np.eye(Vmat_inv.shape[0])
            V = np.linalg.inv(Vmat_inv_reg)

        sq_factors.append(
            (V[0, 0] * m_sig(ms[i]) + V[0, 1] * m_bkg(ms[i], b)) / (n_sig * m_sig(ms[i]) + n_bkg * m_bkg(ms[i], b))
        )
        q_factors.append(inplot(ms[i], *best_fit.values))  # noqa: PD011
        if plot_indices and i in plot_indices:
            plot_qfactor_fit(
                ms[i],
                ms[indices],
                z_fit=best_fit.values[0],  # noqa: PD011
                b_fit=best_fit.values[1],  # noqa: PD011
                event_index=i,
                qfactor_type=f"{name}{tag}",
            )
    return np.array(q_factors), np.array(sq_factors)


def calculate_splot_weights(events: list[Event], sig_frac_init=0.5, b_init=0.5) -> np.ndarray:
    """Calculate sPlot weights for distinguishing signal from background"""
    ms = np.array([e.mass for e in events])  # Extracting the mass values from events

    def model(x: np.ndarray, sig_frac, b, *args) -> np.ndarray:  # noqa: ARG001
        return sig_frac * m_sig(x) + (1 - sig_frac) * m_bkg(x, b)

    # Performing the fit
    c = cost.UnbinnedNLL(ms, model)
    mi = Minuit(c, sig_frac=sig_frac_init, b=b_init)
    mi.limits["sig_frac"] = (0, 1)  # Ensuring physical bounds
    mi.limits["b"] = (0, 1)
    mi.migrad()

    # Extract fit results for signal and background contributions
    n_sig = len(events) * mi.values["sig_frac"]  # noqa: PD011
    n_bkg = len(events) * (1 - mi.values["sig_frac"])  # noqa: PD011
    b = mi.values["b"]  # noqa: PD011

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

    def model(angles: np.ndarray, p00: float, p1n1: float, p10: float) -> np.ndarray | float:
        return w_sig(angles[:, 0], angles[:, 1], p00, p1n1, p10)

    angles = np.array([[e.costheta, e.phi] for e in events])
    wunll = WeightedUnbinnedNLL(angles, model, weights=weights)
    # return wunll.fit([truths["p00"], truths["p1n1"], truths["p10"]])

    def _cost(p00: float, p1n1: float, p10: float, *args) -> float:  # noqa: ARG001
        return wunll([p00, p1n1, p10])

    m = Minuit(_cost, p00=truths["p00"], p1n1=truths["p1n1"], p10=truths["p10"])
    m.migrad()
    m.minos(cl=1)
    return m


def fit_t(events: list[Event], weights: np.ndarray | None = None):
    """Perform a weighted fit to the t distribution of events to estimate the t parameter"""
    ts = np.array([e.t for e in events])
    wunll = WeightedUnbinnedNLL(ts, t_sig, weights=weights)

    def _cost(tau_sig: float, *args) -> float:  # noqa: ARG001
        return wunll([tau_sig])

    m = Minuit(_cost, tau_sig=truths["tau_sig"])
    m.migrad()
    m.minos(cl=1)
    return m


def fit_g(events: list[Event], weights: np.ndarray | None = None):
    """Perform a weighted fit to the g distribution of events to estimate the g parameter"""
    gs = np.array([e.g for e in events])
    wunll = WeightedUnbinnedNLL(gs, g_sig, weights=weights)

    def _cost(sigma_sig: float, *args) -> float:  # noqa: ARG001
        return wunll([sigma_sig])

    m = Minuit(_cost, sigma_sig=truths["sigma_sig"])
    m.migrad()
    m.minos(cl=1)
    return m


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
    return np.where(total_densities > 0, signal_densities / total_densities, 0)


def get_results(method: str, events, weights=None) -> Result:
    mi_angles = fit_angles(events, weights=weights)
    mi_t = fit_t(events, weights=weights)
    mi_g = fit_g(events, weights=weights)
    return Result(
        method,
        [
            ("p00", mi_angles.values["p00"], mi_angles.errors["p00"]),  # noqa: PD011
            ("p1n1", mi_angles.values["p1n1"], mi_angles.errors["p1n1"]),  # noqa: PD011
            ("p10", mi_angles.values["p10"], mi_angles.errors["p10"]),  # noqa: PD011
            ("tau_sig", mi_t.values["tau_sig"], mi_t.errors["tau_sig"]),  # noqa: PD011
            ("sigma_sig", mi_g.values["sigma_sig"], mi_g.errors["sigma_sig"]),  # noqa: PD011
        ],
    )
