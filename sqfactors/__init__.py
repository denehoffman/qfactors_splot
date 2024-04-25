import numpy as np
from rich.console import Console

console = Console()

# Define constants to generate MC according to https://arxiv.org/abs/0804.3382
bounds = {
    'm_min': 0.68,
    'm_max': 0.88,
    't_min': 0.0,
    't_max': 2.0,
    'g_min': -1.8,
    'g_max': 1.8,
    'b_min': 0.0,
    'b_max': 2.0 / (0.88 - 0.68),
}
truths = {
    'b': 3.0,
    'm_omega': 0.78256,
    'G_omega': 0.00844,
    'sigma': 0.005,
    'p00': 0.65,
    'p1n1': 0.05,
    'p10': 0.10,
    'tau_sig': 0.11,
    'tau_bkg': 0.43,
    'sigma_sig': 0.13,
    'sigma_bkg': 0.56,
}

rng = np.random.default_rng(0)


def r():
    global rng
    return rng


def set_seed(n):
    global rng
    rng = np.random.default_rng(n)
