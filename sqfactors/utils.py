from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from rich.table import Table

from sqfactors import truths

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult


class Result:
    def __init__(
        self, method: str, iteration: str, contents: list[tuple[str, float, float]], valid: bool
    ):
        self.method = method
        self.iteration = iteration
        self.contents = contents
        self.valid = valid

    @staticmethod
    def get_deviation(
        fit: float,
        error: float,
        true: float,
    ) -> Literal['good', 'bad', 'worst']:
        if abs(fit - true) > error * 5:
            return 'worst'
        if abs(fit - true) > error * 3:
            return 'bad'
        return 'good'

    def to_latex(self) -> str:
        colors = {'good': 'black', 'bad': 'red', 'worst': 'red'}
        out = self.method
        for variable, value, error in self.contents:
            truth = truths[variable]
            dev = Result.get_deviation(value, error, truth)
            out += rf' & {{\color{{{colors[dev]}}}{value}\pm{error}}}'
        out += '//'
        return out

    def to_rich(self) -> list[str]:
        colors = {'good': 'black', 'bad': 'yellow', 'worst': 'red'}
        out = [self.method]
        for variable, value, error in self.contents:
            truth = truths[variable]
            dev = Result.get_deviation(value, error, truth)
            out += [f'[{colors[dev]}]{value:.3f}±{error:.3f}[/]']
        return out

    def to_tsv(self) -> str:
        return (
            self.iteration
            + '\t'
            + self.method
            + '\t'
            + '\t'.join(f'{value}\t{error}' for (_, value, error) in self.contents)
            + '\t'
            + str(self.valid)
        )


class Results:
    def __init__(self, results: list[Result] | None = None):
        self.results = results if results else []

    @staticmethod
    def tsv_header() -> str:
        return f"Iteration\tMethod\tp00\tp00 Error\tp1n1\tp1n1 Error\tp10\tp10 Error\ttau_sig\ttau_sig Error\tsigma_sig\tsigma_sig Error\tValid\nN/A\tTruth\t{truths['p00']:.3f}\t0.000\t{truths['p1n1']:.3f}\t0.000\t{truths['p10']:.3f}\t0.000\t{truths['tau_sig']:.3f}\t0.000\t{truths['sigma_sig']:.3f}\t0.000\tTrue\n"

    def add_row(self, result: Result):
        self.results.append(result)

    def __rich_console__(self, _console: Console, _options: ConsoleOptions) -> RenderResult:
        t = Table(title='Fit Results')
        t.add_column('Weighting Method')
        t.add_column('ρ⁰₀₀')
        t.add_column('ρ⁰₁,₋₁')
        t.add_column('Re[ρ⁰₁₀]')
        t.add_column('τ')
        t.add_column('σ')  # noqa: RUF001
        t.add_row(
            'Truth',
            f"{truths['p00']:.3f}",
            f"{truths['p1n1']:.3f}",
            f"{truths['p10']:.3f}",
            f"{truths['tau_sig']:.3f}",
            f"{truths['sigma_sig']:.3f}",
            end_section=True,
        )
        for result in self.results:
            t.add_row(*result.to_rich())
        yield t

    def as_latex(self) -> str:
        out = rf"""
\begin{{table}}
\centering
\begin{{tabular}}{{lccccc}}\toprule
Weighting Method & $\rho^0_{{00}}$ & $\rho^0_{{1,-1}}$ & $\Re[\rho^0_{{10}}]$ & $\tau$ & $\sigma$ \\ \midrule
\textbf{{Truth}} & \textbf{{{truths['p00']:.3f}}} & \textbf{{{truths['p1n1']:.3f}}} & \textbf{{{truths['p10']:.3f}}} & \textbf{{{truths['tau_sig']:.3f}}} & \textbf{{{truths['sigma_sig']:.3f}}} \\ \midrule
            """
        out += '\n'.join(result.to_latex() for result in self.results)
        out += r"""
\bottomrule
\end{tabular}
\caption{Fit results from each weighting method. Results which deviate more than $3\sigma$ are highlighted red.}
\label{table:density_fit_results}
\end{table}
            """
        return out

    def as_tsv(self, header: bool = True) -> str:
        if header:
            return self.tsv_header() + '\n'.join(result.to_tsv() for result in self.results) + '\n'
        return '\n'.join(result.to_tsv() for result in self.results) + '\n'
