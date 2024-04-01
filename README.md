# About
This Python program contains a simple analysis and comparison of multiple statistical weighting methods, including inPlot/sPlot[^1] and Q-Factors[^2]. It is intended to show the need for a correction in the Q-Factors method, which occasionally fails because it does not use the proper event weighting scheme.

# Usage
1. Clone the repository: `git clone git@github.com:denehoffman/qfactors_splot.git`
2. Install the required libraries: `pip install -r requirements.txt`
3. Run the analysis script: `python analysis.py`
4. Profit?

# Disclaimer
This analysis is part of an ongoing project which will eventually be published, please don't scoop me, I'll know it and raise hell :)

# References
[^1]: M. Pivk and F.R. Le Diberder. “sPlot: A statistical tool to unfold data
distributions”. In: Nuclear Instruments and Methods in Physics Research
Section A: Accelerators, Spectrometers, Detectors and Associated Equip-
ment 555 (1-2 Dec. 2005), pp. 356–369. issn: 01689002. doi: 10.1016/j.
nima.2005.08.106. [arXiv:physics/0402083](https://arxiv.org/abs/physics/0402083)

[^2]: M Williams, M Bellis, and C A Meyer. “Multivariate side-band subtraction
using probabilistic event weights”. In: Journal of Instrumentation 4 (10
Oct. 2009), P10003–P10003. issn: 1748-0221. doi: 10.1088/1748- 0221/
4/10/P10003. [arXiv:0809.2548](https://arxiv.org/abs/0809.2548)
