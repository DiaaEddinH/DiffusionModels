import numpy as np
from diffusion_models.effects.random_effects import RandomEffectsAnalyser

analyzer = RandomEffectsAnalyser.from_file_paths(snakemake.input)
# snakemake.input is already the correct files for moments or cumulants
Y_hat, sigma_stat, sigma_sys, sigma_tot = analyzer.analyze()

np.savez(
    snakemake.output[0],
    Y_hat=Y_hat,
    sigma_stat=sigma_stat,
    sigma_sys=sigma_sys,
    sigma_tot=sigma_tot,
)
