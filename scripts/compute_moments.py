import numpy as np
from diffusion_models.stats.moments import calc_moments

sample = np.load(snakemake.input[0])
val, err = calc_moments(sample)
np.savez(snakemake.output[0], val=val, err=err)
