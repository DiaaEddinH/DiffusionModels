import numpy as np
from diffusion_models.stats.cumulants import calc_cumulants

sample = np.load(snakemake.input[0])
val, err = calc_cumulants(sample)
np.savez(snakemake.output[0], val=val, err=err)
