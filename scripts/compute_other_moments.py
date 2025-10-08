import numpy as np
from src.stats import calc_other_moments

sample = np.load(snakemake.input[0])
val, err = calc_other_moments(sample)
np.savez(snakemake.output[0], val=val, err=err)
