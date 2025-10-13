import numpy as np
from src.statistics import calc_moments

sample = np.load(snakemake.input[0])
val, err = calc_moments(sample)
np.savez(snakemake.output[0], val=val, err=err)
