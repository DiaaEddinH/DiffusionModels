import yaml
from glob import glob
from pathlib import Path

# Load experiment base name from YAML
# config_path = "configs/GMODEL_EBM_config.yaml"
# with open(config_path) as f:
#     cfg = yaml.safe_load(f)

# Example: snakemake -j 8 --config experiment=configs/GMODEL_EBM_config.yaml
if "experiment" not in config:
    raise ValueError("Please provide --config experiment=<path_to_yaml>")

config_path = config["experiment"]
with open(config_path) as f:
    cfg = yaml.safe_load(f)

EXPERIMENT_BASE = cfg["file"]  # e.g., "GMODEL_EBM"

# Detect all sample stems
SAMPLES = [
    Path(f).stem.replace("_samples", "")
    for f in glob(f"data/samples/{EXPERIMENT_BASE}_*_samples.npy")
]

rule all:
	input:
		f"data/processed/{EXPERIMENT_BASE}_moments_random_effects.npz",
		f"data/processed/{EXPERIMENT_BASE}_cumulants_random_effects.npz",
		f"data/processed/{EXPERIMENT_BASE}_other_moments_random_effects.npz"

# ------------------------------
# Rules
# ------------------------------

rule compute_moments:
    input:
        lambda wildcards: f"data/samples/{wildcards.sample}_samples.npy"
    output:
        "data/processed/{sample}_moments.npz"
    threads: 2
    resources:
        mem_mb=2000
    script:
        "scripts/compute_moments.py"


rule compute_moments:
    input:
        lambda wildcards: f"data/samples/{wildcards.sample}_samples.npy"
    output:
        "data/processed/{sample}_other_moments.npz"
    threads: 2
    resources:
        mem_mb=2000
    script:
        "scripts/compute_other_moments.py"



rule compute_cumulants:
    input:
        lambda wildcards: f"data/samples/{wildcards.sample}_samples.npy"
    output:
        "data/processed/{sample}_cumulants.npz"
    threads: 2
    resources:
        mem_mb=2000
    script:
        "scripts/compute_cumulants.py"

rule random_effects_moments:
    input:
        expand("data/processed/{s}_moments.npz", s=SAMPLES)
    output:
        f"data/processed/{EXPERIMENT_BASE}_moments_random_effects.npz"
    threads: 2
    resources:
        mem_mb=2000
    script:
        "scripts/compute_random_effects.py"

rule random_effects_other_moments:
    input:
        expand("data/processed/{s}_other_moments.npz", s=SAMPLES)
    output:
        f"data/processed/{EXPERIMENT_BASE}_other_moments_random_effects.npz"
    threads: 2
    resources:
        mem_mb=2000
    script:
        "scripts/compute_random_effects.py"

rule random_effects_cumulants:
    input:
        expand("data/processed/{s}_cumulants.npz", s=SAMPLES)
    output:
        f"data/processed/{EXPERIMENT_BASE}_cumulants_random_effects.npz"
    threads: 2
    resources:
        mem_mb=2000
    script:
        "scripts/compute_random_effects.py"
