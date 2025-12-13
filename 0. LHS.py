import numpy as np
import pandas as pd
from scipy.stats import lognorm, norm
from scipy.stats import qmc

seed = 42

CLIMATE_MODEL = [
    "ACCESS-CM2|used", "ACCESS-CM2|used4", "ACCESS-CM2|used5",
    "AWI-CM-1-1-MR|used",
    "BCC-CSM2-MR|used",
    "GFDL-ESM4|used",
    "KACE-1-0-G|used", "KACE-1-0-G|used2", "KACE-1-0-G|used3",
    "MIROC6|used", "MIROC6|used2", "MIROC6|used3",
    "MPI-ESM1-2-HR|used", "MPI-ESM1-2-HR|used2"
]

# 1) Original technology cost ranges (EUR/kW)
techs = {
    "PV": (200, 320),
    "Wind": (650, 1010),
    "Li-ion storage": (154, 437),
    "Electrolyzer": (250, 400),
}

# 2) Exchange rate assumption: 1 EUR = 1.1 USD
rate = 1.1
techs_usd = {k: (low * rate, high * rate) for k, (low, high) in techs.items()}

# 3) Fit lognormal parameters using the 10th and 90th percentiles
params = {}
p_low, p_high = 0.10, 0.90
z_low, z_high = norm.ppf(p_low), norm.ppf(p_high)

for tech, (low_usd, high_usd) in techs_usd.items():
    # Solve for (mu, sigma) in log-space based on two quantiles
    sigma = (np.log(high_usd) - np.log(low_usd)) / (z_high - z_low)
    mu = np.log(low_usd) - sigma * z_low
    params[tech] = (mu, sigma)

# 4) Latin Hypercube Sampling (LHS) in unit hypercube
n_samples = 1000
sampler = qmc.LatinHypercube(d=len(params), seed=seed)
u = sampler.random(n=n_samples)  # shape = (n_samples, n_techs)

rng = np.random.RandomState(seed)

# 5) Map uniform samples to lognormal distributions via inverse CDF (PPF)
samples = {}
for i, (tech, (mu, sigma)) in enumerate(params.items()):
    dist = lognorm(s=sigma, scale=np.exp(mu))
    # Avoid hitting exactly 0 or 1 due to numerical issues
    ui = np.clip(u[:, i], 1e-6, 1 - 1e-6)
    samples[tech] = dist.ppf(ui)

# 6) Assemble table and assign a climate model label per sample
df_samples = pd.DataFrame(samples)
df_samples["CLIMATE_MODEL"] = rng.choice(CLIMATE_MODEL, size=len(df_samples))

df_samples.sort_values(by=["CLIMATE_MODEL"], inplace=True)
df_samples.to_csv("cost_proj_samples.csv", index=False)

df_samples.head()
