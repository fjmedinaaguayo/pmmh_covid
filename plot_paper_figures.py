"""
Generate the PMMH figures shown in the paper for one or more chains.

For each chain it writes: R_t, beta_t, weekly cases, susceptibles, parameter
posteriors (with priors), trace plots, ACFs and the pairwise posterior matrix.

Usage:
    python plot_paper_figures.py --chain chains/pmmh_eta7.npz --tag eta7
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from statsmodels.tsa.stattools import acf

COVID_DATA_FILE = "./data/covid_df.feather"
POPULATION_FILE = "./data/INEGI_2020_State_Population.feather"
BURN_IN_FRACTION = 0.2
ACF_LAGS = 1000
DPI = 300
PARAM_FIGSIZE = (5, 4)
PARAM_TICK_FONTSIZE = 12
PARAM_LABEL_FONTSIZE = 15
PARAM_LEGEND_FONTSIZE = 12

PARAMS = ["kappa", "gamma", "sigma", "nu"]
# (histogram color index, KDE/line color index) in the Paired palette
PARAM_COLORS = {"kappa": (0, 1), "gamma": (2, 3), "sigma": (4, 5), "nu": (6, 7)}
LEGEND_STYLE = dict(
    fontsize=12,
    frameon=True,
    fancybox=True,
    framealpha=1,
    edgecolor="black",
    facecolor="white",
)


def load_inputs(chain_file):
    pop = pd.read_feather(POPULATION_FILE)
    population = pop["population"].sum()

    covid = pd.read_feather(COVID_DATA_FILE)
    weekly = (
        covid.groupby(pd.Grouper(key="date", freq="W-SUN"))
        .agg({"confirmed_cases": "sum"})
        .reset_index()
    )

    chain = np.load(chain_file)
    prior_spec = json.loads(str(chain["prior_spec_json"].item()))

    metadata_file = os.path.splitext(chain_file)[0] + "_metadata.json"
    with open(metadata_file, encoding="utf-8") as f:
        metadata = json.load(f)
    eta = float(metadata["model_specification"]["observation_model"]["underreporting_factor"])

    return population, weekly["date"], chain, prior_spec, eta


def band_plot(dates, paths, color, ylabel, out_file, median_label="Posterior Median",
              outer=(5, 95), hline=None, observed=None, legend_anchor=(0.4, 0.9)):
    colors = sns.color_palette("Paired")
    lo, hi = np.percentile(paths, outer, axis=0)
    q25, q75 = np.percentile(paths, [25, 75], axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, np.median(paths, axis=0), color=colors[1], linestyle=":", label=median_label)
    plt.fill_between(dates, lo, hi, color=colors[color], alpha=0.3,
                     label=f"{outer[1] - outer[0]}% Credibility Interval")
    plt.fill_between(dates, q25, q75, color=colors[color], alpha=1, label="50% Credibility Interval")
    if hline is not None:
        value, label, style = hline
        plt.axhline(value, color="black", alpha=0.5, linestyle="--", linewidth=style, label=label)
    if observed is not None:
        plt.plot(dates, observed, color=colors[3], label="Infected Cases", marker="o",
                 linewidth=0, markersize=3)

    plt.xlim(dates.iloc[0], dates.iloc[-1])
    plt.legend(bbox_to_anchor=legend_anchor, **LEGEND_STYLE)
    plt.ylabel(ylabel, fontsize=12)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.grid(visible=True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI, bbox_inches="tight")
    plt.close()


def prior_pdf_and_grid(spec, samples):
    if spec["dist"] == "beta":
        x = np.linspace(1e-6, 1 - 1e-6, 1000)
        return x, stats.beta(spec["alpha"], spec["beta"]).pdf(x)
    dist = stats.gamma(a=spec["shape"], scale=spec["scale"])
    upper = max(np.percentile(samples, 99.5) * 1.2, 1.0, dist.ppf(0.995))
    x = np.linspace(1e-6, upper, 1000)
    return x, dist.pdf(x)


def posterior_plot(samples, name, spec, out_file):
    colors = sns.color_palette("Paired")
    c_hist, c_kde = PARAM_COLORS[name]
    plot_range = (0.0, 2.0) if name == "sigma" else None

    plt.figure(figsize=PARAM_FIGSIZE)
    plt.hist(samples, bins=20, range=plot_range, color=colors[c_hist], density=True,
             edgecolor="none", alpha=0.5, label="Post. Histogram")
    sns.kdeplot(samples, linewidth=2, color=colors[c_kde], label="Post. KDE")

    x, pdf = prior_pdf_and_grid(spec, samples)
    if plot_range is not None:
        mask = (x >= plot_range[0]) & (x <= plot_range[1])
        x, pdf = x[mask], pdf[mask]
        plt.xlim(*plot_range)
    plt.plot(
        x, pdf,
        linewidth=2,
        color="grey",
        linestyle="--",
        label="Prior dens.",
    )

    plt.xlabel(name, fontsize=PARAM_LABEL_FONTSIZE)
    plt.ylabel("Density", fontsize=PARAM_LABEL_FONTSIZE)
    plt.tick_params(axis="both", which="major", labelsize=PARAM_TICK_FONTSIZE)
    plt.legend(fontsize=PARAM_LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI, bbox_inches="tight")
    plt.close()


def trace_plot(chain_samples, idx, name, burn_in, out_file):
    colors = sns.color_palette("Paired")
    plt.figure(figsize=PARAM_FIGSIZE)
    plt.plot(chain_samples[:, idx], color=colors[PARAM_COLORS[name][0]], alpha=0.5)
    plt.axvline(burn_in, color="black", linestyle="--", label="Burn-in Threshold", linewidth=1)
    plt.tick_params(axis="both", which="major", labelsize=PARAM_TICK_FONTSIZE)
    plt.xlabel("Iteration", fontsize=PARAM_LABEL_FONTSIZE)
    plt.ylabel(name, fontsize=PARAM_LABEL_FONTSIZE)
    plt.legend(fontsize=PARAM_LEGEND_FONTSIZE)
    if name != "nu":
        plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI, bbox_inches="tight")
    plt.close()


def acf_plot(post_burn_samples, name, out_file):
    colors = sns.color_palette("Paired")
    c_bar, c_line = PARAM_COLORS[name]
    autocorr = acf(post_burn_samples, nlags=ACF_LAGS, fft=True)

    plt.figure(figsize=PARAM_FIGSIZE)
    plt.plot(autocorr, color=colors[c_line], linewidth=1.8)
    plt.bar(range(ACF_LAGS + 1), autocorr, color=colors[c_bar], width=1, alpha=0.5)
    plt.xlim(0, ACF_LAGS)
    plt.tick_params(axis="both", which="major", labelsize=PARAM_TICK_FONTSIZE)
    plt.xlabel("Lag", fontsize=PARAM_LABEL_FONTSIZE)
    plt.ylabel("Autocorrelation", fontsize=PARAM_LABEL_FONTSIZE)
    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI, bbox_inches="tight")
    plt.close()


def pairwise_plot(post_burn_samples, out_file, max_points=6000):
    colors = sns.color_palette("Paired")
    labels = ["kappa", "gamma", "sigma", "nu"]
    df = pd.DataFrame(post_burn_samples[:, :4], columns=labels)
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    def padded_limits(series, lo=0.5, hi=99.5, pad=0.10):
        lo_val, hi_val = np.percentile(series, [lo, hi])
        margin = (hi_val - lo_val) * pad
        return lo_val - margin, hi_val + margin

    limits = {c: padded_limits(df[c]) for c in labels}

    def upper_corr(x, y, color=None, **kwargs):
        ax = plt.gca()
        r = np.corrcoef(x, y)[0, 1]
        ax.annotate(f"r = {r:.2f}", xy=(0.5, 0.5), xycoords="axes fraction",
                    ha="center", va="center", fontsize=PARAM_LABEL_FONTSIZE, color="black")
        ax.grid(False)

    grid = sns.PairGrid(df, diag_sharey=False)
    grid.map_lower(sns.kdeplot, fill=True, levels=8, thresh=0.05, cmap="mako")
    grid.map_diag(sns.kdeplot, fill=True, linewidth=1.3, color=colors[1])
    grid.map_upper(upper_corr)

    n = len(labels)
    for r, row_axes in enumerate(grid.axes):
        for c, ax in enumerate(row_axes):
            ax.set_xlim(limits[labels[c]])
            ax.set_ylim(limits[labels[r]])
            ax.tick_params(axis="both", which="major", labelsize=PARAM_TICK_FONTSIZE)
            ax.tick_params(axis="x", which="both", direction="out", bottom=True,
                           labelbottom=(r == n - 1))
            ax.tick_params(axis="y", which="both", direction="out", left=True,
                           labelleft=(c == 0))
            if r != n - 1:
                ax.set_xlabel("")
            if c != 0:
                ax.set_ylabel("")
    for c, ax in enumerate(grid.axes[-1]):
        ax.set_xlabel(labels[c], fontsize=PARAM_LABEL_FONTSIZE)
    for r, ax in enumerate(grid.axes[:, 0]):
        ax.set_ylabel(labels[r], fontsize=PARAM_LABEL_FONTSIZE)

    grid.figure.suptitle("Posterior Pairwise Matrix (Lower: 2D KDE)", y=1.02,
                         fontsize=PARAM_LABEL_FONTSIZE + 1)
    plt.tight_layout()
    grid.figure.savefig(out_file, dpi=DPI, bbox_inches="tight")
    plt.close(grid.figure)


def make_figures(chain_file, tag, out_dir):
    plt.style.use("seaborn-v0_8-ticks")
    os.makedirs(out_dir, exist_ok=True)
    population, dates, chain, prior_spec, eta = load_inputs(chain_file)

    samples = chain["samples"]
    beta_paths = chain["beta_paths"]
    infection_paths = chain["infection_paths"]
    states_paths = chain["states_paths"]
    y_obs = chain["Y_obs"]
    t_obs = len(y_obs)
    dates = dates[:t_obs]

    burn_in = int(BURN_IN_FRACTION * len(samples))
    beta_vals = beta_paths[burn_in:]
    gamma_vals = samples[burn_in:, [1]]
    susceptible = states_paths[burn_in:, 0, :]

    def out(name):
        return os.path.join(out_dir, f"{name}_{tag}.png")

    # Time-series figures
    rt = (beta_vals / gamma_vals) * (susceptible / population)
    band_plot(dates, rt, 0, r"$R_t$", out("rt"), outer=(5, 95),
              hline=(1, r"$R_t = 1$ Threshold", 1.5))
    band_plot(dates, beta_vals, 0, r"$\beta_t$", out("betat"))
    band_plot(dates, infection_paths[burn_in:] / eta, 0, "Registered Weekly Cases",
              out("weekly_cases"), observed=y_obs)
    band_plot(dates, susceptible, 0, "Susceptible Population", out("susceptibles"),
              hline=(0.95 * population, "95% Population Threshold", 1),
              legend_anchor=(0.95, 0.95))

    # Parameter figures
    for idx, name in enumerate(PARAMS):
        spec = prior_spec[name]
        posterior_plot(samples[burn_in:, idx], name, spec, out(f"posterior_{name}"))
        trace_plot(samples, idx, name, burn_in, out(f"trace_{name}"))
        acf_plot(samples[burn_in:, idx], name, out(f"acf_{name}"))

    pairwise_plot(samples[burn_in:], out("correlation"))
    print(f"[{tag}] figures written to {out_dir} (eta={eta}, weeks={t_obs})")


def main():
    parser = argparse.ArgumentParser(description="Create paper figures from a PMMH chain.")
    parser.add_argument("--chain", required=True, help="Path to the PMMH .npz chain.")
    parser.add_argument("--tag", required=True, help="Suffix for output files, e.g. eta7.")
    parser.add_argument("--out-dir", default="./figures")
    args = parser.parse_args()
    make_figures(args.chain, args.tag, args.out_dir)


if __name__ == "__main__":
    main()
