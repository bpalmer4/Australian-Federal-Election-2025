"""Tools for doing Bayesian aggregation of polls"""
from typing import Optional, Any, Iterable

import pandas as pd
import numpy as np
import arviz as az
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt

from common import MIDDLE_DATE
import plotting


QUANTS = (0.025, 0.1, 0.25, 0.75, 0.9, 0.975)


# --- Models for voting intention ...
def prepare_data_for_analysis(
    df: pd.DataFrame,
    column: str,
    right_anchor: Optional[tuple[pd.Period, float]] = None,
    left_anchor: Optional[tuple[pd.Period, float]] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Prepare a dataframe column for Bayesian analysis."""

    # make sure data is in date order
    assert df[
        MIDDLE_DATE
    ].is_monotonic_increasing, "Data must be in ascending date order"

    # get our zero centered observations in the range -1.0 to 1.0
    y = df[column].dropna()  # assume data in percentage points (0..100)
    centre_offset = -y.mean()
    zero_centered_y = y + centre_offset
    n_polls = len(zero_centered_y)

    # get our day-date mapping
    day_zero = left_anchor[0] if left_anchor is not None else df[MIDDLE_DATE].min()
    last_day = right_anchor[0] if right_anchor is not None else df[MIDDLE_DATE].max()
    n_days = int((last_day - day_zero) / pd.Timedelta(days=1)) + 1
    poll_date = df[MIDDLE_DATE]
    poll_day = ((df[MIDDLE_DATE] - day_zero) / pd.Timedelta(days=1)).astype(int)
    poll_day_c_ = np.c_[poll_day]  # column vector of poll_days

    # sanity checks
    if left_anchor is not None:
        assert day_zero <= df[MIDDLE_DATE].min()
    if right_anchor is not None:
        assert last_day >= df[MIDDLE_DATE].max()

    # get our poll-branding information
    poll_firm = df.Brand.astype("category").cat.codes
    n_firms = len(poll_firm.unique())
    firm_map = {code: firm for firm, code in zip(df.Brand, poll_firm)}

    # measurement error
    assumed_sample_size = 750
    measurement_error_sd = np.sqrt((50.0 * 50.0) / assumed_sample_size)

    # Information
    if verbose:
        print(
            f"Series: {column}\n"
            f"Number of polls: {n_polls}\n"
            f"Number of days: {n_days}\n"
            f"Number of pollsters: {n_firms}\n"
            f"Centre offset: {centre_offset}\n"
            f"Measurement sd: {measurement_error_sd}\n"
            f"Pollster map: {firm_map}\n"
            f"Polling days:\n{poll_day.values}\n"
        )
    return locals().copy()  # okay, this is a bit of a hack


def guess_start(inputs: dict[str, Any]) -> np.float64:
    """Guess a starting point for the random walk,
    based on early poll results."""

    guess_first_n_polls = 5  # guess based on first n polls
    educated_guess = inputs["zero_centered_y"][
        : min(guess_first_n_polls, len(inputs["zero_centered_y"]))
    ].mean()
    return educated_guess


def temporal_model(inputs: dict[str, Any], model: pm.Model) -> pt.TensorVariable:
    """The temporal (hidden daily voting intention) model.
    Note: setting the innovation through a distribution
    often results in a model that needs many samples to
    overcome poor traversal of the posterior."""

    with model:
        init_guess_sigma = 5.0  # SD for initial guess
        start_dist = pm.Normal.dist(mu=guess_start(inputs), sigma=init_guess_sigma)
        innovation = 0.175  # to set by hand, comment out the next line
        # innovation = pm.TruncatedNormal("innovation", lower=0.0001, upper=0.5,
        #                                 mu=innovation, sigma=0.1)
        voting_intention = pm.GaussianRandomWalk(
            "voting_intention",
            mu=0,  # no drift in model
            sigma=innovation,
            init_dist=start_dist,
            steps=inputs["n_days"],
        )
    return voting_intention


def house_effects_model(inputs: dict[str, Any], model: pm.Model) -> pt.TensorVariable:
    """The house effects model."""

    with model:
        house_effect_sigma = 5.0
        if inputs["right_anchor"] is None and inputs["left_anchor"] is None:
            # assume house effects sum to zero
            house_effects = pm.ZeroSumNormal(
                "house_effects", sigma=house_effect_sigma, shape=inputs["n_firms"]
            )
        else:
            house_effects = pm.Normal(
                "house_effects", sigma=house_effect_sigma, shape=inputs["n_firms"]
            )
    return house_effects


def observational_model(
    inputs: dict[str, Any],
    model: pm.Model,
    voting_intention: pt.TensorVariable,
    house_effects: pt.TensorVariable,
) -> None:
    """Observational model (likelihood)."""

    with model:
        pm.Normal(
            "polling_observations",
            mu=voting_intention[inputs["poll_day"].values]
            + house_effects[inputs["poll_firm"].values],
            sigma=inputs["measurement_error_sd"],
            observed=inputs["zero_centered_y"],
        )
        if inputs["left_anchor"] is not None:
            # --- there should be a better way to left anchor.
            # --- NEED TO THINK ABOUT THIS SOME MORE.
            pm.Normal(
                "previous_election_observation",
                mu=voting_intention[0],
                sigma=0.001,  # near zero
                observed=inputs["left_anchor"][1] + inputs["centre_offset"],
            )
        if inputs["right_anchor"] is not None:
            # --- there should be a better way to right anchor.
            # --- NEED TO THINK ABOUT THIS SOME MORE.
            pm.Normal(
                "election_observation",
                mu=voting_intention[inputs["n_days"] - 1],
                sigma=0.001,  # near zero
                observed=inputs["right_anchor"][1] + inputs["centre_offset"],
            )


def grw_model(inputs: dict[str, Any]) -> pm.Model:
    """PyMC model for pooling/aggregating voter opinion polls,
    using a Gaussian random walk. Model assumes poll data
    (in percentage points) has been zero-centered (by
    subtracting the mean for the series). Model can be
    left or right anchored. Unanchored model assumes house
    effects sum to zero.
    Note: looked at reparameterizing the data in decimal fractions,
    but that did not yield any benefits."""

    model = pm.Model()
    voting_intention = temporal_model(inputs, model)
    house_effects = house_effects_model(inputs, model)
    observational_model(inputs, model, voting_intention, house_effects)
    return model


def grw_la_model(inputs: dict[str, Any]) -> pm.Model:
    """A more nuanced left_anchor model which captures the 
    systemic polling error in the posterior. But otherwise,
    much the same as the left anchor version of grw_model()."""

    assert inputs["left_anchor"] is not None
    model = pm.Model()
    voting_intention = temporal_model(inputs, model)

    with model:
        poll_error_sigma = 2  # assume smaller systemic poll error possible
        house_effect_sigma = 5  # assume larger error on individual house effects
        systemic_poll_error = pm.Normal(
            "systemic_poll_error", mu=0, sigma=poll_error_sigma
        )
        zero_sum_house_effects = pm.ZeroSumNormal(
            "zero_sum_house_effects",
            sigma=house_effect_sigma,
            shape=inputs["n_firms"],
        )
        house_effects = pm.Deterministic(
            "house_effects",
            var=zero_sum_house_effects + systemic_poll_error,
        )

    observational_model(inputs, model, voting_intention, house_effects)
    return model


def gp_prior(inputs: dict[str, Any], model: pm.Model) -> pt.TensorVariable:
    """Construct the Latent Gaussian Process prior.
    The prior represents voting intention on a polling day."""

    with model:
        # we are going to help here ...
        length_scale = 30  # pm.Gamma("length", alpha=2, beta=1)
        eta = 1.2  # pm.HalfNormal("eta", sigma=5)
        cov = eta**2 * pm.gp.cov.ExpQuad(1, length_scale)
        gp = pm.gp.Latent(cov_func=cov)
        gauss_prior = gp.prior("gauss_prior", X=inputs["poll_day_c_"])
    return gauss_prior


def gp_likelihood(
    inputs: dict[str, Any],
    model: pm.Model,
    gauss_prior: pt.TensorVariable,
    house_effects: pt.TensorVariable,
) -> None:
    """Observational model (likelihood) - Latent Gaussian Process model."""

    with model:
        # Normal observational model
        observed_polls = pm.Normal(
            "observed_polls",
            mu=gauss_prior + house_effects[inputs["poll_firm"]],
            sigma=inputs["measurement_error_sd"],
            observed=inputs["zero_centered_y"],
        )


def gp_model(inputs: dict[str, Any]) -> pm.Model:
    """TO DO."""

    model = pm.Model()
    gauss_prior = gp_prior(inputs, model)
    house_effects = house_effects_model(inputs, model)
    gp_likelihood(inputs, model, gauss_prior, house_effects)
    return model
    

def draw_samples(
    model: pm.Model, n_cores: int = 10, **kwargs
) -> az.InferenceData:
    """Draw samples from the posterior distribution (ie. run the model)."""

    with model:
        idata = pm.sample(
            progressbar=True,
            cores=n_cores,
            chains=n_cores,
            return_inferencedata=True,
            **kwargs,
        )
        az.plot_trace(idata)
    return idata


#  --- Plotting support ...
def plot_univariate(
    idata: az.InferenceData,
    var_names: str | Iterable[str],
    hdi_prob: float = 0.80,
    title_stem: str = "",
    **kwargs,
) -> None:
    """Plot univariate posterior variables. Fail quietly if
    a variable name is not found in the posterior idata."""

    if isinstance(var_names, str):
        var_names = (var_names,)

    for variable in var_names:
        if variable not in idata.posterior:
            continue
        axes = az.plot_posterior(idata, var_names=variable, hdi_prob=hdi_prob)
        defaults = {  # default arguments for finalise_plot()
            "xlabel": None,
            "ylabel": None,
            "title": f"{title_stem}{variable}",
            "show": False,
            **plotting.footers,
        }
        kwargs_copy, defaults = plotting.generate_defaults(kwargs, defaults)
        plotting.finalise_plot(axes, **defaults, **kwargs_copy)


PERCENTS = [1, 10, 25, 40, 49]
_COLORS = [(p - min(PERCENTS)) / (max(PERCENTS) - min(PERCENTS)) for p in PERCENTS]
MIN_COLOR = 0.25
COLOR_FRACS = [c * (1.0 - MIN_COLOR) + MIN_COLOR for c in _COLORS]


def plot_voting(inputs, idata, var_name, palette, title_stem, **kwargs) -> pd.Series:
    """Plot voting intention from a Latent Gaussian Process model."""

    xdata = (
        az.extract(idata, var_names=var_name).transpose("sample", ...)
        - inputs["centre_offset"]
    )
    reframe_gpp = xdata.to_dataframe()[var_name].unstack(level=2).T
    if len(reframe_gpp) == inputs["n_polls"]:
        reframe_gpp.index = [x.to_timestamp() for x in inputs["poll_date"]]
    else:
        reframe_gpp.index = pd.date_range(
            start=inputs["day_zero"].to_timestamp(), periods=inputs["n_days"] + 1
        )
    fig, ax = plotting.initiate_plot()
    cmap = plt.get_cmap(palette)
    x = reframe_gpp.index
    # plot a sample of samples
    for y in reframe_gpp.sample(n=100, axis=1):
        ax.plot(
            x, reframe_gpp[y], color=cmap(0.9), lw=0.25, alpha=0.5, label=None, zorder=0
        )
    # plot quantiles
    for i, p in enumerate(PERCENTS):
        quants = p, 100 - p
        label = f"{quants[1] - quants[0]}% HDI"
        lower, upper = [reframe_gpp.quantile(q=x / 100.0, axis=1) for x in quants]
        color = COLOR_FRACS[i]
        ax.fill_between(
            x, upper, lower, color=cmap(color), alpha=0.5, label=label, zorder=i + 1
        )
    # plot data points
    plotting.add_data_points_by_pollster(
        ax,
        df=inputs["df"],
        column=inputs["column"],
        p_color='#555555',
    )
    middle = reframe_gpp.quantile(q=0.5, axis=1)
    plotting.annotate_min_max_end(ax, middle)
    lo, hi = ax.get_ylim()
    halfway = (hi - lo) / 2 + lo
    space = 'lower' if middle.iloc[0] > halfway else 'upper'
    plotting.finalise_plot(
        ax,
        title=f"Voting intention: {title_stem}",
        ylabel="Per cent",
        y50=True,
        legend=plotting.LEGEND_SET | {'fontsize': 'xx-small', 'loc': f'{space} left', 'ncols': 2},
        concise_dates=True,
        **kwargs,
    )
    return middle


def plot_house_effects(
    inputs: dict[str, Any],
    idata: az.InferenceData,
    palette: str,
    **kwargs,
) -> None:
    """Plot the House effects."""

    xdata = (
        az.extract(idata, var_names="house_effects").transpose("sample", ...)
    )
    reframe = xdata.to_dataframe()["house_effects"].unstack(level=2).T
    reframe = reframe.rename(index=inputs["firm_map"])
    middle = reframe.quantile(0.5, axis=1).sort_values()
    reframe = reframe.reindex(middle.index)

    fig, ax = plotting.initiate_plot()
    cmap = plt.get_cmap(palette)
    x = reframe.index

    # plot quantiles
    for i, p in enumerate(PERCENTS):
        quants = p, 100 - p
        label = f"{quants[1] - quants[0]}% HDI"
        lower, upper = [reframe.quantile(q=x / 100.0, axis=1) for x in quants]
        color = COLOR_FRACS[i]
        ax.barh(
            x,
            width=upper - lower,
            left=lower,
            color=cmap(color),
            label=label,
            zorder = i+1
        )
    for index, value in middle.items():
        ax.text(s=f"{value:.1f}", x=value, y=index, ha="center", 
                    va="center", fontsize=10, color="white", zorder = i+2)
    ax.tick_params(axis="y", labelsize="small")
    ax.axvline(0, c="darkgrey", lw=1)
    defaults = {  # default arguments for finalise_plot()
        "xlabel": "Relative effect (percentage points)",
        "ylabel": None,
        "show": False,
        "legend": plotting.LEGEND_SET | {"ncols": 1, "fontsize": "xx-small"},
        **plotting.footers,
    }
    kwargs_copy, defaults = plotting.generate_defaults(kwargs, defaults)
    plotting.finalise_plot(ax, **defaults, **kwargs_copy)
