"""Tools for doing Bayesian aggregation of polls"""
from typing import Optional, Any, Iterable

import pandas as pd
import numpy as np
import arviz as az
import pymc as pm
import pytensor.tensor as pt

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

    # get our zero centered observations
    y = df[column].dropna()
    centre_offset = -y.mean()
    zero_centered_y = y + centre_offset
    n_polls = len(zero_centered_y)  ###

    # get our day-date mapping
    day_zero = left_anchor[0] if left_anchor is not None else df[MIDDLE_DATE].min()
    last_day = right_anchor[0] if right_anchor is not None else df[MIDDLE_DATE].max()
    n_days = int((last_day - day_zero) / pd.Timedelta(days=1)) + 1
    poll_day = ((df[MIDDLE_DATE] - day_zero) / pd.Timedelta(days=1)).astype(int)

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
    assumed_sample_size = 1000
    measurement_error_sd = np.sqrt((50 * 50) / assumed_sample_size)

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
        guess_sigma = 10  # percent-points SD for initial guess
        start_dist = pm.Normal.dist(mu=guess_start(inputs), sigma=guess_sigma)
        innovation = 0.175  # set by hand or by distribution ...
        # innovation = pm.TruncatedNormal("innovation", lower=0.01, upper=0.5, mu=0.175, sigma=0.1)
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
        house_effect_sigma = 5  # assume larger house effects possible
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
                sigma=0.001,
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


def the_model(inputs: dict[str, Any]) -> pm.Model:
    """PyMC model for pooling/aggregating voter opinion polls,
    using a Gaussian raandom walk. Model assumes poll data
    (in percentage points) has been zero-centered (by
    subtracting the mean for the series). Model can be
    left or right anchored. Unanchored model assumes house
    effects sum to zero."""

    model = pm.Model()
    voting_intention = temporal_model(inputs, model)
    house_effects = house_effects_model(inputs, model)
    observational_model(inputs, model, voting_intention, house_effects)
    return model


def la_model(inputs: dict[str, Any]) -> pm.Model:
    """Special left_anchor model which captures systemic polling error."""

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


def draw_samples(
    model: pm.Model, n_cores: int = 10, **kwargs
) -> az.InferenceData:  # tuple[az.InferenceData, pd.DataFrame]:
    """Draw samples from the posterior distribution (ie. run the model)."""

    with model:
        trace = pm.sample(
            progressbar=True,
            cores=n_cores,
            chains=n_cores,
            return_inferencedata=True,
            **kwargs,
        )
        az.plot_trace(trace)
    return trace


#  --- Plotting support ...
def get_var_as_frame(inferencedata, variable_name):
    """Return a dataframe - column(s) are variables - rows are draws"""
    return pd.DataFrame(
        inferencedata.posterior.stack(draws=("chain", "draw"))[variable_name].values.T
    )


def quants_and_mean(frame, quants):
    """Return a DataFrame - columns are quants and mean - rows are variable"""
    results = frame.quantile(quants).T
    results["mean"] = frame.mean()
    return results


def get_quant_iterator(quants):
    """get iterator for paired quantiles, with matching labels"""
    length = len(quants)
    assert length > 0, "Quantile list should not be empty"
    assert length % 2 == 0, "Quantile list must have an even number of members"
    assert pd.Series(
        quants
    ).is_monotonic_increasing, "Qantiles must be ascending ordered"
    middle = int(length / 2)
    start = quants[:middle]
    stop = quants[-1 : (middle - 1) : -1]
    label = (
        ((pd.Series(stop) - pd.Series(start)) * 100).round(0).astype(int).astype(str)
        + "% HDI"
    ).to_list()
    return zip(start, stop, label)


def plot_aggregation(
    inputs: dict[str, Any],
    trace: az.InferenceData,
    line_color: str,
    point_color: str,
    **kwargs,
) -> pd.Series:
    """Plot the pooled poll. Return the mean series."""

    # get the data
    grw = get_var_as_frame(trace, "voting_intention") - inputs["centre_offset"]
    grw.columns = pd.period_range(
        start=inputs["day_zero"], periods=inputs["n_days"] + 1
    )
    grw_summary = quants_and_mean(grw, QUANTS)

    # plot
    _, axes = plotting.initiate_plot()
    grw_summary["mean"].plot(
        lw=2.0,
        c=line_color,
        label="Mean Voting Intention",
        ax=axes,
    )
    alpha = 0.1
    for start, stop, label in get_quant_iterator(QUANTS):
        axes.fill_between(
            x=grw_summary.index,
            y1=grw_summary[start],
            y2=grw_summary[stop],
            color=line_color,
            alpha=alpha,
            label=label,
        )
        alpha += 0.1

    plotting.annotate_endpoint(axes, grw_summary["mean"])

    plotting.add_data_points_by_pollster(
        ax=axes,
        df=inputs["df"],
        column=inputs["column"],
        p_color=point_color,
    )

    defaults = {  # default arguments for finalise_plot()
        "title": f"Voting Intention: {inputs['column']}",
        "xlabel": None,
        "ylabel": "Per cent",
        "y50": True,
        "show": False,
        "legend": plotting.LEGEND_SET | {"ncols": 3, "fontsize": "xx-small"},
        **plotting.footers,
    }
    kwargs_copy, defaults = plotting.generate_defaults(kwargs, defaults)
    plotting.finalise_plot(axes, **defaults, **kwargs_copy)

    return grw_summary["mean"]


def plot_house_effects(
    inputs: dict[str, Any],
    trace: az.InferenceData,
    line_color: str,
    point_color: str,
    **kwargs,
) -> None:
    """Plot the House effects."""

    # get the relevant data
    h_eff = get_var_as_frame(trace, "house_effects")
    h_eff.columns = h_eff.columns.map(inputs["firm_map"])
    h_eff_summary = quants_and_mean(h_eff, QUANTS)
    h_eff_summary = h_eff_summary.sort_values(by="mean")

    # and plot it
    _, axes = plotting.initiate_plot()
    alpha = 0.1
    for start, stop, label in get_quant_iterator(QUANTS):
        axes.barh(
            h_eff_summary.index,
            width=h_eff_summary[stop] - h_eff_summary[start],
            left=h_eff_summary[start],
            color=line_color,
            alpha=alpha,
            label=label,
        )
        alpha += 0.1
    axes.scatter(
        h_eff_summary["mean"],
        h_eff_summary.index,
        s=50,
        marker="o",
        color=point_color,
        label="Mean estimate",
    )
    axes.tick_params(axis="y", labelsize="small")
    axes.axvline(0, c="darkgrey", lw=1)
    defaults = {  # default arguments for finalise_plot()
        "xlabel": "Relative effect (percentage points)",
        "ylabel": None,
        "show": False,
        "legend": plotting.LEGEND_SET | {"ncols": 1, "fontsize": "xx-small"},
        **plotting.footers,
    }
    kwargs_copy, defaults = plotting.generate_defaults(kwargs, defaults)
    plotting.finalise_plot(axes, **defaults, **kwargs_copy)


def plot_univariate(
    inputs: dict[str, Any],
    trace: az.InferenceData,
    var_names: str | Iterable[str],
    hdi_prob: float = 0.80,
    title_stem: str = "",
    **kwargs,
) -> None:
    """Plot univariate posterior variables."""

    if isinstance(var_names, str):
        var_names = (var_names,)

    for variable in var_names:
        if variable not in trace.posterior:
            continue
        axes = az.plot_posterior(trace, var_names=variable, hdi_prob=hdi_prob)
        defaults = {  # default arguments for finalise_plot()
            "xlabel": None,
            "ylabel": None,
            "title":f"{title_stem}{variable}",
            "show": False,
            **plotting.footers,
        }
        kwargs_copy, defaults = plotting.generate_defaults(kwargs, defaults)
        plotting.finalise_plot(axes, **defaults, **kwargs_copy)
