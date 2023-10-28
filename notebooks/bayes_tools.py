"""Tools for doing Bayesian aggregation of polls"""
from typing import Optional

import pandas as pd
import numpy as np
import arviz as az
import pymc as pm

import plotting


QUANTS = (0.025, 0.1, 0.25, 0.75, 0.9, 0.975)


def prepare_data_for_analysis(df: pd.DataFrame, column: str) -> tuple:
    """Prepare a column of a dataframe for Bayesian analysis."""

    # make sure data is in date order
    assert df[
        "Mean Date"
    ].is_monotonic_increasing, "Data must be in ascending date order"

    # get our zero centered observations
    y = df[column].dropna()
    centre_offset = -y.mean()
    zero_centered_y = y + centre_offset
    n_polls = len(zero_centered_y)  ###

    # get our day-date mapping
    day_zero = df["Mean Date"].min()
    n_days = int((df["Mean Date"].max() - day_zero) / pd.Timedelta(days=1)) + 1
    poll_day = ((df["Mean Date"] - day_zero) / pd.Timedelta(days=1)).astype(int)

    # get our poll-branding information
    poll_firm = df.Brand.astype("category").cat.codes
    n_firms = len(poll_firm.unique())
    firm_map = {code: firm for firm, code in zip(df.Brand, poll_firm)}

    # measurement error
    assumed_sample_size = 1000
    measurement_error_sd = np.sqrt((50 * 50) / assumed_sample_size)

    # Information
    print(
        f"Series: {column}\n"
        f"Number of polls: {n_polls}\n"
        f"Number of days: {n_days}\n"
        f"Number of pollsters: {n_firms}\n"
        f"Centre offset: {centre_offset}\n"
    )

    return (
        zero_centered_y,
        centre_offset,
        n_polls,
        n_days,
        day_zero,
        poll_day,
        poll_firm,
        firm_map,
        n_firms,
        measurement_error_sd,
    )


def define_zs_model(  # zs = zero-sum (house effects)
    n_firms: int,
    n_days: int,
    poll_day: pd.Series,  # of int, length is number of polls
    poll_brand: pd.Series,  # of int, length is number of polls
    zero_centered_y: pd.Series,  # of float, length is number of polls
    measurement_error_sd: float,
) -> pm.Model:
    """PyMC model for pooling/aggregating voter opinion polls.
    Model assumes poll data (in percentage points)
    has been zero-centered (by subtracting the mean for
    the series). Model assumes that House Effects sum to zero."""

    model = pm.Model()
    with model:
        # --- Temporal voting-intention model
        # Guess a starting point for the random walk
        guess_first_n_polls = 5  # guess based on first n polls
        guess_sigma = 15  # percent-points SD for initial guess
        educated_guess = zero_centered_y[
            : min(guess_first_n_polls, len(zero_centered_y))
        ].mean()
        start_dist = pm.Normal.dist(mu=educated_guess, sigma=guess_sigma)
        # Establish a Gaussian random walk ...
        daily_innovation = 0.20  # from experience ... daily change in VI
        voting_intention = pm.GaussianRandomWalk(
            "voting_intention",
            mu=0,  # no drift in model
            sigma=daily_innovation,
            init_dist=start_dist,
            steps=n_days,
        )

        # --- House effects model
        house_effect_sigma = 15  # assume big house effects possible
        house_effects = pm.ZeroSumNormal(
            "house_effects", sigma=house_effect_sigma, shape=n_firms
        )

        # --- Observational model (likelihood)
        polling_observations = pm.Normal(
            "polling_observations",
            mu=voting_intention[poll_day.values] + house_effects[poll_brand.values],
            sigma=measurement_error_sd,
            observed=zero_centered_y,
        )
    return model


def define_ra_model(  # ra = right anchored
    n_firms: int,
    n_days: int,
    poll_day: pd.Series,  # of int, l
    poll_brand: pd.Series,  # of int, length is number poll
    zero_centered_y: pd.Series,  # of float, length is number of polls
    measurement_error_sd: float,
    right_anchor: float,
) -> pm.Model:
    """PyMC model for pooling/aggregating voter opinion polls.
    Model assumes poll data (in percentage points)
    has been zero-centered (by subtracting the mean for
    the series). Model anchors the last day of the random
    walk to the election result."""

    model = pm.Model()
    with model:
        # --- Temporal voting-intention model
        # Guess a starting point for the random walk
        guess_first_n_polls = 5  # guess based on mean of first n polls
        guess_sigma = 15  # percent-points SD for initial guess
        educated_guess = zero_centered_y[
            : min(guess_first_n_polls, len(zero_centered_y))
        ].mean()
        start_dist = pm.Normal.dist(mu=educated_guess, sigma=guess_sigma)
        # Establish a Gaussian random walk ...
        daily_innovation = 0.20  # from experience ... daily change in VI
        voting_intention = pm.GaussianRandomWalk(
            "voting_intention",
            mu=0,  # no drift in model
            sigma=daily_innovation,
            init_dist=start_dist,
            steps=n_days,
        )

        # --- House effects model
        house_effect_sigma = 15  # assume big house effects possible
        poll_error_sigma = 15  # assume polls collective can be off by a lot
        systemic_poll_error = pm.Normal(
            "systemic_poll_error", mu=0, sigma=poll_error_sigma
        )
        zero_sum_house_effects = pm.ZeroSumNormal(
            "zero_sum_house_effects", sigma=house_effect_sigma, shape=n_firms
        )
        house_effects = pm.Deterministic(
            "house_effects",
            var=zero_sum_house_effects + systemic_poll_error,
        )

        # --- observational model (likelihood)
        # OM1 - observed polls
        polling_observations = pm.Normal(
            "polling_observations",
            mu=voting_intention[poll_day.values] + house_effects[poll_brand.values],
            sigma=measurement_error_sd,
            observed=zero_centered_y,
        )
        # OM2 - observed election result
        # --- there should be a better way to anchor.
        # --- NEED TO THINK ABOUT THIS SOME MORE.
        election_observation = pm.Normal(
            "election_observation",
            mu=voting_intention[n_days - 1],
            sigma=0.0000001,  # near zero
            observed=right_anchor,
        )
    return model


def draw_samples(
    model: pm.Model,
    draws: int = 1_000,
    tune: int = 1_000,
    n_cores: int = 10,
) -> tuple[az.InferenceData, pd.DataFrame]:
    """Draw samples from the posterior distribution."""

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            progressbar=True,
            cores=n_cores,
            chains=n_cores,
            return_inferencedata=True,
        )
        summary = az.summary(trace)  # used below
        az.plot_trace(trace)

    return trace, summary


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
    trace,
    df,
    column,
    day_zero,
    n_days,
    centre_offset,
    point_color,
    line_color,
    **kwargs,
) -> None:
    """Plot the pooled poll."""

    # get the data
    grw = get_var_as_frame(trace, "voting_intention") - centre_offset
    grw.columns = pd.period_range(start=day_zero, periods=n_days + 1)
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
        df=df,
        column=column,
        p_color=point_color,
    )

    defaults = {  # default arguments for finalise_plot()
        "title": f"Voting Intention: {column}",
        "xlabel": None,
        "ylabel": "Per cent",
        "y50": True,
        "show": False,
        "legend": plotting.LEGEND_SET | {"ncols": 3, "fontsize": "xx-small"},
        **plotting.footers,
    }
    kwargs_copy, defaults = plotting.generate_defaults(kwargs, defaults)
    plotting.finalise_plot(axes, **defaults, **kwargs_copy)


def plot_house_effects(
    trace,
    column,
    brand_mapping,
    point_color,
    line_color,
    **kwargs,
) -> None:
    """Plot the House effects."""

    # get the relevant data
    h_eff = get_var_as_frame(trace, "house_effects")
    h_eff.columns = h_eff.columns.map(brand_mapping)
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
        "title": f"House effects: {column}",
        "xlabel": "Relative effect (percentage points)",
        "ylabel": None,
        "show": False,
        "legend": plotting.LEGEND_SET | {"ncols": 1, "fontsize": "xx-small"},
        **plotting.footers,
    }
    kwargs_copy, defaults = plotting.generate_defaults(kwargs, defaults)
    plotting.finalise_plot(axes, **defaults, **kwargs_copy)
