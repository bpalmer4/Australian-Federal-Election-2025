"""Tools for doing Bayesian aggregation of polls"""
from typing import Any, Iterable, Optional, Mapping

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

import plotting
from common import MIDDLE_DATE


# --- Data preparation
def prepare_data_for_analysis(
    df: pd.DataFrame,
    column: str,
    right_anchor: Optional[tuple[pd.Period, float]] = None,
    left_anchor: Optional[tuple[pd.Period, float]] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Prepare a dataframe column for Bayesian analysis.
    Returns a python dict with all the necessary values within."""

    # make sure data is in date order
    assert df[
        MIDDLE_DATE
    ].is_monotonic_increasing, "Data must be in ascending date order"

    # get our zero centered observations
    y = df[column].dropna()  # assume data in percentage points (0..100)
    centre_offset = -y.mean()
    zero_centered_y = y + centre_offset
    n_polls = len(zero_centered_y)

    # get our day-to-date mapping
    day_zero = left_anchor[0] if left_anchor is not None else df[MIDDLE_DATE].min()
    last_day = right_anchor[0] if right_anchor is not None else df[MIDDLE_DATE].max()
    n_days = int((last_day - day_zero) / pd.Timedelta(days=1)) + 1
    poll_date = df[MIDDLE_DATE]
    poll_day = ((df[MIDDLE_DATE] - day_zero) / pd.Timedelta(days=1)).astype(int)
    poll_day_c_ = np.c_[poll_day]  # numpy column vector of poll_days

    # sanity checks
    if left_anchor is not None:
        assert day_zero <= df[MIDDLE_DATE].min()
    if right_anchor is not None:
        assert last_day >= df[MIDDLE_DATE].max()

    # get our poll-branding information
    poll_firm = df.Brand.astype("category").cat.codes
    n_firms = len(poll_firm.unique())
    firm_map = {code: firm for firm, code in zip(df.Brand, poll_firm)}

    # Information
    if verbose:
        print(
            f"Series: {column}\n"
            f"Number of polls: {n_polls}\n"
            f"Number of days: {n_days}\n"
            f"Number of pollsters: {n_firms}\n"
            f"Centre offset: {centre_offset}\n"
            f"Pollster map: {firm_map}\n"
            f"Polling days:\n{poll_day.values}\n"
        )
    return locals().copy()  # okay, this is a bit of a hack


# --- Bayesian models and model components
# - Gaussian Random Walk
def _guess_start(inputs: dict[str, Any]) -> np.float64:
    """Guess a starting point for the random walk,
    based on early poll results."""

    guess_first_n_polls = 5  # guess based on first n polls
    educated_guess = inputs["zero_centered_y"][
        : min(guess_first_n_polls, len(inputs["zero_centered_y"]))
    ].mean()
    return educated_guess


def temporal_model(
    inputs: dict[str, Any], model: pm.Model, **kwargs
) -> pt.TensorVariable:
    """The temporal (hidden daily voting intention) model component.
    Used in Gaussian Random Walk (GRW) models.
    Note: setting the innovation through a prior
    often results in a model that needs many samples to
    overcome poor traversal of the posterior."""

    # check for specified parameters
    innovation = kwargs.get("innovation", None)

    # construct the temporal model
    with model:
        if innovation is None:
            # this does not mix well, but I haven't found anything that does
            beta_hint = {"alpha": 3, "beta": 0.5}
            print(f"innovation InverseGamma prior: {beta_hint}")
            innovation = pm.InverseGamma("innovation", **beta_hint)

        init_guess_sigma = 5.0  # SD for initial guess
        start_dist = pm.Normal.dist(mu=_guess_start(inputs), sigma=init_guess_sigma)

        voting_intention = pm.GaussianRandomWalk(
            "voting_intention",
            mu=0,  # no drift in this model
            sigma=innovation,
            init_dist=start_dist,
            steps=inputs["n_days"],
        )
    return voting_intention


def house_effects_model(inputs: dict[str, Any], model: pm.Model) -> pt.TensorVariable:
    """The house effects model. This model component is used with
    both the GRW and Gaussian Process (GP) models."""

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


def core_likelihood(
    inputs: dict[str, Any],
    model: pm.Model,
    voting_intention: pt.TensorVariable,
    house_effects: pt.TensorVariable,
    **kwargs,
) -> None:
    """Likelihood for both GP and GRW models."""

    # check for specified parameters
    likelihood = kwargs.get("likelihood", "Normal")
    nu = kwargs.get("nu", None)
    sigma_likelihood = kwargs.get("sigma_likelihood", None)
    grw = kwargs.get("grw", True)

    # construct likelihood
    with model:
        if sigma_likelihood is None:
            sigma_likelihood_hint = {"sigma": 5}
            print(f"sigma_likelihood HalfNormal prior: {sigma_likelihood_hint}")
            sigma_likelihood = pm.HalfNormal(
                "sigma_likelihood", **sigma_likelihood_hint
            )
        mu = (
            voting_intention[inputs["poll_day"]] + house_effects[inputs["poll_firm"]]
            if grw
            else voting_intention + house_effects[inputs["poll_firm"]]
        )
        common_args = {
            "name": "observed_polls",
            "mu": mu,
            "sigma": sigma_likelihood,
            "observed": inputs["zero_centered_y"],
        }

        match likelihood:
            case "Normal":
                pm.Normal(**common_args)

            case "StudentT":
                if nu is None:
                    nu_hint = {"alpha": 2, "beta": 0.1}
                    print(f"nu Gamma prior: {nu_hint}")
                    nu = pm.Gamma("nu", **nu_hint) + 1

                pm.StudentT(
                    **common_args,
                    nu=nu,
                )


def grw_model(inputs: dict[str, Any], **kwargs) -> pm.Model:
    """PyMC model for pooling/aggregating voter opinion polls,
    using a Gaussian random walk (GRW). Model assumes poll data
    (in percentage points) has been zero-centered (by
    subtracting the mean for the series). Model assumes house
    effects sum to zero."""

    model = pm.Model()
    voting_intention = temporal_model(inputs, model, **kwargs)
    house_effects = house_effects_model(inputs, model)
    core_likelihood(inputs, model, voting_intention, house_effects, **kwargs)
    return model


def grw_observational_model(
    inputs: dict[str, Any],
    model: pm.Model,
    voting_intention: pt.TensorVariable,
    house_effects: pt.TensorVariable,
    **kwargs,
) -> None:
    """Observational model (likelihood) component.
    Used with GRW models."""

    core_likelihood(inputs, model, voting_intention, house_effects, **kwargs)

    with model:
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


def grw_la_model(inputs: dict[str, Any], **kwargs) -> pm.Model:
    """A more nuanced left_anchor model which captures the
    systemic polling error in the posterior. But otherwise,
    much the same as the left anchor version of grw_model()."""

    assert inputs["left_anchor"] is not None
    model = pm.Model()
    voting_intention = temporal_model(inputs, model, **kwargs)

    with model:
        # different house-effects model
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

    grw_observational_model(inputs, model, voting_intention, house_effects, **kwargs)
    return model


# - Gaussian Process model ...
def gp_prior(
    inputs: dict[str, Any],
    model: pm.Model,
    **kwargs,
) -> pt.TensorVariable:
    """Construct the Gaussian Process (GP) latent variable model prior.
    The prior reflects voting intention on specific polling days.

    Note: Reasonably smooth looking plots only emerge with a lenjth_scale
    greater than (say) 15. Divergences occur when eta resolves as being
    close to zero, (which is obvious when you think about it, but also
    harder to avoid with series that are fairly flat). To address
    these sampling issues, we give the gamma distribution a higher alpha,
    as the mean of the gamma distribution is a/b. And we truncate eta to
    well avoid zero (noting eta is squared before being multiplied by the
    covariance matrix).

    Also note: for quick test runs, length_scale and eta can be fixed
    to (say) 20 and 1.2 respectively. With both specified, the model runs
    in around 1.4 seconds. With one or both unspecified, it takes about
    7-8 minutes per run."""

    # check for specified parameters
    length_scale = kwargs.get("length_scale", None)
    eta = kwargs.get("eta", None)

    # construct the gaussian prior
    with model:
        if length_scale is None:
            gamma_hint = {"alpha": 20, "beta": 1}
            print(f"length_scale Gamma prior: {gamma_hint}")
            length_scale = pm.Gamma("length_scale", **gamma_hint)
        if eta is None:
            halfnormal_hint = {"sigma": 5}
            print(f"eta HalfNormal prior: {halfnormal_hint}")
            eta = pm.HalfNormal("eta", **halfnormal_hint)
        cov = eta**2 * pm.gp.cov.ExpQuad(1, length_scale)
        gp = pm.gp.Latent(cov_func=cov)
        voting_intention = gp.prior("voting_intention", X=inputs["poll_day_c_"])
    return voting_intention


def gp_model(inputs: dict[str, Any], **kwargs) -> pm.Model:
    """PyMC model for pooling/aggregating voter opinion polls,
    using a Gaussian Process (GP). Note: **kwargs allows one to
    pass length_scale and eta to gp_prior() and/or approach,
    nu and sigma to gp_likelihood()."""

    model = pm.Model()
    voting_intention = gp_prior(inputs, model, **kwargs)
    house_effects = house_effects_model(inputs, model)
    core_likelihood(inputs, model, voting_intention, house_effects, **kwargs)
    return model


# --- Condition the model on the data
def report_glitches(idata: az.InferenceData) -> str:
    """Display some quick summary diagnostics."""

    glitches = []
    summary = az.summary(idata)

    max_r_hat = 1.01
    statistic = summary.r_hat.max()
    print(text := f"max_r_hat: {statistic}")
    if statistic > max_r_hat:
        glitches.append(text)

    min_ess = 500
    statistic = summary[["ess_tail", "ess_bulk"]].min().min()
    print(text := f"min_ess: {statistic}")
    if statistic < min_ess:
        glitches.append(text)

    try:
        diverging_count = int(np.sum(idata.sample_stats.diverging))
    except (ValueError, AttributeError):  # No sample_stats, or no `.diverging`
        diverging_count = 0
    print(text := f"{diverging_count} divergences")
    if diverging_count:
        glitches.append(text)

    return (
        "" if not glitches else f'Issues with Bayesian sampling: {", ".join(glitches)}'
    )


def draw_samples(
    model: pm.Model, n_cores: int = 10, **kwargs
) -> tuple[az.InferenceData, str]:
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
    glitches = report_glitches(idata)
    return (idata, glitches)


#  --- Plotting support ...
def generate_model_map(
    model: pm.Model,
    filemame_stem: str,
    model_dir: str = "../model-images/",
    display_images: bool = False,
) -> None:
    """Generate a map image for the model."""

    gv = pm.model_to_graphviz(model)
    gv.render(
        format="png",
        filename=(f"{model_dir}" f"model-graphviz-{filemame_stem.replace('/', '')}"),
    )
    if display_images:
        display(gv)


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
            "title": f"{title_stem}: {variable}",
            "show": False,
            **plotting.footers,
        }
        kwargs_copy, defaults = plotting.generate_defaults(kwargs, defaults)
        plotting.finalise_plot(axes, **defaults, **kwargs_copy)


PERCENTS = [2.5, 12.5, 25, 37.5, 47.5]
_COLORS = [(p - min(PERCENTS)) / (max(PERCENTS) - min(PERCENTS)) for p in PERCENTS]
MIN_COLOR = 0.25
COLOR_FRACS = [c * (1.0 - MIN_COLOR) + MIN_COLOR for c in _COLORS]


def _get_var(var_name: str, idata: az.InferenceData) -> pd.DataFrame:
    """Extract the chains/draws for a specified var_name."""

    return (
        az.extract(idata, var_names=var_name)
        .transpose("sample", ...)
        .to_dataframe()[var_name]
        .unstack(level=2)
        .T
    )


def plot_voting(inputs, idata, palette, **kwargs) -> pd.Series:
    """Plot voting intention from both GRW and GP models."""

    # get the relevant data as a DataFrame
    df = _get_var("voting_intention", idata) - inputs["centre_offset"]
    plot_some_samples: int = kwargs.get("plot_some_samples", 0)
    df.index = (
        [x.to_timestamp() for x in inputs["poll_date"]]
        if len(df) == inputs["n_polls"]
        else pd.date_range(
            start=inputs["day_zero"].to_timestamp(), periods=inputs["n_days"] + 1
        )
    )

    # make the plot
    _, ax = plotting.initiate_plot()
    cmap = plt.get_cmap(palette)
    x = df.index
    # plot samples
    if plot_some_samples:
        for y in df.sample(n=plot_some_samples, axis=1):
            ax.plot(
                df.index,
                df[y],
                color=cmap(0.9),
                lw=0.25,
                alpha=0.5,
                label=None,
                zorder=0,
            )
    # plot quantiles
    for i, p in enumerate(PERCENTS):
        quants = p, 100 - p
        label = f"{quants[1] - quants[0]}% HDI"
        lower, upper = [df.quantile(q=q / 100.0, axis=1) for q in quants]
        color = COLOR_FRACS[i]
        ax.fill_between(
            df.index,
            upper,
            lower,
            color=cmap(color),
            alpha=0.5,
            label=label,
            zorder=i + 1,
        )
    # plot data points
    plotting.add_data_points_by_pollster(
        ax,
        df=inputs["df"],
        column=inputs["column"],
        p_color="#555555",
    )
    # annotate the min, max and end
    middle = df.quantile(q=0.5, axis=1)
    plotting.annotate_min_max_end(ax, middle)
    # make it pretty
    lo, hi = ax.get_ylim()
    halfway = (hi - lo) / 2 + lo
    space = "lower" if middle.iloc[0] > halfway else "upper"
    defaults = {  # default arguments for finalise_plot()
        "ylabel": "Per cent",
        "xlabel": None,
        "show": False,
        "y50": True,
        "legend": plotting.LEGEND_SET
        | {"fontsize": "xx-small", "loc": f"{space} left", "ncols": 2},
        "concise_dates": True,
        **plotting.footers,
    }
    kwargs_copy, defaults = plotting.generate_defaults(kwargs, defaults)
    plotting.finalise_plot(ax, **defaults, **kwargs_copy)

    return middle


def plot_house_effects(
    inputs: dict[str, Any],
    idata: az.InferenceData,
    palette: str,
    **kwargs,
) -> None:
    """Plot the House effects for both GRW and GP models."""

    # get the data as a DataFrame
    df = _get_var("house_effects", idata).rename(index=inputs["firm_map"])
    middle = df.quantile(0.5, axis=1).sort_values()
    df = df.reindex(middle.index)

    # plot quantiles, with text over median
    _, ax = plotting.initiate_plot()
    cmap = plt.get_cmap(palette)
    for i, p in enumerate(PERCENTS):
        quants = p, 100 - p
        label = f"{quants[1] - quants[0]}% HDI"
        lower, upper = [df.quantile(q=q / 100.0, axis=1) for q in quants]
        color = COLOR_FRACS[i]
        ax.barh(
            df.index,
            width=upper - lower,
            left=lower,
            color=cmap(color),
            label=label,
            zorder=i + 1,
        )
    for index, value in middle.items():
        ax.text(
            s=f"{value:.1f}",
            x=value,
            y=index,
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            zorder=i + 2,
        )
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


def plot_std_set(
    inputs: dict[str, Any],
    idata: az.InferenceData,
    title_stem: str,
    glitches: str,
    show: bool,
) -> pd.Series:  # returns the median sample
    """Produce the standard set of plots for a Bayesian
    Analysis. Return the median sample."""

    core_plot_args: Mapping = {
        "show": show,
        "rheader": None if not glitches else {"rheader": glitches},
    }

    # plot single variables from the models
    plot_univariate(
        idata,
        var_names=(
            "length_scale",
            "eta",
            "nu",
            "sigma_likelihood",
            "innovation",
            "alpha",
            "beta",
            "systemic_poll_error",
        ),
        title_stem=title_stem,
        **core_plot_args,
    )

    # plot voting intention over time
    palette = plotting.get_gp_palette(title_stem)
    middle = plot_voting(
        inputs,
        idata,
        palette,
        title=f"Vote share: {title_stem}",
        **core_plot_args,
    )

    # plot house effects
    plot_house_effects(
        inputs,
        idata,
        palette,
        title=f"House Effects: {title_stem}",
        **core_plot_args,
    )

    return middle
