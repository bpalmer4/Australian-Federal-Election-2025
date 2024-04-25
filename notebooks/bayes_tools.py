"""Tools for doing Bayesian aggregation of polls"""
from itertools import product
from operator import mul
from typing import Any, Iterable, Mapping, Optional

import arviz as az
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import scipy.stats as ss

from common import MIDDLE_DATE, ensure
import plotting


# --- Data preparation
def _jitter_for_unique_dates(
    df: pd.DataFrame,  # data frame
    column: str = MIDDLE_DATE,  # column name of pd.Period to jitter
    max_movement: int = 3,  # days
) -> pd.DataFrame:
    """Jitter dates to avoid conflicts. Why? Because
    dates should not be duplicated in a Gaussian Process."""

    # See if we need to do anything
    duplicated = df[df[column].duplicated(keep="first")]
    if not (how_many := len(duplicated)):
        print("No dates are duplicated.")
        return df
    print(f"There are {how_many} duplicated dates to be adjusted.")

    # change the dates for each date collision.
    df = df.copy()  # let's not change the original DataFrame
    all_dates = set(df[column])
    adjustments = tuple(
        # check for free earlier dates first ...
        pd.Timedelta(mul(*x), unit="D")
        for x in product((-1, 1), range(1, max_movement + 1))
    )
    for index, date in duplicated[column].items():
        for adjustment in adjustments:
            check = date + adjustment
            if check not in all_dates:
                df.loc[index, column] = check
                all_dates.add(check)
                break

    # collisions may remain if too many to adjust within max_movement
    ensure(not df[column].duplicated().any(), "Could not successfully jitter dates.")

    return df


TAIL_CENTRED = "tail_centred"
def prepare_data_for_analysis(
    df: pd.DataFrame,
    column: str,
    **kwargs,
) -> dict[str, Any]:
    """Prepare a dataframe column for Bayesian analysis.
    Returns a python dict with all the necessary values within."""

    # make sure data is in date order and uniquely indexed
    df = df.sort_values(MIDDLE_DATE).reset_index(drop=True)
    if kwargs.get("jitter_dates", False):
        df = _jitter_for_unique_dates(df)

    # get our zero centered observations, ignore missing data
    # assume data in percentage points (0..100)
    y = df[column].dropna()
    df = df.loc[y.index]  # for consistency, in case there were nulls in y
    if (n := kwargs.get(TAIL_CENTRED, 0)):
        # centre around last n polls 
        # to minimise mean-reversion issues with GP, but may introduce bias
        centre_offset = -y.iloc[-n:].mean()
        del kwargs[TAIL_CENTRED]
    else:
        centre_offset = -y.mean()
    zero_centered_y = y + centre_offset
    n_polls = len(zero_centered_y)

    # get our day-to-date mapping
    right_anchor: Optional[tuple[pd.Period, float]] = kwargs.get("right_anchor", None)
    left_anchor: Optional[tuple[pd.Period, float]] = kwargs.get("left_anchor", None)
    day_zero = left_anchor[0] if left_anchor is not None else df[MIDDLE_DATE].min()
    last_day = right_anchor[0] if right_anchor is not None else df[MIDDLE_DATE].max()
    n_days = int((last_day - day_zero) / pd.Timedelta(days=1)) + 1
    poll_date = df.loc[y.index, MIDDLE_DATE]
    poll_day = ((poll_date - day_zero) / pd.Timedelta(days=1)).astype(int)
    poll_day_c_ = np.c_[poll_day]  # numpy column vector of poll_days for GP

    # sanity checks - anchors must be before or after polling data
    ensure(
        (left_anchor and day_zero < df[MIDDLE_DATE].min())
        or (right_anchor and last_day > df[MIDDLE_DATE].max())
        or (not left_anchor and not right_anchor),
        "Anchors must be outside of the range of polling dates.",
    )

    # get our poll-branding information
    poll_firm = df.Brand.astype("category").cat.codes
    n_firms = len(poll_firm.unique())
    firm_map = {code: firm for firm, code in zip(df.Brand, poll_firm)}

    # Information
    if kwargs.get("verbose", False):
        print(
            f"Series: {column}\n"
            f"Number of polls: {n_polls}\n"
            f"Number of days: {n_days}\n"
            f"Number of pollsters: {n_firms}\n"
            f"Centre offset: {centre_offset}\n"
            f"Pollster map: {firm_map}\n"
            f"Polling days:\n{poll_day.values}\n"
        )

    # pop everything we need to know into a dictionary and return it
    inputs = locals().copy()  # okay, this is a bit of a hack
    del inputs["kwargs"]  # don't need the original kwargs
    return inputs


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
    Used in Gaussian Random Walk (GRW and GRWLA) models.
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
    """Likelihood for both GP and GRW models. But you
    must pass grw=False for GP analysis."""

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
    Used with GRW models. Model can be either left or right
    anchored."""

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

    ensure(inputs["left_anchor"] is not None)
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
    to (say) 40 and 1.6 respectively. With both specified, the model runs
    in around 1.4 seconds. With one or both unspecified, it takes about
    20 minutes per run, sometimes with divergences."""

    # check for specified parameters
    length_scale = kwargs.get("length_scale", None)
    eta = kwargs.get("eta", None)
    eta_prior = kwargs.get("eta_prior", "HalfCauchy")

    # construct the gaussian prior
    with model:
        if length_scale is None:
            # a length_scale around 40 appears to work reasonably
            gamma_hint = {"alpha": 40, "beta": 1}
            print(f"length_scale Gamma prior: {gamma_hint}")
            length_scale = pm.Gamma("length_scale", **gamma_hint)
            # Note: with the exponentiated quadratic kernel (without eta) ...
            #       at ls, correlation is around 0.61,
            #       at 2 * ls, correlation is around 0.14,
            #       at 3 * ls, correlation is around 0.01, etc.
            #       https://stats.stackexchange.com/questions/445484/

        if eta is None:
            # an eta around 1.6 appers to work well
            hint, function = {
                # I think HalfCauchy works best of those below
                "HalfCauchy": ({"beta": 4}, pm.HalfCauchy),
                "Gamma": ({"alpha": 2, "beta": 1}, pm.Gamma),
                "TruncatedNormal": (
                    {"lower": 0.5, "upper": 1000, "mu": 1.6, "sigma": 3},
                    pm.TruncatedNormal,
                ),
            }.get(eta_prior, ({"beta": 4}, pm.HalfNormal))
            print(f"eta {function.__name__} prior: {hint}")
            eta = function("eta", **hint)

        cov = (eta**2) * pm.gp.cov.ExpQuad(input_dim=1, ls=length_scale)
        gp = pm.gp.Latent(cov_func=cov)
        voting_intention = gp.prior("voting_intention", X=inputs["poll_day_c_"])
    return voting_intention


def gp_model(inputs: dict[str, Any], **kwargs) -> pm.Model:
    """PyMC model for pooling/aggregating voter opinion polls,
    using a Gaussian Process (GP). Note: **kwargs allows one to
    pass length_scale and eta to gp_prior() and/or pass approach,
    nu and sigma to gp_likelihood()."""

    model = pm.Model()
    voting_intention = gp_prior(inputs, model, **kwargs)
    house_effects = house_effects_model(inputs, model)
    core_likelihood(inputs, model, voting_intention, house_effects, **kwargs)
    return model


# --- Condition the model on the data
def report_glitches(idata: az.InferenceData) -> str:
    """Display summary diagnostics from the sampling process.
    Return a string that can be displayed on charts if
    there has been any Bayesian sampling issues."""

    glitches = []
    summary = az.summary(idata)

    max_r_hat = 1.01
    statistic = summary.r_hat.max()
    print(text := f"Max r_hat: {statistic}")
    if statistic > max_r_hat:
        glitches.append(text)

    min_ess = 400
    statistic = summary[["ess_tail", "ess_bulk"]].min().min()
    print(text := f"Min ess: {statistic}")
    if statistic < min_ess:
        glitches.append(text)

    try:
        diverging_count = int(np.sum(idata.sample_stats.diverging))
    except (ValueError, AttributeError):  # No sample_stats, or no .diverging
        diverging_count = 0
    print(text := f"Divergences: {diverging_count}")
    if diverging_count:
        glitches.append(text)

    return f"Bayesian sampling issues: {', '.join(glitches)}" if glitches else ""


def draw_samples(
    model: pm.Model, **kwargs
) -> tuple[az.InferenceData, str]:
    """Draw samples from the posterior distribution (ie. run the model)."""

    with model:
        idata = pm.sample(
            progressbar=True,
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


def _get_var(var_name: str, idata: az.InferenceData) -> pd.DataFrame:
    """Extract the chains/draws for a specified var_name."""

    return (
        az.extract(idata, var_names=var_name)
        .transpose("sample", ...)
        .to_dataframe()[var_name]
        .unstack(level=2)
        .T
    )


def plot_univariate(
    idata: az.InferenceData,
    var_names: str | Iterable[str],
    hdi_prob: float = 0.95,
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


def _plot_he_kde(df: pd.DataFrame, kwargs: dict) -> None:
    """Plot house effects using kernel density estimates (KDE)."""

    colors = plotting.MULTI_COLORS
    if len(colors) < len(df.columns):
        print("Warning: plot_he_kde() does not have enough unique colors")
        return

    fig, ax = plotting.initiate_plot()
    styles = plotting.STYLES * 4
    for index, col in enumerate(df.columns):
        mini, maxi = (
            df[col].quantile([0.0005, 0.9995]).to_list()
        )  # avoid super long tails
        x = np.linspace(mini, maxi, 1000)
        kde = ss.gaussian_kde(df[col])
        y = kde.evaluate(x)
        pd.Series(y, index=x).plot.line(
            ax=ax, ls=styles[index], lw=3, color=colors[index], label=col
        )

    ax.axvline(x=0, c="#333333", lw=0.75)
    ax.set_yticklabels([])
    defaults = {  # default arguments for finalise_plot()
        "xlabel": "Relative effect (percentage points)",
        "ylabel": "Probability density",
        "show": False,
        "tag": "kde",
        "y0": True,
        "zero_y": True,
        "legend": plotting.LEGEND_SET | {"ncols": 1, "fontsize": "xx-small"},
        **plotting.footers,
    }
    kwargs_copy, defaults = plotting.generate_defaults(kwargs, defaults)
    plotting.finalise_plot(ax, **defaults, **kwargs_copy)
    plotting.finalise_plot(ax, show=True)


def _plot_he_bar(df: pd.DataFrame, palette: str, middle: pd.Series, kwargs) -> None:
    """Plot house effects as a stacked bar chart."""

    _, ax = plotting.initiate_plot()
    cmap = plt.get_cmap(palette)
    vi_middle = df.quantile(0.5, axis=1)
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
    for index, value in vi_middle.items():
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
    ax.axvline(x=0, c="#333333", lw=0.75)
    defaults = {  # default arguments for finalise_plot()
        "xlabel": "Relative effect (percentage points)",
        "ylabel": None,
        "show": False,
        "legend": plotting.LEGEND_SET | {"ncols": 1, "fontsize": "xx-small"},
        **plotting.footers,
    }
    kwargs_copy, defaults = plotting.generate_defaults(kwargs, defaults)
    plotting.finalise_plot(ax, **defaults, **kwargs_copy)

    return vi_middle


def plot_residuals(vi_middle, he_middle, title_stem, inputs, **kwargs) -> None:
    """If polling company methodologies have not changed, then
    we would expect the residuals to be normally distributed."""

    print(kwargs.keys())  # debug   

    minimum_required = 10
    vi_middle = vi_middle.to_period("D")
    poll_firm = inputs['poll_firm'].map(inputs['firm_map'])
    dates = inputs['poll_date']
    poll_days = inputs['poll_day']
    for firm in sorted(he_middle.index):
        selected_polls = poll_firm[poll_firm == firm].index
        if len(selected_polls) < minimum_required:
            continue
        adjusted_polls = inputs['y'].loc[selected_polls] - he_middle[firm]
        key_dates = dates.loc[selected_polls]
        adjusted_polls.index = key_dates
        on_day = vi_middle[vi_middle.index.isin(key_dates)]
        residual = adjusted_polls - on_day
        
        ax = residual.plot.bar()
        ax.tick_params(axis='x', labelsize='x-small')
        ax.tick_params(axis='y', labelsize='x-small')
        plotting.finalise_plot(
            ax,
            title=f"Residuals for {title_stem}:\n{firm}",
            xlabel=None,
            ylabel="Percentage points",
            **kwargs
        )

        
def plot_house_effects(
    inputs: dict[str, Any],
    idata: az.InferenceData,
    palette: str,
    **kwargs,
) -> None:
    """Plot the House effects for both GRW and GP models.
    Choice of charts by setting bool kwargs plot_he_bar and
    plot_he_kde for either/both (a) stacked bar chart, or
    (b) kernel density estimates chart."""

    # get the data as a DataFrame
    df = _get_var("house_effects", idata).rename(index=inputs["firm_map"])
    he_middle = df.quantile(0.5, axis=1).sort_values()
    df = df.reindex(he_middle.index)

    he_bar = kwargs.pop("plot_he_bar", True)  # default
    he_kde = kwargs.pop("plot_he_kde", False)  # use pop to remove from kwargs

    if he_bar:
        _plot_he_bar(df, palette, he_middle, kwargs)

    if he_kde:
        _plot_he_kde(df.T, **kwargs)
    
    return he_middle
    


def plot_std_set(
    inputs: dict[str, Any],
    idata: az.InferenceData,
    **kwargs,
) -> pd.Series:  # returns the median sample
    """Produce the standard set of charts for a Bayesian
    Analysis. Return the median voting intention sample."""

    # we dont want to pass these arguments on
    glitches: str = kwargs.pop("glitches", "")
    show: bool = kwargs.pop("show", False)
    title_stem = kwargs.pop("title_stem", "title")

    core_plot_args: Mapping = {
        "show": show,
        "rheader": (None if not glitches else glitches),
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
    palette = plotting.get_party_palette(title_stem)
    vi_middle = plot_voting(
        inputs,
        idata,
        palette,
        title=f"Vote share: {title_stem}",
        **core_plot_args,
    )

    # plot house effects
    he_middle = plot_house_effects(
        inputs,
        idata,
        palette,
        title=f"House Effects: {title_stem}",
        **(core_plot_args | kwargs),
    )

    plot_residuals(vi_middle, he_middle, title_stem, inputs,
                   **(core_plot_args | kwargs))

    return vi_middle
