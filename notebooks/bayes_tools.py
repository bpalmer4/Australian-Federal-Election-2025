"""Tools for doing Bayesian aggregation of polls"""

from typing import Any, Iterable, Sequence, Mapping, Optional

import arviz as az
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[import-untyped]
import pytensor.tensor as pt
import scipy.stats as ss  # type: ignore[import-untyped]

from common import MIDDLE_DATE, ensure
import plotting


# --- Data preparation
TAIL_CENTRED = "tail_centred"


def _check_he_constraints(box: dict[str, Any]) -> None:
    """Check that the house effect constraints are as expected.
    The list of firms should be in the correct order, with the
    pollsters excluded from the sum-to-zero constraint should
    be at the end of the list."""

    # -- get the data we need to check
    he_sum_exclusions = box["he_sum_exclusions"]
    he_sum_inclusions = box["he_sum_inclusions"]
    firms = box["firm_list"]

    # -- check that our house effects are all lists of strings
    for check in (he_sum_exclusions, he_sum_inclusions, firms):
        ensure(
            isinstance(check, list),
            "House effect constraints must be lists of strings.",
        )
        for element in check:
            ensure(
                isinstance(element, str),
                "House effect constraints must be lists of strings.",
            )

    # -- check we have at least minimum_houses of constraints
    minimum_houses = 2
    ensure(
        len(he_sum_inclusions) >= minimum_houses,
        f"Need at least {minimum_houses} firm for house effects.",
    )

    # -- check the includesions are first
    he_inclusions2 = firms[: len(he_sum_inclusions)]
    ensure(
        set(he_sum_inclusions) == set(he_inclusions2),
        "The house-effect-constrained pollsters should be first in the 'firm-list'.",
    )

    # -- check the exclusions are last
    he_sum_exclusions2 = firms[-len(he_sum_exclusions) :]
    ensure(
        set(he_sum_exclusions) == set(he_sum_exclusions2),
        "The unconstrained pollsters should be last in the 'firm-list'.",
    )


def prepare_data_for_analysis(
    df: pd.DataFrame,
    column: str,
    **kwargs,
) -> dict[str, Any]:
    """Prepare a dataframe column for Bayesian analysis.
    Returns a python dict with all the necessary values within."""

    ensure(column in df.columns, "Column not found in DataFrame.")
    box: dict[str, Any] = {}  # container we will return
    box["column"] = column
    df = df.copy().loc[df[column].notnull()]  # remove nulls

    # make sure data is properly sorted by date, with an unique index
    df = df.sort_values(MIDDLE_DATE).reset_index(drop=True)
    box["df"] = df

    # get our zero centered observations, ignore missing data
    # assume data in percentage points (0..100)
    y = df[column]
    box["y"] = y

    if n := kwargs.get(TAIL_CENTRED, 0):
        # centre around last n polls
        # to minimise mean-reversion issues with GP, but may introduce bias
        centre_offset = -y.iloc[-n:].mean()
    else:
        centre_offset = -y.mean()
    zero_centered_y = y + centre_offset
    box["centre_offset"] = centre_offset
    box["zero_centered_y"] = zero_centered_y
    box["n_polls"] = len(zero_centered_y)

    # get our day-to-date mapping
    right_anchor: Optional[tuple[pd.Period, float]] = kwargs.get("right_anchor", None)
    box["right_anchor"] = right_anchor
    left_anchor: Optional[tuple[pd.Period, float]] = kwargs.get("left_anchor", None)
    box["left_anchor"] = left_anchor
    day_zero = pd.Period(
        left_anchor[0] if left_anchor is not None else df[MIDDLE_DATE].min(),
        freq="D",
    )
    box["day_zero"] = day_zero
    last_day = pd.Period(
        right_anchor[0] if right_anchor is not None else df[MIDDLE_DATE].max(),
        freq="D",
    )
    box["last_day"] = last_day
    poll_date = pd.Series(
        [pd.Period(x, freq="D") for x in df[MIDDLE_DATE]], index=df.index
    )
    box["poll_date"] = poll_date
    poll_day = pd.Series([(x - day_zero).n for x in poll_date], index=df.index)
    box["poll_day"] = poll_day
    box["n_days"] = poll_day.max() + 1
    box["poll_day_c_"] = np.c_[poll_day]  # numpy column vector of poll_days for GP

    # sanity checks - anchors must be before or after polling data
    ensure(
        (left_anchor and day_zero < df[MIDDLE_DATE].min())
        or (right_anchor and last_day > df[MIDDLE_DATE].max())
        or (not left_anchor and not right_anchor),
        "Anchors must be outside of the range of polling dates.",
    )

    # get house effects inputs
    empty_list: list[str] = []
    he_sum_exclusions: list[str] = kwargs.get("he_sum_exclusions", empty_list)
    missing_firm: list[str] = [
        e for e in he_sum_exclusions if e not in df.Brand.unique()
    ]
    if missing_firm:
        # firm is not in the data, but it is one we should exclude?
        he_sum_exclusions = sorted(list(set(he_sum_exclusions) - set(missing_firm)))
    box["he_sum_exclusions"] = he_sum_exclusions
    he_sum_inclusions: list[str] = [
        e for e in df.Brand.unique() if e not in he_sum_exclusions
    ]
    box["he_sum_inclusions"] = he_sum_inclusions

    # get pollster map - ensure polsters at end of the list are the excluded ones
    firm_list = he_sum_inclusions + he_sum_exclusions  # ensure inclusions first
    box["firm_list"] = firm_list
    ensure(
        len(firm_list) == len(set(firm_list)),
        "Remove duplicate pollsters in the firm_list.",
    )
    n_firms = len(firm_list)
    box["n_firms"] = n_firms
    ensure(n_firms > len(he_sum_exclusions), "Number of exclusions == number of firms.")
    firm_map = {firm: code for code, firm in enumerate(firm_list)}
    box["firm_map"] = firm_map
    box["back_firm_map"] = {v: k for k, v in firm_map.items()}
    box["poll_firm_number"] = pd.Series([firm_map[b] for b in df.Brand], index=df.index)

    # final sanity checks ...
    _check_he_constraints(box)

    # Information
    if kwargs.get("verbose", False):
        print(box)

    return box


# --- Bayesian models and model components
# - Gaussian Random Walk
def _guess_start(inputs: dict[str, Any], n=10) -> np.float64:
    """Guess a starting point for the random walk,
    based on the first n poll results."""

    if n > (m := len(inputs["zero_centered_y"])):
        n = m
        print(f"Caution: Input data series is only {n} observations long.")
    return inputs["zero_centered_y"][:n].mean()


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
    both the GRW and the Gaussian Process (GP) models."""

    house_effect_sigma = inputs.get("house_effect_sigma", 5.0)
    with model:
        if inputs["right_anchor"] is None and inputs["left_anchor"] is None:
            if len(inputs["he_sum_exclusions"]) > 0:
                # sum to zero constraint for some (but not all) houses
                zero_sum_he = pm.ZeroSumNormal(
                    "zero_sum_he",
                    sigma=house_effect_sigma,
                    shape=len(inputs["he_sum_inclusions"]),
                )
                unconstrained_he = pm.Normal(
                    "unconstrained_he",
                    sigma=house_effect_sigma,
                    shape=len(inputs["he_sum_exclusions"]),
                )
                house_effects = pm.Deterministic(
                    "house_effects",
                    var=pm.math.concatenate([zero_sum_he, unconstrained_he]),
                )
            else:
                # sum to zero constraint for all houses
                house_effects = pm.ZeroSumNormal(
                    "house_effects", sigma=house_effect_sigma, shape=inputs["n_firms"]
                )
        else:
            # all house effects are unconstrained, used in anchored models
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
            voting_intention[inputs["poll_day"]]
            + house_effects[inputs["poll_firm_number"]]
            if grw
            else voting_intention + house_effects[inputs["poll_firm_number"]]
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
        diverging_count = int(np.sum(idata.sample_stats.diverging))  # type: ignore[attr-defined]
    except (ValueError, AttributeError):  # No sample_stats, or no .diverging
        diverging_count = 0
    print(text := f"Divergences: {diverging_count}")
    if diverging_count:
        glitches.append(text)

    return f"Bayesian sampling issues: {', '.join(glitches)}" if glitches else ""


def draw_samples(model: pm.Model, **kwargs) -> tuple[az.InferenceData, str]:
    """Draw samples from the posterior distribution (ie. run the model)."""

    plot_trace = kwargs.pop("plot_trace", True)
    with model:
        idata = pm.sample(
            progressbar=True,
            return_inferencedata=True,
            **kwargs,
        )
        if plot_trace:
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
        if variable not in idata.posterior:  # type: ignore[attr-defined]
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


def plot_voting(inputs, idata, previous: float, palette, **kwargs) -> pd.Series:
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
    # x = df.index  # to do - remove this line
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
    # plot previous election
    ax.axhline(y=previous, c="#555555", lw=0.75, ls="--", label="Previous election")
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
        | {"fontsize": "xx-small", "loc": f"{space} left", "ncols": 3},
        "concise_dates": True,
        **plotting.footers,
    }
    kwargs_copy, defaults = plotting.generate_defaults(kwargs, defaults)
    plotting.finalise_plot(ax, **defaults, **kwargs_copy)

    return middle


def _plot_he_kde(df: pd.DataFrame, kwargs: dict) -> None:
    """Plot house effects using kernel density estimates (KDE)."""

    colors: Sequence = plotting.MULTI_COLORS
    if len(colors) < len(df.columns):
        cm = plt.get_cmap("gist_rainbow")
        colors = [cm(x) for x in np.linspace(0, 1, len(df.columns))]

    _fig, ax = plotting.initiate_plot()
    styles = plotting.STYLES * 4
    for index, col in enumerate(df.columns):
        mini, maxi = (
            df[col].quantile([0.0005, 0.9995]).to_list()
        )  # avoid super long tails
        x = np.linspace(mini, maxi, 1000)
        kde = ss.gaussian_kde(df[col])
        y = kde.evaluate(x)
        pd.Series(y, index=x).plot.line(
            ax=ax, ls=styles[index], lw=2, color=colors[index], label=col
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


def _plot_he_bar(
    df: pd.DataFrame, inputs: dict[str, Any], palette: str, kwargs: dict
) -> None:
    """Plot house effects as a stacked bar chart."""

    _, ax = plotting.initiate_plot()
    cmap = plt.get_cmap(palette)
    vi_middle = df.quantile(0.5, axis=1)
    bottom = 0
    for i, p in enumerate(PERCENTS):
        quants = p, 100 - p
        label = f"{quants[1] - quants[0]}% HDI"
        lower, upper = [df.quantile(q=q / 100.0, axis=1) for q in quants]
        bottom = min(bottom, lower.min())
        color = COLOR_FRACS[i]
        ax.barh(
            df.index,
            width=upper - lower,
            left=lower,
            color=cmap(color),
            label=label,
            zorder=i + 1,
        )
    for index, value in enumerate(vi_middle):
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
    message = "HE constrained to sum to zero"
    for name in inputs["he_sum_inclusions"]:
        ax.scatter(
            bottom + 0.1, name, c="black", marker="o", s=10, zorder=i + 3, label=message
        )
        message = "_"
    ax.tick_params(axis="y", labelsize="small")
    ax.axvline(x=0, c="#333333", lw=0.75)
    defaults = {  # default arguments for finalise_plot()
        "xlabel": "Relative effect (percentage points)",
        "ylabel": None,
        "show": False,
        "legend": plotting.LEGEND_SET
        | {"ncols": 1, "fontsize": "xx-small", "loc": "lower right"},
        # "lheader": f"Included in sum-to-zero: {', '.join(inputs['he_sum_inclusions'])}",
        **plotting.footers,
    }
    kwargs_copy, defaults = plotting.generate_defaults(kwargs, defaults)
    plotting.finalise_plot(ax, **defaults, **kwargs_copy)


def plot_residuals(vi_middle, he_middle, title_stem, inputs, **kwargs) -> None:
    """If polling company methodologies have not changed, then
    we would expect the residuals to be normally distributed."""

    minimum_required = 5
    vi_middle = vi_middle.to_period("D")
    dates = inputs["poll_date"]
    poll_numbers = inputs["poll_firm_number"]
    # poll_days = inputs["poll_day"]  # to do remove this line
    for firm_name in sorted(he_middle.index):
        firm_number = inputs["firm_map"][firm_name]
        selected_polls = poll_numbers[poll_numbers == firm_number].index
        if len(selected_polls) < minimum_required:
            continue
        adjusted_polls = inputs["y"].loc[selected_polls] - he_middle[firm_name]
        key_dates = dates.loc[selected_polls]
        adjusted_polls.index = key_dates
        on_day = vi_middle[vi_middle.index.isin(key_dates)]
        residual = adjusted_polls - on_day

        ax = residual.plot.bar()
        ax.tick_params(axis="x", labelsize="x-small")
        ax.tick_params(axis="y", labelsize="x-small")
        plotting.finalise_plot(
            ax,
            title=f"Residuals for {title_stem}:\n{firm_name}",
            xlabel=None,
            ylabel="Percentage points",
            **kwargs,
        )


def plot_house_effects(
    inputs: dict[str, Any],
    idata: az.InferenceData,
    palette: str,
    **kwargs,
) -> pd.Series:
    """Plot the House effects for both GRW and GP models.
    Choice of charts by setting bool kwargs plot_he_bar and
    plot_he_kde for either/both (a) stacked bar chart, or
    (b) kernel density estimates chart."""

    # get the data as a DataFrame
    inv_map = {v: k for k, v in inputs["firm_map"].items()}
    df = _get_var("house_effects", idata).rename(index=inv_map)
    he_middle = df.quantile(0.5, axis=1).sort_values()
    df = df.reindex(he_middle.index)

    he_bar = kwargs.pop("plot_he_bar", True)  # default
    he_kde = kwargs.pop("plot_he_kde", False)  # use pop to remove from kwargs

    if he_bar:
        _plot_he_bar(df, inputs, palette, kwargs)

    if he_kde:
        _plot_he_kde(df.T, kwargs)

    return he_middle


def plot_std_set(
    inputs: dict[str, Any],
    idata: az.InferenceData,
    previous: float,
    **kwargs,
) -> pd.Series:  # returns the median sample
    """Produce the standard set of charts for a Bayesian
    Analysis. Return the median voting intention sample."""

    # we dont want to pass these arguments on
    glitches: str = kwargs.pop("glitches", "")
    show: bool = kwargs.pop("show", False)
    title_stem = kwargs.pop("title_stem", "title")
    residuals = kwargs.pop("residuals", False)

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
        previous,
        palette,
        title=f"Bayesian Aggregation: {title_stem}",
        **(core_plot_args | kwargs),  # type: ignore[operator]
    )

    # plot house effects
    he_middle = plot_house_effects(
        inputs,
        idata,
        palette,
        title=f"House Effects: {title_stem}",
        **(core_plot_args | kwargs),  # type: ignore[operator]
    )

    if residuals:
        plot_residuals(
            vi_middle,
            he_middle,
            title_stem,
            inputs,
            **(core_plot_args | kwargs),  # type: ignore[operator]
        )

    return vi_middle
