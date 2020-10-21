import logging
import numpy
import pandas
import typing

import arviz
import pymc3
import theano
import theano.tensor as tt
import theano.tensor.signal.conv
import xarray

from . import assumptions

__version__ = '1.0.0'
_log = logging.getLogger(__file__)


def _reindex_observed(observed:pandas.DataFrame, buffer_days:int=10):
    _log.info("Model will start with %i unobserved buffer days before the data.", buffer_days)
    first_index = observed.positive.gt(0).argmax()
    observed = observed.iloc[first_index:]
    new_index = pandas.date_range(
        start=observed.index[0] - pandas.Timedelta(days=buffer_days),
        end=observed.index[-1],
        freq="D",
    )
    observed = observed.reindex(new_index, fill_value=0)
    return observed


def _to_convolution_ready_gt(gt, len_observed):
    """ Speeds up theano.scan by pre-computing the generation time interval
        vector. Thank you to Junpeng Lao for this optimization.
        Please see the outbreak simulation math here:
        https://staff.math.su.se/hoehle/blog/2020/04/15/effectiveR0.html """
    convolution_ready_gt = np.zeros((len_observed - 1, len_observed))
    for t in range(1, len_observed):
        begin = np.maximum(0, t - len(gt) + 1)
        slice_update = gt[1 : t - begin + 1][::-1]
        convolution_ready_gt[
            t - 1, begin : begin + len(slice_update)
        ] = slice_update
    convolution_ready_gt = theano.shared(convolution_ready_gt)
    return convolution_ready_gt


class GenerativeModel:
    def __init__(self, region: str, observed: pd.DataFrame, buffer_days=10):
        """ Takes a region (ie State) name and observed new positive and
            total test counts per day. buffer_days is the default number of
            blank days we pad on the leading edge of the time series because
            infections occur long before reports and we need to infer values
            on those days """
        self._trace = None
        self._inference_data = None
        self.model = None
        self.observed = _reindex_observed(observed.dropna(subset=['positive', 'total']), buffer_days)
        self.region = region

    def build(self):
        """ Builds and returns the Generative model. Also sets self.model """

        p_delay = assumptions.delay_distribution()
        nonzero_days = self.observed.total.gt(0)
        len_observed = len(self.observed)
        convolution_ready_gt = _to_convolution_ready_gt(assumptions.generation_time(), len_observed)
        x = np.arange(len_observed)[:, None]

        coords = {
            "date": self.observed.index.values,
            "nonzero_date": self.observed.index.values[self.observed.total.gt(0)],
        }
        with pm.Model(coords=coords) as self.model:

            # Let log_r_t walk randomly with a fixed prior of ~0.035. Think
            # of this number as how quickly r_t can react.
            log_r_t = pm.GaussianRandomWalk(
                "log_r_t",
                sigma=0.035,
                dims=["date"]
            )
            r_t = pm.Deterministic("r_t", pm.math.exp(log_r_t), dims=["date"])

            # For a given seed population and R_t curve, we calculate the
            # implied infection curve by simulating an outbreak. While this may
            # look daunting, it's simply a way to recreate the outbreak
            # simulation math inside the model:
            # https://staff.math.su.se/hoehle/blog/2020/04/15/effectiveR0.html
            seed = pm.Exponential("seed", 1 / 0.02)
            y0 = tt.zeros(len_observed)
            y0 = tt.set_subtensor(y0[0], seed)
            outputs, _ = theano.scan(
                fn=lambda t, gt, y, r_t: tt.set_subtensor(y[t], tt.sum(r_t * y * gt)),
                sequences=[tt.arange(1, len_observed), convolution_ready_gt],
                outputs_info=y0,
                non_sequences=r_t,
                n_steps=len_observed - 1,
            )
            infections = pm.Deterministic("infections", outputs[-1], dims=["date"])

            # Convolve infections to confirmed positive reports based on a known
            # p_delay distribution. See patients.py for details on how we calculate
            # this distribution.
            test_adjusted_positive = pm.Deterministic(
                "test_adjusted_positive",
                conv2d(
                    tt.reshape(infections, (1, len_observed)),
                    tt.reshape(p_delay, (1, len(p_delay))),
                    border_mode="full",
                )[0, :len_observed],
                dims=["date"]
            )

            # Picking an exposure with a prior that exposure never goes below
            # 0.1 * max_tests. The 0.1 only affects early values of Rt when
            # testing was minimal or when data errors cause underreporting
            # of tests.
            tests = pm.Data("tests", self.observed.total.values, dims=["date"])
            exposure = pm.Deterministic(
                "exposure",
                pm.math.clip(tests, self.observed.total.max() * 0.1, 1e9),
                dims=["date"]
            )

            # Test-volume adjust reported cases based on an assumed exposure
            # Note: this is similar to the exposure parameter in a Poisson
            # regression.
            positive = pm.Deterministic(
                "positive", exposure * test_adjusted_positive,
                dims=["date"]
            )

            # Save data as part of trace so we can access in inference_data
            observed_positive = pm.Data("observed_positive", self.observed.positive.values, dims=["date"])
            nonzero_observed_positive = pm.Data("nonzero_observed_positive", self.observed.positive[nonzero_days.values].values, dims=["nonzero_date"])

            positive_nonzero = pm.NegativeBinomial(
                "nonzero_positive",
                mu=positive[nonzero_days.values],
                alpha=pm.Gamma("alpha", mu=6, sigma=1),
                observed=nonzero_observed_positive,
                dims=["nonzero_date"]
            )

        return self.model


def sample(pmodel:pymc3.Model, **kwargs):
    """ Run sampling with default settings.

    Parameters
    ----------
    pmodel : pymc3.Model
        the PyMC3 model to sample from
    **kwargs
        additional keyword-arguments to pass to pm.sample
        (overriding the defaults from this implementation)

    Returns
    -------
    idata : arviz.InferenceData
        the sampling and posterior predictive result
    """
    with pmodel:
        sample_kwargs = dict(
            return_inferencedata=False,
            target_accept=0.95,
            init='jitter+adapt_diag',
            cores=4,
            chains=4,
            tune=700, draws=200,
        )
        sample_kwargs.update(kwargs)
        trace = pymc3.sample(**sample_kwargs)

        idata = arviz.from_pymc3(
            trace=trace,
            posterior_predictive=pymc3.sample_posterior_predictive(trace),
        )
        idata.posterior.attrs["model_version"] = __version__
    return idata


def get_scale_factor(idata: arviz.InferenceData) -> xarray.DataArray:
    """ Calculate a scaling factor so we can work/plot with
    the inferred "infections" curve.
    
    The scaling factor depends on the probability that an infection is observed
    (sum of p_delay distribution). The current p_delay distribution sums to 0.9999999,
    so right now the scaling ASSUMES THAT THERE'S NO DARK FIGURE !!
    Therefore the factor should be interpreted as the lower-bound!!

    Parameters
    ----------
    idata : arviz.InferenceData
        sampling result of Rtlive model v1.0.2 or higher

    Returns
    -------
    factor : xarray.DataArray
        scaling factors (sample,)
    """
    p_observe = numpy.sum(idata.constant_data.p_delay)
    total_observed = numpy.sum(idata.constant_data.observed_positive)

    # new method: normalizing using the integral of exposure-adjusted test_adjusted_positive
    # - assumes that over time testing is not significantly steered towards high-risk individuals
    exposure_profile = idata.constant_data.exposure / idata.constant_data.exposure.max()
    total_inferred = (idata.posterior.test_adjusted_positive * exposure_profile) \
        .stack(sample=('chain', 'draw')) \
        .sum('date')
    scale_factor = total_observed / total_inferred / p_observe
    return scale_factor
