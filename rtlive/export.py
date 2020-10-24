"""
This module contains functions for creating exportable summaries from sampling results.
"""
import numpy
import typing

import arviz


def summarize_median_and_hdi(samples, prefix:str, hdi_prob:typing.Union[float, typing.Iterable[float]]=0.9) -> typing.Dict[str, float]:
    """ Extract median, lower and upper bound and return it as a dict.

    Parameters
    ----------
    samples : array-like
        samples to summarize (N_samples,) or (N_dates, N_samples)
    prefix : str
        name of the quantity
    hdi_prob : float, iterable
        see arviz.hdi
        If multiple HDI probs are passed, each will get its own entry in the resulting dict.
        Numpy arrays are automatically converted to lists, to avoid problems in JSON serialization.

    Returns
    -------
    summary : dict
        a dict with median, lower and upper HDI(s)
    """
    samples = numpy.array(samples).T
    result = {
        prefix: numpy.median(samples, axis=0)
    }
    if numpy.isscalar(hdi_prob):
        hdi = arviz.hdi(samples, hdi_prob=hdi_prob).T
        result[f'{prefix}_lower'] = hdi[0]
        result[f'{prefix}_upper'] = hdi[1]
    else:
        for hp in hdi_prob:
            hdi = arviz.hdi(samples, hdi_prob=hp).T
            result[f'{prefix}_lower_{hp}'] = hdi[0]
            result[f'{prefix}_upper_{hp}'] = hdi[1]
    # convert numpy arrays to lists, to avoid problems in json serialization
    for k, v in result.items():
        if numpy.shape(v):
            result[k] = list(v)
    return result


def summarize_r_t(samples, hdi_prob=0.9):
    return {
        'r_t_threshold_probability': float((samples > 1).mean()),
        **summarize_median_and_hdi(samples, "r_t", hdi_prob=hdi_prob)
    }


def summarize_infections(samples, region, date, hdi_prob=0.9) -> typing.Dict[str, float]:
    """ Summarizes lower/upper bounds of daily infections and the probability
    that the infection rate is greater than 20/100_000/week at that day.
    """
    # Berliner "Corona-Ampel" nimmt 20/100_000/week als "gr�n"
    # Hier wird dies pro Tag heruntergebrochen
    population = DE_REGION_POPULATION[region]
    threshold_by_100k = 20 / 7
    absolute_threshold = population * threshold_by_100k / 100_000
    return {
        "population": population,
        "infections_threshold_by_100k": threshold_by_100k,
        "infections_threshold_absolute": absolute_threshold,
        "infections_threshold_probability": float(numpy.mean(samples > absolute_threshold)),
        **summarize_median_and_hdi(samples / population * 100_000, "infections_by_100k", hdi_prob=hdi_prob),
        **summarize_median_and_hdi(samples, "infections_absolute", hdi_prob=hdi_prob),
    }
