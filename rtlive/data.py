import dataclasses
import enum
import iso3166
import logging
import numpy
import typing
import pandas

from . import preprocessing

_log = logging.getLogger(__file__)
LoadFunction = typing.Callable[
    [pandas.Timestamp],
    pandas.DataFrame
]
ProcessFunction = typing.Callable[
    [pandas.DataFrame],
    typing.Tuple[
        # input: result of LoadFunction
        pandas.DataFrame,
        # output: dictionary of forecasting results for all regions
        typing.Dict[str, preprocessing.ForecastingResult]
    ]
]

class Zone(enum.Enum):
    Asia = "Asia"
    Europe = "Europe"
    America = "America"


@dataclasses.dataclass
class SupportedCountry:
    alpha2: str
    compute_zone: Zone
    region_name: typing.Dict[str, str]
    region_population: typing.Dict[str, int]
    fn_load: LoadFunction
    fn_process: ProcessFunction


SUPPORTED_COUNTRIES: typing.Dict[str, SupportedCountry] = {}


def set_country_support(
    country_alpha2: str,
    *,
    compute_zone: Zone,
    region_name: typing.Dict[str, str],
    region_population: typing.Dict[str, int],
    fn_load: LoadFunction,
    fn_process: ProcessFunction,
):
    """ Function to set support for a country.

    Parameters
    ----------
    country_alpha2 : str
        ISO-3166 alpha-2 short code of the country (key in SUPPORTED_COUNTRIES dict)
    compute_zone : Zone
        used to group countries by timezone for automated computing
    region_name : dict
        dictionary of { region_code : str }
        to map machine-readable region codes to human-readable names
    region_population : dict
        dictionary of { region_code : int }
        to map machine-readable region codes to number of inhabitants
    fn_load : callable
        A function that takes one date argument `run_date` and returns a DataFrame
        indexed by ["region", "date"] with columns ["new_cases", "new_tests"].
        Use NaN to indicate missing data (e.g. in new_tests).

        Ideally the function should return data "as it was on `run_date`", meaning that information
        such as corrections that became available after `run_date` should not be taken into account.
        This is important to realistically back-test how the model would have performed at `run_date`.
    fn_process : callable
        A processing function that takes the ["region", "date"]-indexed DataFrame
        returned by the load function as the input.
        The return value must be a dictionary (by region code) of forecasting results.
    """
    if country_alpha2 not in iso3166.countries_by_alpha2:
        raise KeyError(f"Unknown ISO-3166 alpha 2 country code '{country_alpha2}'.")
    # register loading functions
    SUPPORTED_COUNTRIES[country_alpha2] = SupportedCountry(
        country_alpha2,
        compute_zone,
        region_name,
        region_population,
        fn_load,
        fn_process,
    )
    return


def get_data(country: str, run_date: pandas.Timestamp) -> pandas.DataFrame:
    """ Retrieves data for a country using the registered data loader method.

    Parameters
    ----------
    country : str
        ISO-3166 alpha-2 short code of the country (key in SUPPORTED_COUNTRIES dict)
    run_date : pandas.Timestamp
        date when the analysis is performed

    Returns
    -------
    model_input : pandas.DataFrame
        Data as returned by data loader function.
    """
    country = country.upper()
    if country not in SUPPORTED_COUNTRIES:
        raise KeyError(
            f"The country '{country}' is not in the collection of supported countries."
        )
    scountry = SUPPORTED_COUNTRIES[country]
    result = scountry.fn_load(run_date)
    assert isinstance(result, pandas.DataFrame)
    assert result.index.names == ("region", "date")
    missing_names = set(result.reset_index().region) - set(scountry.region_name.keys())
    missing_pop = set(result.reset_index().region) - set(
        scountry.region_population.keys()
    )
    if missing_names:
        raise Exception(
            f"Data contains regions {missing_names} for which no names were registered."
        )
    if missing_pop:
        raise Exception(
            f"Data contains regions {missing_pop} for which no population were registered."
        )
    assert "new_cases" in result.columns, f"Columns were: {result.columns}"
    assert "new_tests" in result.columns, f"Columns were: {result.columns}"
    for col in ["new_cases", "new_tests", "new_deaths"]:
        if col in result and any(result[col] < 0):
            _log.warning(
                f"Column '%s' has %i negative entries!! Overriding with NaN...",
                col,
                sum(result[col] < 0),
            )
            result.loc[result[col] < 0, col] = numpy.nan
    return result


def process_testcounts(
    country: str, df_raw: pandas.DataFrame
) -> typing.Tuple[pandas.DataFrame, typing.Dict[str, preprocessing.ForecastingResult]]:
    """ Fills and forecasts test counts with country-specific logic.

    Parameters
    ----------
    country : str
        ISO-3166 alpha-2 short code of the country (key in FORECASTERS dict)
    df_raw : pandas.DataFrame
        Data as returned by data loader function.

    Returns
    -------
    df_result : pandas.DataFrame
        Input dataframe with a new column "predicted_new_tests"
    forecasting_results : dict
        the fbprophet results by region
    """
    country = country.upper()
    if country not in SUPPORTED_COUNTRIES:
        raise KeyError(
            f"The country '{country}' is not in the collection of supported countries."
        )
    df, results = SUPPORTED_COUNTRIES[country].fn_process(df_raw.copy())
    assert isinstance(df, pandas.DataFrame)
    assert df.index.names == ("region", "date")
    assert "predicted_new_tests" in df.columns, f"Columns were: {df.columns}"
    return df, results


def iter_countries_by_zone() -> typing.Iterator[
    typing.Tuple[Zone, typing.List[SupportedCountry]]
]:
    """ Iterates over supported countries, grouped by Zone.

    Yields
    ------
    zone : Zone
        the compute zone
    countries_in_zone : list
        the supported countries in the respective zone        
    """
    for zone in Zone:
        countries_in_zone = [
            country
            for _, country in SUPPORTED_COUNTRIES.items()
            if country.compute_zone == zone
        ]
        yield zone, countries_in_zone
    return
