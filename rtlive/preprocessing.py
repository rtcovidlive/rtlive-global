import datetime
import iso3166
import logging
import pandas
import typing
import numpy

import fbprophet
import holidays

_log = logging.getLogger(__file__)


# custom type shortcuts
NamedDates = typing.Dict[datetime.datetime, str]
ForecastingResult = typing.Tuple[
    pandas.Series, fbprophet.Prophet, pandas.DataFrame, NamedDates
]


def get_holidays(
    country: str,
    region: typing.Optional[typing.Union[str, typing.List[str]]],
    years: typing.Sequence[int],
) -> NamedDates:
    """ Retrieve a dictionary of holidays in the region.

    Implemented by Laura Helleckes and Michael Osthege.

    Parameters
    ----------
    country : str
        name or short code of country (as used by https://github.com/dr-prodigy/python-holidays)
    region : optional, [str]
        if None or []: only nation-wide
        if "all": nation-wide and all regions
        if "CA": nation-wide and those for region "CA"
        if ["CA", "NY", "FL"]: nation-wide and those for all listed regions
    years : list of str
        years to get holidays for

    Returns
    -------
    holidays : dict
        datetime as keys, name of holiday as value
    """
    country = iso3166.countries.get(country).alpha3
    if not hasattr(holidays, country):
        raise KeyError(f'Country "{country}" was not found in the `holidays` package.')
    country_cls = getattr(holidays, country)
    use_states = hasattr(country_cls, "STATES")

    if not region:
        region = []
    if region == "all":
        # select all
        regions = country_cls.STATES if use_states else country_cls.PROVINCES
    else:
        regions = numpy.atleast_1d(region)

    result = country_cls(years=years)
    for region in regions:
        is_province = region in country_cls.PROVINCES
        is_state = use_states and region in country_cls.STATES
        if is_province:
            result.update(country_cls(years=years, prov=region))
        elif is_state:
            result.update(country_cls(years=years, state=region))
        else:
            raise KeyError(
                f'Region "{region}" not found in {country} states or provinces.'
            )
    return result
