import logging
import pandas

from . import ourworldindata
from .. import preprocessing

_log = logging.getLogger(__file__)


def forecast_CZ(df: pandas.DataFrame):
    """ Applies testcount interpolation/extrapolation.

    Currently this assumes the OWID data, which only has an "all" region.
    In the future, this should be replaced with more fine graned data loading!
    """
    # forecast with existing data
    df['predicted_new_tests'], results = preprocessing.predict_testcounts_all_regions(df, 'CZ')
    # interpolate the initial testing ramp-up to account for missing data
    df_region = df.xs('all')
    df_region.loc['2020-01-01', 'predicted_new_tests'] = 0
    df_region.predicted_new_tests = df_region.predicted_new_tests.interpolate('linear')
    df_region['region'] = 'all'
    df = df_region.reset_index().set_index(['region', 'date'])
    return df, results


from .. import data
data.set_country_support(
    country_alpha2="CZ",
    compute_zone=data.Zone.Europe,
    region_name={
        "all": "Czech Republic",
    },
    region_population={
        "all": 10_637_794,
    },
    fn_load=ourworldindata.create_loader_function("CZ"),
    fn_process=forecast_CZ,
)
