import logging
import pandas

from . import ourworldindata
from .. import preprocessing

_log = logging.getLogger(__file__)


def forecast_PT(df: pandas.DataFrame):
    """ Applies testcount interpolation/extrapolation.

    Currently this assumes the OWID data, which only has an "all" region.
    In the future, this should be replaced with more fine graned data loading!
    """
    # forecast with existing data
    df['predicted_new_tests'], results = preprocessing.predict_testcounts_all_regions(df, 'PT')
    # interpolate the initial testing ramp-up to account for missing data
    return df, results


from .. import data
data.set_country_support(
    country_alpha2="PT",
    compute_zone=data.Zone.Europe,
    region_name={
        "all": "Portugal",
    },
    region_population={
        "all": 10_295_909,
    },
    fn_load=ourworldindata.create_loader_function("PT"),
    fn_process=forecast_PT,
)
