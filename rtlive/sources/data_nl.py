import logging
import numpy
import pandas

from . import ourworldindata
from .. import preprocessing

_log = logging.getLogger(__file__)


def forecast_NL(df: pandas.DataFrame):
    """ Applies testcount interpolation/extrapolation.

    Currently this assumes the OWID data, which only has an "all" region.
    In the future, this should be replaced with more fine graned data loading!
    """
    # only weekly total_tests are available -> interpolate the totals, then diff to get daily new_tests
    df.at[("all", "2020-01-01"), 'total_tests'] = 0
    df["new_tests"] = df.total_tests.interpolate("linear").diff()
    df["new_tests"][df.new_tests == 0] = numpy.nan
    # forecast with existing data
    df['predicted_new_tests'], results = preprocessing.predict_testcounts_all_regions(df, 'NL')
    return df, results


from .. import data
data.set_country_support(
    country_alpha2="NL",
    compute_zone=data.Zone.Europe,
    region_name={
        "all": "Netherlands",
    },
    region_population={
        "all": 17_290_688,
    },
    fn_load=ourworldindata.create_loader_function("NL"),
    fn_process=forecast_NL,
)
