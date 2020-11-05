import logging
import pandas

from . import ourworldindata
from .. import preprocessing

_log = logging.getLogger(__file__)

IT_REGION_NAMES = {
    '1': 'Piemonte',
    '2': 'Valle d\'Aosta',
    '3': 'Lombardia',
    '5': 'Veneto',
    '6': 'Friuli Venezia Giulia',
    '7': 'Liguria',
    '8': 'Emilia-Romagna,
    '9': 'Toscana',
    '10': 'Umbria',
    '11': 'Marche',
    '12': 'Lazio',
    '13': 'Abruzzo',
    '14': 'Molise',
    '15': 'Campania',
    '16': 'Puglia',
    '17': 'Basilicata',
    '18': 'Calabria',
    '19': 'Sicilia',
    '20': 'Sardegna',
    '21': 'P.A. Bolzano',
    '22': 'P.A. Trento',
    'all': 'Italy',
}

IT_REGION_CODES = {
    v : k
    for k, v in IT_REGION_NAMES.items()
}

IT_REGION_POPULATION = {
    'all': 60_244_639,
    '1': 4_341_375,
    '2': 125_501,
    '3': 10_103_969,
    '5': 4_907_704,
    '6': 1_211_357,
    '7': 1_543_127,
    '8': 4_467_118,
    '9': 3_722_729,
    '10': 880_285,
    '11': 1_518_400,
    '12': 5_865_544,
    '13': 1_305_770,
    '14': 302_265,
    '15': 5_785_861,
    '16': 4_008_296,
    '17': 556_934,
    '18': 1_924_701,
    '19': 4_968_410,
    '20': 1_630_474,
    '21': 532_080,
    '22': 542_739,
}

LABEL_TRANSLATIONS = {
    "curves_ylabel": "Giornalieri",
    "testcounts_ylabel": "Tests",
    "probability_ylabel": "Probabilità\n$R_t$>1",
    "rt_ylabel": "$R_t$",
    "curve_infections": "Infezioni",
    "curve_adjusted": "Test positivi attesi",
    "bar_positive": "Test positivi confermati",
    "bar_actual_tests": "Dati",
    "curve_predicted_tests": "Previsione",
}
IT_REGIONS = list(IT_REGION_NAMES.keys())


def forecast_IT(df: pandas.DataFrame):
    """ Applies testcount interpolation/extrapolation.

    Currently this assumes the OWID data, which only has an "all" region.
    In the future, this should be replaced with more fine graned data loading!
    """
    # forecast with existing data
    df['predicted_new_tests'], results = preprocessing.predict_testcounts_all_regions(df, 'IT')
    # interpolate the initial testing ramp-up to account for missing data
    df_region = df.xs('all')
    df_region.loc['2020-01-01', 'predicted_new_tests'] = 0
    df_region.predicted_new_tests = df_region.predicted_new_tests.interpolate('linear')
    df_region['region'] = 'all'
    df = df_region.reset_index().set_index(['region', 'date'])    
    return df, results


from .. import data
data.set_country_support(
    country_alpha2="IT",
    compute_zone=data.Zone.Europe,
    region_name=IT_REGION_NAMES,
    region_population=IT_REGION_POPULATION,
    fn_load=ourworldindata.create_loader_function("IT"),
    fn_process=forecast_IT,
)
