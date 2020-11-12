import logging
import pandas
import numpy

from .. import preprocessing

_log = logging.getLogger(__file__)

# From https://en.wikipedia.org/wiki/Provinces_of_Belgium
BE_REGION_NAMES = {
    'all': 'Belgium',
    '01': 'Flanders',
    '02': 'Wallonia',
    '03': 'Brussels',
    '04': 'Antwerp',
    '05': 'Limburg',
    '06': 'East Flanders',
    '07': 'Flemish Brabant',
    '08': 'West Flanders',
    '09': 'Hainaut',
    '10': 'Liège',
    '11': 'Luxembourg',
    '12': 'Namur',
    '13': 'Walloon Brabant',
}

BE_REGION_ABBR = {
    '01': 'BEL',
    '02': 'FLA',
    '03': 'WAL',
    '04': 'BRU',
    '05': 'ANT',
    '06': 'LIM',
    '07': 'EFL',
    '08': 'FBR',
    '09': 'WFL',
    '10': 'LIE',
    '11': 'LUX',
    '12': 'NAM',
    '13': 'WBR',
    'all': 'all',
}

BE_REGION_CODES = {
    v : k
    for k, v in BE_REGION_NAMES.items()
}

# Source: https://www.ibz.rrn.fgov.be/fileadmin/user_upload/fr/pop/statistiques/population-bevolking-20200101.pdf
BE_REGION_POPULATION = {
    'all': 11_476_279,
    '01':   6_623_505,
    '02':   3_641_748,
    '03':   1_211_026,
    '04':   1_867_366,
    '05':     876_785,
    '06':   1_155_148,
    '07':   1_200_129,
    '08':   1_345_270,
    '09':   1_108_481,
    '10':     624_841,
    '11':     286_571,
    '12':     495_474,
    '13':     405_952
}

def get_data_BE(run_date) -> pandas.DataFrame:
    """
    Retrieve daily (run_date) regions and append national data (key 'all') to it
    Parameters
    ----------
    run_date : pandas.Timestamp
        date for which the data shall be downloaded
    
    Returns
    -------
    df : pandas.DataFrame
        table with columns as required by rtlive/data.py API
    """
    
    def redistribute(g, col):
        gdata = g.groupby('REGION')[col].sum()
        gdata.loc['Brussels'] += gdata.loc['Nan'] * (gdata.loc['Brussels']/(gdata.loc['Brussels'] + gdata.loc['Flanders'] + gdata.loc['Wallonia']))
        gdata.loc['Flanders'] += gdata.loc['Nan'] * (gdata.loc['Flanders']/(gdata.loc['Brussels'] + gdata.loc['Flanders'] + gdata.loc['Wallonia']))
        gdata.loc['Wallonia'] += gdata.loc['Nan'] * (gdata.loc['Wallonia']/(gdata.loc['Brussels'] + gdata.loc['Flanders'] + gdata.loc['Wallonia']))
        gdata.drop(index='Nan', inplace=True)
        gdata = numpy.round(gdata.fillna(0)).astype(int)
        return gdata
    
    if run_date.date() > datetime.date.today():
        raise ValueError("Run date is in the future. Nice try.")
    if run_date.date() < datetime.date.today():
        # TODO: implement downloading of historic data
        raise NotImplementedError(
            "Downloading with a run_date is not yet supported. "
            f"Today: {datetime.date.today()}, run_date: {run_date}"
        )
        
    # Download data from Sciensano
    df_tests = pandas.read_csv('https://epistat.sciensano.be/Data/COVID19BE_tests.csv', parse_dates=['DATE'])
    # Reformat data into Rtlive.de format at country level
    df_tests_all_per_day = (df_tests
       .assign(region='all')
       .groupby('DATE', as_index=False)
       .agg(new_cases=('TESTS_ALL_POS', 'sum'), new_tests=('TESTS_ALL', 'sum'), region=('region', 'first'))
       .rename(columns={'DATE':'date'})
       .set_index(["region", "date"])
       .sort_index()
    )
    # Redistribute the nan for the column TESTS_ALL_POS at region level
    df_tests_positive = (df_tests
        .fillna('Nan')
        .groupby(['DATE'])
        .apply(redistribute, 'TESTS_ALL_POS')
        .stack()
        .reset_index()
        .rename(columns={'DATE':'date', 'REGION':'region', 0:'new_cases'})
    )
    # Redistribute the nan for the column TESTS_ALL at region level
    df_tests_all = (df_tests
        .fillna('Nan')
        .groupby(['DATE'])
        .apply(redistribute, 'TESTS_ALL')
        .stack()
        .reset_index()
        .rename(columns={'DATE':'date', 'REGION':'region', 0:'new_tests'})
    )
    
    # Combine the total number of tests and the number of positive tests into a basetable
    df_tests_per_province_day = pd.concat([df_tests_all, df_tests_positive['new_cases']], axis=1).set_index(['region', 'date'])
    
    data = pd.concat([df_tests_all_per_day, df_tests_per_province_day], axis=0)
    
    assert isinstance(data, pandas.DataFrame)
    assert data.index.names == ("region", "date")
    assert "new_cases" in data.columns, f"Columns were: {data.columns}"
    assert "new_tests" in data.columns, f"Columns were: {data.columns}"
    for col in ["new_cases", "new_tests"]:
        if any(data[col] < 0):
            _log.warning(
                f"Column {col} has {sum(data[col] < 0)} negative entries!! Overriding with NaN..."
            )
            data.loc[data[col] < 0, col] = numpy.nan

    return data


def forecast_BE(df: pandas.DataFrame):
    """ Applies testcount interpolation/extrapolation.

    Currently this assumes the OWID data, which only has an "all" region.
    In the future, this should be replaced with more fine graned data loading!
    """
    # forecast with existing data
    df['predicted_new_tests'], results = preprocessing.predict_testcounts_all_regions(df, 'BE')
    # interpolate the initial testing ramp-up to account for missing data
    df_region = df.xs('all')
    df_region.loc['2020-01-01', 'predicted_new_tests'] = 0
    df_region.predicted_new_tests = df_region.predicted_new_tests.interpolate('linear')
    df_region['region'] = 'all'
    df = df_region.reset_index().set_index(['region', 'date'])    
    return df, results


from .. import data
data.set_country_support(
    country_alpha2="BE",
    compute_zone=data.Zone.Europe,
    region_name=BE_REGION_NAMES,
    region_short_name=BE_REGION_ABBR,
    region_population=BE_REGION_POPULATION,
    fn_load=get_data_BE,
    fn_process=forecast_BE,
)
