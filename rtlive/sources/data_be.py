import logging
import pandas
import datetime
import requests
import io

from typing import Dict, Tuple, Union

from .. import preprocessing

_log = logging.getLogger(__file__)

# From https://en.wikipedia.org/wiki/Provinces_of_Belgium
BE_REGION_NAMES = {
    'all': 'Belgium',
    'FLA': 'Vlaanderen',
    'WAL': 'Wallonie',
    'BRU': 'Brussel',
    'ANT': 'Antwerpen',
    'LIM': 'Limburg',
    'EFL': 'Oost-Vlaanderen',
    'FBR': 'Vlaams-Brabant',
    'WFL': 'West-Vlaanderen',
    'HAI': 'Hainaut',
    'LIE': 'Liège',
    'LUX': 'Luxembourg',
    'NAM': 'Namur',
    'WBR': 'Brabant wallon',
}

# Province and region codes
# [ISO 3166-2:BE](https://en.wikipedia.org/wiki/ISO_3166-2:BE#Provinces) has no english codes
# Mapping of the keys in columns 'REGION' and 'PROVINCE' in the input file to a short code.
BE_REGION_INPUT_ABBR = {
    'all': 'all',
    'Flanders': 'FLA',
    'Wallonia': 'WAL',
    'Brussels': 'BRU',
    'Antwerpen': 'ANT',
    'Limburg': 'LIM',
    'OostVlaanderen': 'EFL',
    'VlaamsBrabant': 'FBR',
    'WestVlaanderen': 'WFL',
    'Hainaut': 'HAI',
    'Liège': 'LIE',
    'Luxembourg': 'LUX',
    'Namur': 'NAM',
    'BrabantWallon': 'WBR',   
}

BE_REGION_CODES = {
    v : k
    for k, v in BE_REGION_NAMES.items()
}

# Source: https://www.ibz.rrn.fgov.be/fileadmin/user_upload/fr/pop/statistiques/population-bevolking-20200101.pdf
BE_REGION_POPULATION = {
    'all': 11_476_279, # Belgium
    'FLA':  6_623_505,
    'WAL':  3_641_748,
    'BRU':  1_211_026,
    'ANT':  1_867_366,
    'LIM':    876_785,
    'EFL':  1_524_077,
    'FBR':  1_155_148,
    'WFL':  1_200_129,
    'HAI':  1_345_270,
    'LIE':  1_108_481,
    'LUX':    286_571,
    'NAM':    495_474,
    'WBR':    405_952
}

def get_data_BE(run_date: pandas.Timestamp) -> pandas.DataFrame:
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
    
    def redistribute(group: pandas.DataFrame, col: str) -> pandas.Series:
        gdata = group.groupby('REGION')[col].sum()
        gdata.loc['Brussels'] += gdata.loc['Nan'] * (gdata.loc['Brussels']/(gdata.loc['Brussels'] + gdata.loc['Flanders'] + gdata.loc['Wallonia']))
        gdata.loc['Flanders'] += gdata.loc['Nan'] * (gdata.loc['Flanders']/(gdata.loc['Brussels'] + gdata.loc['Flanders'] + gdata.loc['Wallonia']))
        gdata.loc['Wallonia'] += gdata.loc['Nan'] * (gdata.loc['Wallonia']/(gdata.loc['Brussels'] + gdata.loc['Flanders'] + gdata.loc['Wallonia']))
        gdata.drop(index='Nan', inplace=True)
        gdata = gdata.fillna(0).round(0).astype(int)
        return gdata
    
    if run_date.date() > datetime.date.today():
        raise ValueError('Run date is in the future. Nice try.')
    if run_date.date() < datetime.date.today():
        # TODO: implement downloading of historic data
        raise NotImplementedError(
            'Downloading with a run_date is not yet supported. '
            f'Today: {datetime.date.today()}, run_date: {run_date}'
        )
        
    # Download data from Sciensano
    content = requests.get('https://epistat.sciensano.be/Data/COVID19BE_tests.csv', verify=False,).content
    df_tests = pandas.read_csv(
            io.StringIO(content.decode('utf-8')),
            sep=',',
            parse_dates=['DATE'],
            usecols=['DATE', 'REGION', 'PROVINCE', 'TESTS_ALL_POS', 'TESTS_ALL']
    )
    # Reformat data into Rtlive.de format at country level all
    df_tests_per_all_day = (df_tests
       .assign(region='all')
       .groupby('DATE', as_index=False)
       .agg(new_cases=('TESTS_ALL_POS', 'sum'), new_tests=('TESTS_ALL', 'sum'), region=('region', 'first'))
       .rename(columns={'DATE':'date'})
       .set_index(['region', "date"])
       .sort_index()
    )
    # Redistribute the nan for the column TESTS_ALL_POS for regions Flanders, Wallonia and Brussels
    df_tests_positive = (df_tests
        .fillna('Nan')
        .groupby(['DATE'])
        .apply(redistribute, 'TESTS_ALL_POS')
        .stack()
        .reset_index()
        .rename(columns={'DATE':'date', 'REGION':'region', 0:'new_cases'})
    )
    # Redistribute the nan for the column TESTS_ALL for regions Flanders, Wallonia and Brussels
    df_tests_all = (df_tests
        .fillna('Nan')
        .groupby(['DATE'])
        .apply(redistribute, 'TESTS_ALL')
        .stack()
        .reset_index()
        .rename(columns={'DATE':'date', 'REGION':'region', 0:'new_tests'})
    )
    
    # Combine the total number of tests and the number of positive tests into a basetable
    df_tests_per_region_day = pandas.concat([df_tests_all, df_tests_positive['new_cases']], axis=1).set_index(['region', 'date'])
    
    # Test per province (Ignore the nan's for the moment)
    df_tests_per_province_day = (df_tests
       .groupby(['PROVINCE', 'DATE'], as_index=False)
       .agg(new_cases=('TESTS_ALL_POS', 'sum'), new_tests=('TESTS_ALL', 'sum'))
       .rename(columns={'DATE':'date', 'PROVINCE':'region'})
       .set_index(["region", "date"])
       .sort_index()
    )
    
    # Combine the results at country level with region level
    data = pandas.concat([df_tests_per_all_day, df_tests_per_region_day, df_tests_per_province_day], axis=0).sort_index()
    
    data.index = data.index.set_levels(data.index.levels[0].map(BE_REGION_INPUT_ABBR.get), 'region')
    
    assert isinstance(data, pandas.DataFrame)
    assert data.index.names == ('region', 'date')
    assert 'new_cases' in data.columns, f'Columns were: {data.columns}'
    assert 'new_tests' in data.columns, f'Columns were: {data.columns}'
    for col in ['new_cases', 'new_tests']:
        if any(data[col] < 0):
            _log.warning(
                f'Column {col} has {sum(data[col] < 0)} negative entries!! Overriding with NaN...'
            )
            data.loc[data[col] < 0, col] = numpy.nan

    return data


def forecast_BE(df: pandas.DataFrame) -> Tuple[pandas.DataFrame, dict]:
    """
    Applies test count interpolation/extrapolation to Belgium data.
    Parameters
    ----------
    df : pandas.DataFrame
        Data as returned by data loader function.
    Returns
    -------
    df : pandas.DataFrame
        Input dataframe with a new column "predicted_new_tests" and an index expanded back to
        01/01/2020 (filled with zeros until 13/05/2020) to account for the absence of tests in this
        period.
    results : dict
        The fbprophet results by region
    """
    # forecast with existing data
    df["predicted_new_tests"], results = preprocessing.predict_testcounts_all_regions(
        df, "BE"
    )
    # interpolate the initial testing ramp-up to account for missing data
    df_list = []
    for region in df.index.get_level_values(level="region").unique():
        df_region = df.xs(region).copy()

        df_complement = pandas.DataFrame(
            index=pandas.date_range(
                start="2020-01-01",
                end=df_region.index.get_level_values(level="date")[0]
                - pandas.DateOffset(1, "D"),
                freq="D",
            ),
            columns=df_region.columns,
        )
        df_complement["predicted_new_tests"] = 0

        df_region = df_complement.append(df_region)
        df_region.index.name = "date"
        df_region.predicted_new_tests = df_region.predicted_new_tests.interpolate(
            "linear"
        )

        df_region["region"] = region
        df_list.append(df_region.reset_index().set_index(["region", "date"]))

    return pandas.concat(df_list), results


from .. import data
data.set_country_support(
    country_alpha2='BE',
    compute_zone=data.Zone.Europe,
    region_name=BE_REGION_NAMES,
    region_population=BE_REGION_POPULATION,
    fn_load=get_data_BE,
    fn_process=forecast_BE,
)
