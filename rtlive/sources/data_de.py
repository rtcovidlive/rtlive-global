""" Data loader for Germany.

Depends on "2020-mm-dd_tests_daily_BL.CSV" files to be present in the
/data folder at the root of the rtlive-global repository.

These files are currently not publicly available, but their structure is expected
like in the following example. Date formatting must be 2020-03-23 or 23.09.2020 for
files specified in the NON_ISO_TESTCOUNTS constant.

Bundesland;Datum;Testungen (auf Fünfzig aufgerundet );Anteil positiv (auf zwei Stellen gerundet )
...
"Baden-Württemberg";2020-04-07;1000;0,11
"Baden-Württemberg";2020-04-08;1250;0,11
"Baden-Württemberg";2020-04-09;850;0,12
"Baden-Württemberg";2020-04-10;1200;0,09
"Baden-Württemberg";2020-04-11;1300;0,07
...
"nicht zugeordnet";2020-05-18;6550;0,1
"nicht zugeordnet";2020-05-19;7350;0,05
"nicht zugeordnet";2020-05-20;2850;0,02
"nicht zugeordnet";2020-05-21;1000;0,01
...
"""
import datetime
import logging
import matplotlib
import numpy
import os
import pandas
import pathlib
import requests
import shutil
import tempfile
import typing

from .. import preprocessing


_log = logging.getLogger(__file__)

DATA_DIR = pathlib.Path(pathlib.Path(__file__).parent.parent.parent, "data")
if not DATA_DIR.exists():
    _log.info("Data directory expected at '%s'. Creating...", DATA_DIR)
    DATA_DIR.mkdir()

DE_REGION_NAMES = {
    'BW': 'Baden-Württemberg',
    'BY': 'Bayern',
    'BE': 'Berlin',
    'BB': 'Brandenburg',
    # commented out to exclude from DAG:
    #'HB': 'Bremen',
    'HH': 'Hamburg',
    'HE': 'Hessen',
    'MV': 'Mecklenburg-Vorpommern',
    'NI': 'Niedersachsen',
    'NW': 'Nordrhein-Westfalen',
    'RP': 'Rheinland-Pfalz',
    #'SL': 'Saarland',
    'SN': 'Sachsen',
    'ST': 'Sachsen-Anhalt',
    'SH': 'Schleswig-Holstein',
    'TH': 'Thüringen',
    'all': 'Germany',
}
DE_REGION_CODES = {
    v : k
    for k, v in DE_REGION_NAMES.items()
}
DE_REGION_POPULATION = {
    'all': 83_166_711,
    'BB': 2_521_893,
    'BE': 3_669_491,
    'BW': 11_100_394,
    'BY': 13_124_737,
    'HB': 681_202,
    'HE': 6_288_080,
    'HH': 1_847_253,
    'MV': 1_608_138,
    'NI': 7_993_608,
    'NW': 17_947_221,
    'RP': 4_093_903,
    'SH': 2_903_773,
    'SL': 986_887,
    'SN': 4_071_971,
    'ST': 2_194_782,
    'TH': 2_133_378,
}
LABEL_TRANSLATIONS = {
    "curves_ylabel": "Pro Tag",
    "testcounts_ylabel": "Testungen",
    "probability_ylabel": "Wahrscheinlichkeit\n$R_t$>1",
    "rt_ylabel": "$R_t$",
    "curve_infections": "Infektionen",
    "curve_adjusted": "Erwartete positive Tests ohne Dunkelziffer",
    "bar_positive": "Bestätigte positive Tests",
    "bar_actual_tests": "Daten",
    "curve_predicted_tests": "Vorhersage",
}
DE_REGIONS = list(DE_REGION_NAMES.keys())
# this constant is used in the Airflow DAG to save a copy of the raw data for archiving
CSV_SAVEPATH = None
# at least one testcount CSV wasn't formatted according to ISO 8603
NON_ISO_TESTCOUNTS = { "2020-10-06 tests_daily_BL.CSV", "2020-11-13 tests_daily_BL.CSV" }


def get_data_DE(run_date) -> pandas.DataFrame:
    """ Data loader for Germany.

    Parameters
    ----------
    run_date : datetime.datetime
        date for which the data shall be downloaded

    Returns
    -------
    df : pandas.DataFrame
        table with columns as required by rtlive/data.py API
    """
    df_positives = get_positives_DE(run_date)
    df_testcounts = get_testcounts_DE(run_date)

    case_regions = set(df_positives.reset_index().region)
    test_regions = set(df_testcounts.reset_index().region)
    _log.warning('Missing test counts for %s', str(case_regions.difference(test_regions)))

    df_result = pandas.merge(df_positives, df_testcounts, left_index=True, right_index=True, how='outer').sort_index()
    assert len(set(df_result.reset_index().region))
    return df_result


def get_positives_DE(run_date) -> pandas.DataFrame:
    """ Retrieves table of positives & deaths for all German regions.

    Parameters
    ----------
    run_date : pandas.Timestamp
        use the data as it was release on that day

    Returns
    -------
    result : pandas.DataFrame
        [region, date]-indexed table that has rows for every region & date combination in [2020-03-01, run_date - 1]
        contains columns "positive" and "deaths" that are the number of NEW positives/deaths for each day/region
        "all" region is the sum over all states.
    """
    date_str = run_date.strftime('%Y-%m-%d')
    with tempfile.TemporaryDirectory() as td:
        fp_tempfile = pathlib.Path(td, 'data_arcgis.csv')
        if run_date.date() < datetime.datetime.utcnow().date():
            release_id = (run_date + pandas.DateOffset(1)).strftime('%Y-%m-%d')
            release_url = f'https://github.com/ihucos/rki-covid19-data/releases/download/{release_id}/data.csv'
            # For explanations of the columns, see https://www.arcgis.com/home/item.html?id=f10774f1c63e40168479a1feb6c7ca74
            # the CSV is automatically released at 01 AM Berlin local time, but has one day offset to the RKI data
            _log.info('Downloading German data from %s', release_url)
            with open(fp_tempfile, 'wb') as file:
                file.write(requests.get(release_url).content)
            encoding = 'utf-8'       
        else:
            _log.info('Downloading RKI COVID-19 dataset from ArcGIS')
            from arcgis.gis import GIS
            anon_gis = GIS()
            features = anon_gis.content.get('dd4580c810204019a7b8eb3e0b329dd6').tables[0].query()
            features.save(save_location=td, out_name='download.csv')
            shutil.copy2(os.path.join(td, 'download.csv'), fp_tempfile)
            encoding = 'unicode_escape'  if os.name == 'nt' else 'utf-8'
        if CSV_SAVEPATH:
            shutil.copy2(fp_tempfile, CSV_SAVEPATH)  
        df = pandas.read_csv(
            fp_tempfile,
            usecols=['Bundesland', 'Meldedatum', 'Datenstand', 'AnzahlFall', 'AnzahlTodesfall'],
            encoding=encoding
        )
    _log.info('Data was loaded for the following regions: %s', df.Bundesland.unique())
    df.Meldedatum = pandas.to_datetime(df.Meldedatum, unit='ms')
    assert len(set(df.Datenstand)) == 1
    datenstand = df.Datenstand[0]
    assert run_date.strftime('%d.%m.%Y') in df.Datenstand[0]

    # transform to multi-indexed dataframe with required columns
    _log.info('Transforming to multi-indexed dataframe')
    df_sparse = df.rename(columns={
        'Meldedatum': 'date',
        'Bundesland': 'region',
        'AnzahlFall': 'new_cases',
        'AnzahlTodesfall': 'new_deaths',
    }).replace(DE_REGION_CODES).groupby(['region', 'date']).sum().sort_index()

    # make sure that the result has rows for every region/date combination.
    _log.info('Inserting 0-counts for missing dates')
    full_index = pandas.date_range(
        '2020-03-01',
        run_date - pandas.DateOffset(2)
        # ToDo: use max(run_date-2, date in data)
        #max(run_date - pandas.DateOffset(2), 
    )
    df_full = pandas.concat({
        region : df_sparse.xs(region).reindex(full_index, fill_value=0)
        for region in DE_REGIONS
        if region != 'all'
    }, names=['region', 'date'])

    # add region "all" that is the sum over all states
    df_all = df_full.sum(level='date')
    df_all.insert(0, column='region', value='all')
    df_all = df_all.reset_index().set_index(['region', 'date'])
    df_merged = pandas.concat([df_full, df_all]).sort_index()

    return df_merged


def get_testcounts_DE(run_date, take_latest:bool=True) -> pandas.DataFrame:
    """ Builds a table of testcount data for German states.
    Saarland (SL) and Bremen (HB) are missing from the result, but their
    contribution is inclued in the "all" region that sums over all states.

    Parameters
    ----------
    run_date : pandas.Timestamp
    take_latest : bool
        if True, the most recent testcount CSV is used
        otherwise, the last testcount CSV before run_date is used

    Returns
    -------
    df_testcounts : pandas.DataFrame
        [region, date]-indexed testcount information
        The reported numbers are just a subset of actual testcounts, due to non-mandatory reporting.
        "all" region is included as the sum over all states.
        Saarland (SL) and Bremen (HB) are missing from region-level reporting.
    """
    date_str = run_date.strftime('%Y-%m-%d')

    # find the latest tescounts file before `run_date`
    dp_testcounts = pathlib.Path(DATA_DIR)
    fp_testcounts = None
    for fp in sorted(dp_testcounts.glob(r'*tests_daily_BL.CSV')):
        file_date = pandas.Timestamp(fp.name[:10])
        if not take_latest:
            if file_date < run_date:
               fp_testcounts = fp
        else:
            fp_testcounts = fp
    if not fp_testcounts:
        raise FileNotFoundError(f'No testcounts file found in {dp_testcounts} for {date_str}')
    _log.info('Reading testcounts from %s', fp_testcounts)

    df_testcounts = pandas.read_csv(
        fp_testcounts,
        sep=';', decimal=',',
        encoding='unicode_escape',
        parse_dates=[1],
        # at least one testcount CSV wasn't formatted according to ISO 8603
        dayfirst=(fp_testcounts.name in NON_ISO_TESTCOUNTS),
    ).rename(columns={
        'Bundesland': 'region',
        'Datum': 'date',
        'Testungen (auf Fünfzig aufgerundet )': 'new_tests',
        'Anteil positiv (auf zwei Stellen gerundet )': 'positive_fraction',
    }).replace(DE_REGION_CODES)
    df_testcounts.date = pandas.to_datetime(df_testcounts.date)
    df_testcounts = df_testcounts.set_index(['region', 'date']).sort_index()

    # add region "all" that is the sum over all states
    df_all = df_testcounts.sum(level='date')
    df_all.insert(0, column='region', value='all')
    df_all = df_all.reset_index().set_index(['region', 'date'])
    df_merged = pandas.concat([df_testcounts, df_all]).sort_index()

    # drop non-associated AFTER calculating the sum
    df_merged.drop(index='nicht zugeordnet', inplace=True)
    return df_merged


def forecast_DE(df: pandas.DataFrame):
    """ Applies testcount interpolation/extrapolation to french data.

    Currently this assumes the OWID data, which only has an "all" region.
    In the future, this should be replaced with more fine graned data loading!
    """
    # forecast with existing data
    df['predicted_new_tests'], results = preprocessing.predict_testcounts_all_regions(df, 'DE')
    return df, results


def download_rki_nowcast(run_date, target_filename) -> pathlib.Path:
    """ Downloads RKI nowcasting data unless [target_filename] already exists.

    Parameters
    ----------
    run_date : date-like
        the date for which to download the nowcast
    target_filename : path-like
        filename/path to save to

    Raises
    ------
    FileExistsError
        if the [target_filename] already exists

    Returns
    -------
    filepath : pathlib.Path
        points to the downloaded file
    """
    today = datetime.date.today().strftime('%Y-%m-%d')
    if str(run_date) != today: 
        raise Exception("Can only download for today.")
    url = 'https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Projekte_RKI/Nowcasting_Zahlen.xlsx;jsessionid=BA3A51DFC0A5E3FEE716B2966FDC5E54.internet071?__blob=publicationFile'
    filepath = pathlib.Path(target_filename)
    if not filepath.exists():
        with open(filepath, 'wb') as file:
            file.write(requests.get(url).content)
    else:
        raise FileExistsError(f'Target file {filepath} already exists.')
    return filepath


def get_rki_nowcast(date_str: str, label_german:bool=False):
    """ Helper function to parse RKI nowcasting data from cached files.

    Parameters
    ----------
    date_str : str
        ISO datetime for which the latest nowcast data shall be loaded
    label_german : bool
        if True, the keys in the result dictionary will be German

    Returns
    -------
    result : dict
        maps series names to tuple of
            r_values : pandas.Series
            lower : optional, pandas.Series
            upper : optional, pandas.Series
            color : str
    """
    if label_german:
        label_week = '$R_t$ 7 Tage'
    else:
        label_week = '$R_t$ 7 days'

    # find & read the relevant nowcast XLSX
    data_rki = None
    for file in DATA_DIR.iterdir():
        if 'Nowcasting' in str(file) and date_str in str(file):
            data_rki = pandas.read_excel(file, sheet_name='Nowcast_R')
            data_rki = data_rki.rename(columns={
                "Datum des Erkrankungsbeginns": "date"
            }).set_index("date")

    result = {}
    if data_rki is not None:
        params = {
            'Rt': ('der Reproduktionszahl R', '$R_t$', 'green'),
            'Rt_7': ('des 7-Tage-R Wertes', label_week, 'orange')
        }
        for (identifier, label, color) in params.values():
            r_values = data_rki[f'Punktschätzer {identifier}']
            lower = data_rki[f'Untere Grenze des 95%-Prädiktionsintervalls {identifier}']
            upper = data_rki[f'Obere Grenze des 95%-Prädiktionsintervalls {identifier}']
            result[f'(RKI) {label}'] = (r_values, lower, upper, color)
    return result


from .. import data
data.set_country_support(
    country_alpha2="DE",
    compute_zone=data.Zone.Europe,
    region_name=DE_REGION_NAMES,
    region_population=DE_REGION_POPULATION,
    fn_load=get_data_DE,
    fn_process=forecast_DE,
)
