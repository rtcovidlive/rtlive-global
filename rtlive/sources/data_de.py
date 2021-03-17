""" Data loader for Germany.

Depends on "202d-mm-dd_tests_daily_BL.CSV" files to be present in the
/data folder at the root of the rtlive-global repository.

These files are currently not publicly available, but their structure is expected
like in the following example. Date formatting must be 2020-03-23 or 23.09.2020.

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

from . import ourworldindata

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
    candidate_files = {
        *dp_testcounts.glob(r'*tests_daily_BL.csv'),
        *dp_testcounts.glob(r'*tests_daily_BL.CSV'),
    }
    for fp in sorted(candidate_files):
        file_date = pandas.Timestamp(fp.name[:10])
        if not take_latest:
            if file_date < run_date:
               fp_testcounts = fp
        else:
            fp_testcounts = fp
    if not fp_testcounts:
        raise FileNotFoundError(f'No testcounts file found in {dp_testcounts} for {date_str}')
    _log.info('Reading testcounts from %s', fp_testcounts)

    # detect the datetime format
    iso_format = True
    with open(fp_testcounts, encoding="latin-1") as tcfile:
        tcfile.readline()
        line2 = tcfile.readline()
        iso_format = ";2020-" in line2
        _log.info("Detected iso_format=%s from line %s", iso_format, line2)

    df_testcounts = pandas.read_csv(
        fp_testcounts,
        sep=';', decimal=',',
        encoding='unicode_escape',
        parse_dates=[1],
        # at least one testcount CSV wasn't formatted according to ISO 8603
        dayfirst=not iso_format,
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

     # Get the sparse data of total tests from OWID
    df_owid = get_owid_summarized_totals(run_date)
    df_merged = df_merged.assign(owid_total_tests=df_owid)
    return df_merged


def forecast_DE(df: pandas.DataFrame) -> typing.Tuple[pandas.DataFrame, typing.Dict[str, preprocessing.ForecastingResult]]:
    """ Applies testcount interpolation/extrapolation to German data.
    """
    # forecast with existing data
    df['predicted_new_tests_raw'], results = preprocessing.predict_testcounts_all_regions(df, 'DE')

    # scale the daily forecast by OWID summary reports (RKI weekly test report)
    df_factors = calculate_daily_scaling_factors(
        forecasted_daily_tests=df.loc['all', 'predicted_new_tests_raw'],
        sparse_reported_totals=df.loc['all', 'owid_total_tests']
    )
    df["scaling_factor"] = numpy.nan
    df["predicted_new_tests"] = numpy.nan
    # the scaling factor calculated from "all"-level forecasts and total reports is used
    # for all regions, because regional totals are currently not available from OWID
    for region in numpy.unique(df.index.get_level_values("region")):
        # the scaling factor column will be included in the result
        sfs = df_factors.scaling_factor
        df.loc[pandas.IndexSlice[region, list(sfs.index)], 'scaling_factor'] = sfs.to_numpy()
    df['predicted_new_tests'] =  df['predicted_new_tests_raw'] * df['scaling_factor']
    return df, results


def get_owid_summarized_totals(run_date):
    """ Get the total amount of tests reported to OWID.
    
    At the moment only the `all` region is included.
    At time of writing only sundays have a value that is not NaN.
    """
    f = ourworldindata.create_loader_function("DE")
    data = f(run_date)
    return data.total_tests.rename("owid_total_tests").to_frame()


def calculate_daily_scaling_factors(
    *,
    forecasted_daily_tests: pandas.Series,
    sparse_reported_totals: pandas.Series
) -> pandas.DataFrame:
    """ Scale the daily test counts per region coming from the Prophet forecast by the test count report 
    from OurWorldInData, which is available before the real daily testcounts are known.
    
    Parameters
    ----------
    forecasted_daily_tests: pandas.Series
        Series from the Prophet forecast containing the confirmed daily test counts
        sent from RKI privately as well as predicted test counts.
        Both data are scaled by the total reported tests by OurWorldInData (OWID!
    sparse_reported_totals : pandas.Series
        Series from OWID containing total test counts summarized for a period of time 
        (mostly one week) for all of Germany. It is expected to contain NaN gaps in the data.
        The differences between this report  and the forecast data will be used to make sure
        the total number of tests in the forecast  matches the OWID data.

    Returns
    -------
    correction_factor: pandas.DataFrame
        The scaling factor for all dates including the future.
    """
    assert isinstance(forecasted_daily_tests, pandas.Series)
    assert isinstance(sparse_reported_totals, pandas.Series)
    
    df_factors = pandas.DataFrame(
        index=forecasted_daily_tests.index,
        columns=["sum_predicted", "diff_reported", "scaling_factor"]
    )
    sum_dates = list(sparse_reported_totals.dropna().index)
    for dfrom, dto in zip(sum_dates[:-1], sum_dates[1:]):
        day = pandas.Timedelta("1D")
        interval = slice(dfrom + day, dto)
        # sum over the predictions in this inverval
        sum_predicted = forecasted_daily_tests.loc[dfrom + day : dto].sum()
        df_factors.loc[interval, ["sum_predicted"]] = sum_predicted

        # diff of the reports
        prevtot = float(sparse_reported_totals.loc[dfrom])
        nexttot = float(sparse_reported_totals.loc[dto])
        diff_reported = nexttot - prevtot
        df_factors.loc[interval, ["diff_reported"]] = diff_reported

    df_factors["scaling_factor"] = df_factors.diff_reported / df_factors.sum_predicted
    # extrapolate backwards at the beginning
    first = df_factors.dropna().iloc[0]
    df_factors.loc[:first.name, "scaling_factor"] = first.scaling_factor
    # continue into the future with the last known scaling factor
    last = df_factors.dropna().iloc[-1]
    df_factors.loc[last.name:, "scaling_factor"] = last.scaling_factor
    return df_factors


def estimate_test_percentages_for_regions(df: pandas.DataFrame) -> pandas.Series:
    """ Calculates the fraction of tests per region.

    Uses the 7 days up to the last day for which daily new_test data is available for all regions.

    WARNING: If any region has a gap _before_ the last day for which all of them have data, this
    function will fail to return the correct result.
    
    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe containing the new_test column with a [region, date] index as genereated by get_testcounts_DE. An `all` region has to be included.
        
    Returns
    -------
    region_test_percentages: pandas.Series
        Region-indexed series of fractions of all tests.
    """
    rows_with_testcounts = df.new_tests[~df.new_tests.isna()]
    last_date_with_testcounts_for_all_regions = rows_with_testcounts.groupby('region').tail(1).reset_index()['date'].min()

    # select the last 7 days up to the latest testcount data point
    last_week_of_testcounts = slice(last_date_with_testcounts_for_all_regions - pandas.Timedelta('6D'), last_date_with_testcounts_for_all_regions)

    # Then calculate the sum of tests one week up to that date
    testcounts_in_last_daily_data = df.new_tests.xs(last_week_of_testcounts, level='date').groupby('region').sum()

    # Finally convert absolutes to fractions
    return testcounts_in_last_daily_data / testcounts_in_last_daily_data['all']


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
    mapping = {
        "Datum des Erkrankungsbeginns": "date",
        "Punktschätzer der Anzahl Neuerkrankungen (ohne Glättung)": "new_cases",
        "Untere Grenze des 95%-Prädiktionsintervalls der Anzahl Neuerkrankungen (ohne Glä": "new_cases_lower",
        "Untere Grenze des 95%-Prädiktionsintervalls der Anzahl Neuerkrankungen (ohne Glättung)": "new_cases_lower",
        "Obere Grenze des 95%-Prädiktionsintervalls der Anzahl Neuerkrankungen (ohne Glät": "new_cases_upper",
        "Obere Grenze des 95%-Prädiktionsintervalls der Anzahl Neuerkrankungen (ohne Glättung)": "new_cases_upper",
        "Punktschätzer der Anzahl Neuerkrankungen": "new_cases_smooth",
        "Untere Grenze des 95%-Prädiktionsintervalls der Anzahl Neuerkrankungen": "new_cases_smooth_lower",
        "Obere Grenze des 95%-Prädiktionsintervalls der Anzahl Neuerkrankungen": "new_cases_smooth_upper",
        "Punktschätzer der Reproduktionszahl R": "r4",
        "Punktschätzer der 4-Tages R-Wert": "r4",
        "Punktschätzer der 4-Tage R-Wert": "r4",
        "Punktschätzer des 4-Tage-R-Wertes": "r4",
        "Untere Grenze des 95%-Prädiktionsintervalls der Reproduktionszahl R": "r4_lower",
        "Untere Grenze des 95%-Prädiktionsintervalls der 4-Tages R-Wert": "r4_lower",
        "Untere Grenze des 95%-Prädiktionsintervalls der 4-Tage R-Wert": "r4_lower",
        "Untere Grenze des 95%-Prädiktionsintervalls des 4-Tage-R-Wertes": "r4_lower",
        "Obere Grenze des 95%-Prädiktionsintervalls der Reproduktionszahl R": "r4_upper",
        "Obere Grenze des 95%-Prädiktionsintervalls der 4-Tages R-Wert": "r4_upper",
        "Obere Grenze des 95%-Prädiktionsintervalls der 4-Tage R-Wert": "r4_upper",
        "Obere Grenze des 95%-Prädiktionsintervalls des 4-Tage-R-Wertes": "r4_upper",
        "Punktschätzer des 7-Tage-R Wertes": "r7",
        "Punktschätzer des 7-Tage-R-Wertes": "r7",
        "Untere Grenze des 95%-Prädiktionsintervalls des 7-Tage-R Wertes": "r7_lower",
        "Untere Grenze des 95%-Prädiktionsintervalls des 7-Tage-R-Wertes": "r7_lower",
        "Obere Grenze des 95%-Prädiktionsintervalls des 7-Tage-R Wertes": "r7_upper",
        "Obere Grenze des 95%-Prädiktionsintervalls des 7-Tage-R-Wertes": "r7_upper",
    }
    for file in DATA_DIR.iterdir():
        if 'Nowcasting' in str(file) and date_str in str(file):
            data_rki = pandas.read_excel(
                file, sheet_name='Nowcast_R',
                na_values=".",
            ).rename(columns=mapping)
            # apply type conversions and set index
            data_rki = data_rki.set_index("date")
            data_rki.index = pandas.to_datetime(
                data_rki.index,
                dayfirst=isinstance(data_rki.index[0], str) and ".2020" in data_rki.index[0]
            )
            if isinstance(data_rki["r4"][10], str) and "," in data_rki["r4"][10]:
                # thousands="." messes with the date parsing, so make a backup
                # copy of the previously parsed dates and re-apply them later.
                dates = data_rki.index
                # convert german floats
                data_rki = pandas.read_excel(
                    file, sheet_name='Nowcast_R',
                    thousands=".", na_values=".",
                ).rename(columns=mapping)
                data_rki.index = dates
                # brute force everything to floats
                data_rki = data_rki.apply(lambda col: [float(str(v).replace(",", ".")) for v in col])

    result = {}
    if data_rki is not None:
        params = {
            'Rt': ('r4', '$R_t$', 'green'),
            'Rt_7': ('r7', label_week, 'orange')
        }
        for (identifier, label, color) in params.values():
            r_values = data_rki[identifier]
            lower = data_rki[f'{identifier}_lower']
            upper = data_rki[f'{identifier}_upper']
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
