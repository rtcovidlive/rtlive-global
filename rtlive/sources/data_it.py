import datetime
import io
import logging
import pandas
import requests

from .. import preprocessing

_log = logging.getLogger(__file__)

IT_DATA_BASE_PATH = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master"
IT_DATA_NATION_FILENAME = "/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale-%s.csv"
IT_DATA_REGION_FILENAME = "/dati-regioni/dpc-covid19-ita-regioni-%s.csv"

IT_REGION_NAMES = {
    '01': 'Piemonte',
    '02': 'Valle d\'Aosta',
    '03': 'Lombardia',
    '05': 'Veneto',
    '06': 'Friuli Venezia Giulia',
    '07': 'Liguria',
    '08': 'Emilia-Romagna',
    '09': 'Toscana',
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

IT_REGION_ABBR = {
    '01': 'PIE',
    '02': 'VAL',
    '03': 'LOM',
    '05': 'VEN',
    '06': 'FRI',
    '07': 'LIG',
    '08': 'EMI',
    '09': 'TOS',
    '10': 'UMB',
    '11': 'MAR',
    '12': 'LAZ',
    '13': 'ABR',
    '14': 'MOL',
    '15': 'CAM',
    '16': 'PUG',
    '17': 'BAS',
    '18': 'CAL',
    '19': 'SIC',
    '20': 'SAR',
    '21': 'PBZ',
    '22': 'PTN',
    'all': 'ITA',
}

IT_REGION_CODES = {
    v : k
    for k, v in IT_REGION_NAMES.items()
}

IT_REGION_POPULATION = {
    'all': 60_244_639,
    '01': 4_341_375,
    '02': 125_501,
    '03': 10_103_969,
    '05': 4_907_704,
    '06': 1_211_357,
    '07': 1_543_127,
    '08': 4_467_118,
    '09': 3_722_729,
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


def get_data_IT(run_date) -> pandas.DataFrame:
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
    if run_date.date() > datetime.date.today():
        raise ValueError("Run date is in the future. Nice try.")
    if run_date.date() < datetime.date.today():
        # TODO: implement downloading of historic data
        raise NotImplementedError(
            "Downloading with a run_date is not yet supported. "
            f"Today: {datetime.date.today()}, run_date: {run_date}"
        )
    data = get_regions_data(run_date)
    global_data = get_global_data(run_date)
    data = data.append(global_data, ignore_index=True)
    return data.set_index(["region", "date"])


def get_global_data(run_date) -> pandas.DataFrame:
    """
    Retrieve daily (run_date) global CSV and substract today's tests from yesterday's tests
    Italian data do not have daily number of tests done

    Parameters
    ----------
    run_date : pandas.Timestamp
        date for which the data shall be downloaded
    
    Returns
    -------
    df : pandas.DataFrame
        table with columns as required by rtlive/data.py API
    """
    today_obj = run_date
    today = today_obj.strftime('%Y%m%d')

    content = requests.get(
        IT_DATA_BASE_PATH + (IT_DATA_NATION_FILENAME % today),
    ).content
    today_data = pandas.read_csv(
        io.StringIO(content.decode("utf-8")),
        sep=",",
        parse_dates=["data"],
        usecols=["data", "nuovi_positivi", "tamponi"],
    ).rename(
        columns={
            "data": "date",
            "nuovi_positivi": "new_cases",
            "tamponi": "new_tests",
        }
    )

    yesterday_obj = today_obj - datetime.timedelta(days=1)
    yesterday = yesterday_obj.strftime('%Y%m%d')
    content = requests.get(
        IT_DATA_BASE_PATH + (IT_DATA_NATION_FILENAME % yesterday),
    ).content
    yesterday_data = pandas.read_csv(
        io.StringIO(content.decode("utf-8")),
        sep=",",
        parse_dates=["data"],
        usecols=["data", "nuovi_positivi", "tamponi"],
    ).rename(
        columns={
            "data": "date",
            "nuovi_positivi": "new_cases",
            "tamponi": "new_tests",
        }
    )

    current_tests = today_data.loc[0, 'new_tests'] - yesterday_data.loc[0, 'new_tests']
    today_data.loc[0, 'new_tests'] = current_tests
    today_data['region'] = ['all']
    return today_data


def get_regions_data(run_date) -> pandas.DataFrame:
    """
    Retrieve daily (run_date) regions CSV and substract today's tests from yesterday's tests
    Italian data do not have daily number of tests done

    Parameters
    ----------
    run_date : pandas.Timestamp
        date for which the data shall be downloaded
    
    Returns
    -------
    df : pandas.DataFrame
        table with columns as required by rtlive/data.py API
    """
    today_obj = run_date
    today = today_obj.strftime('%Y%m%d')

    content = requests.get(
        IT_DATA_BASE_PATH + (IT_DATA_REGION_FILENAME % today),
    ).content
    today_data = pandas.read_csv(
        io.StringIO(content.decode("utf-8")),
        sep=",",
        dtype={"codice_regione": str},
        parse_dates=["data"],
        usecols=["codice_regione", "data", "nuovi_positivi", "tamponi"],
    ).rename(
        columns={
            "codice_regione": "region",
            "data": "date",
            "nuovi_positivi": "new_cases",
            "tamponi": "new_tests",
        }
    )
    today_data.set_index(["region", "date"]).sort_index()

    yesterday_obj = today_obj - datetime.timedelta(days=1)
    yesterday = yesterday_obj.strftime('%Y%m%d')
    content = requests.get(
        IT_DATA_BASE_PATH + (IT_DATA_REGION_FILENAME % yesterday),
    ).content
    yesterday_data = pandas.read_csv(
        io.StringIO(content.decode("utf-8")),
        sep=",",
        dtype={"codice_regione": str},
        parse_dates=["data"],
        usecols=["codice_regione", "data", "nuovi_positivi", "tamponi"],
    ).rename(
        columns={
            "codice_regione": "region",
            "data": "date",
            "nuovi_positivi": "new_cases",
            "tamponi": "new_tests",
        }
    )
    yesterday_data.set_index(["region", "date"]).sort_index()

    for row in yesterday_data.itertuples():
        current = today_data[today_data['region'].isin([row.region])]
        current_tests = current['new_tests'] - row.new_tests
        today_data.loc[current.index, 'new_tests'] = current_tests
    
    return today_data
    

def forecast_IT(df: pandas.DataFrame):
    """ Applies testcount interpolation/extrapolation to italian data.

    Currently this assumes the OWID data, which only has an "all" region.
    In the future, this should be replaced with more fine graned data loading!
    """
    # forecast with existing data
    df['predicted_new_tests'], results = preprocessing.predict_testcounts_all_regions(df, 'IT')
    return df, results


from .. import data
data.set_country_support(
    country_alpha2="IT",
    compute_zone=data.Zone.Europe,
    region_name=IT_REGION_NAMES,
    region_short_name=IT_REGION_ABBR,
    region_population=IT_REGION_POPULATION,
    fn_load=get_data_IT,
    fn_process=forecast_IT,
)
