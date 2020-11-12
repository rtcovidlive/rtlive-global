import logging
import pandas

from .. import preprocessing

_log = logging.getLogger(__file__)

# From https://nl.wikipedia.org/wiki/Provincies_van_Belgi%C3%AB
BE_REGION_NAMES = {
    'all': 'Belgium',
#     '01': 'Vlaanderen',
#     '02': 'Wallonie',
#     '03': 'Brussel',
#     '05': 'Antwerpen',
#     '06': 'Limburg',
#     '07': 'Oost-Vlaanderen',
#     '08': 'Vlaams-Brabant',
#     '09': 'West-Vlaanderen',
#     '10': 'Henegouwen',
#     '11': 'Luik',
#     '12': 'Luxemburg',
#     '13': 'Namen',
#     '14': 'Waals-Brabant',
}

# https://www.ibz.rrn.fgov.be/fileadmin/user_upload/fr/pop/statistiques/population-bevolking-20200101.pdf
BE_REGION_POPULATION = {
    'all': 11_476_279,
#     '01': 6_623_505,
#     '02': 3_641_748,
#     '03': 1_211_026,
#     '05': 1_867_366,
#     '06': 876_785,
#     '07': 1_524_077,
#     '08': 1_155_148,
#     '09': 1_200_129,
#     '10': 1_345_270,
#     '11': 1_108_481,
#     '12': 286_571,
#     '13': 495_474,
#     '14': 302_265,
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
    # Reformat data into Rtlive format
    df_tests_per_day = (df_tests
       .assign(region='all')
       .groupby('DATE', as_index=False)
       .agg(positive=('TESTS_ALL_POS', 'sum'), total=('TESTS_ALL', 'sum'), region=('region', 'first'))
       .rename(columns={'DATE':'date'})
       .set_index(["region", "date"])
       .sort_index()
    )
    
    return df_tests_per_day


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
    region_population=BE_REGION_POPULATION,
    fn_load=get_data_BE,
    fn_process=forecast_BE,
)
