"""
This module contains all US-specific data loading and data cleaning routines.
"""
import datetime
import requests
import pandas as pd
import numpy as np

from .. import data

idx = pd.IndexSlice


def get_raw_covidtracking_data(run_date: pd.Timestamp):
    """ Gets the current daily CSV from COVIDTracking """
    if run_date.date() > datetime.date.today():
        raise ValueError("Run date is in the future. Nice try.")
    if run_date.date() < datetime.date.today():
        # TODO: implement downloading of historic data
        raise NotImplementedError(
            "Downloading with a run_date is not yet supported. "
            f"Today: {datetime.date.today()}, run_date: {run_date}"
        )

    url = "https://covidtracking.com/api/v1/states/daily.csv"
    data = pd.read_csv(url).rename(columns={
        "state": "region",
    })
    data["date"] = pd.to_datetime(data["date"], format="%Y%m%d")
    data = data.set_index(["region", "date"]).sort_index()

    # Too little data or unreliable reporting in the data source.
    df_raw = data.drop(["MP", "GU", "AS", "PR", "VI"])

    # the data in these columns is crap. But it will be corrected by the process_covidtracking_data function
    # here we just add the columns so the original data is kept
    for region in df_raw.reset_index().region.unique():
        df_raw.loc[idx[region, :], "new_cases"] = df_raw.xs(region).positive.diff().values
        df_raw.loc[idx[region, :], "new_tests"] = df_raw.xs(region).total.diff().values

    # calculate the sum over all states
    df_all = df_raw.sum(level='date')
    df_all.insert(0, column='region', value='all')
    df_all = df_all.reset_index().set_index(['region', 'date'])
    df_merged = pd.concat([df_raw, df_all]).sort_index()
    return df_merged


def apply_corrections(data: pd.DataFrame) -> pd.DataFrame:

    # On Jun 5 Covidtracking started counting probable cases too
    # which increases the amount by 5014.
    # https://covidtracking.com/screenshots/MI/MI-20200605-184320.png
    data.loc[idx["MI", pd.Timestamp("2020-06-05") :], "total"] -= 5014

    # From CT: On June 19th, LDH removed 1666 duplicate and non resident cases
    # after implementing a new de-duplicaton process.
    data.loc[idx["LA", pd.Timestamp("2020-06-19") :], ["positive", "total"]] += 1666

    # calculate the daily counts
    for region in data.reset_index().region.unique():
        data.loc[idx[region, :], "new_cases"] = data.xs(region).positive.diff().values
        data.loc[idx[region, :], "new_tests"] = data.xs(region).total.diff().values
    data["new_cases"][data["new_cases"] < 0] = np.nan
    data["new_tests"][data["new_tests"] < 0] = np.nan

    # Michigan missed 6/18 totals and lumped them into 6/19 so we've
    # divided the totals in two and equally distributed to both days.
    data.loc[idx["MI", pd.Timestamp("2020-06-18")], "new_tests"] = 14871
    data.loc[idx["MI", pd.Timestamp("2020-06-19")], "new_tests"] = 14871

    # Note that when we set new_cases/new_tests to NaN, the model ignores that date. See
    # the likelihood function in GenerativeModel.build

    # Huge outlier in NJ causing sampling issues.
    data.loc[idx["NJ", pd.Timestamp("2020-05-11")], ["new_cases", "new_tests"]] = np.nan
    # Same tests and positives, nulling out
    data.loc[idx["NJ", pd.Timestamp("2020-07-25")], ["new_cases", "new_tests"]] = np.nan

    # Huge outlier in CA causing sampling issues.
    data.loc[idx["CA", pd.Timestamp("2020-04-22")], ["new_cases", "new_tests"]] = np.nan

    # Huge outlier in CA causing sampling issues.
    # TODO: generally should handle when # tests == # positives and that
    # is not an indication of positive rate.
    data.loc[idx["SC", pd.Timestamp("2020-06-26")], ["new_cases", "new_tests"]] = np.nan

    # Two days of no new data then lumped sum on third day with lack of new total tests
    data.loc[idx["OR", pd.Timestamp("2020-06-26") : pd.Timestamp("2020-06-28")], 'new_cases'] = 174
    data.loc[idx["OR", pd.Timestamp("2020-06-26") : pd.Timestamp("2020-06-28")], 'new_tests'] = 3296

    #https://twitter.com/OHdeptofhealth/status/1278768987292209154
    data.loc[idx["OH", pd.Timestamp("2020-07-01")], ["new_cases", "new_tests"]] = np.nan
    data.loc[idx["OH", pd.Timestamp("2020-07-09")], ["new_cases", "new_tests"]] = np.nan

    # Nevada didn't report total tests this day
    data.loc[idx["NV", pd.Timestamp("2020-07-02")], ["new_cases", "new_tests"]] = np.nan

    # A bunch of incorrect values for WA data so nulling them out.
    data.loc[idx["WA", pd.Timestamp("2020-06-05") : pd.Timestamp("2020-06-07")], ["new_cases", "new_tests"]] = np.nan
    data.loc[idx["WA", pd.Timestamp("2020-06-20") : pd.Timestamp("2020-06-21")], ["new_cases", "new_tests"]] = np.nan

    # AL reported tests == positives
    data.loc[idx["AL", pd.Timestamp("2020-07-09")], ["new_cases", "new_tests"]] = np.nan

    # Low reported tests
    data.loc[idx["AR", pd.Timestamp("2020-07-10")], ["new_cases", "new_tests"]] = np.nan

    # Positives == tests
    data.loc[idx["MS", pd.Timestamp("2020-07-12")], ["new_cases", "new_tests"]] = np.nan

    # Positive == Tests; lumpy reporting for CT
    data.loc[idx["CT", pd.Timestamp("2020-07-17")], ["new_cases", "new_tests"]] = np.nan
    data.loc[idx["CT", pd.Timestamp("2020-07-21")], ["new_cases", "new_tests"]] = np.nan

    data.loc[idx["DC", pd.Timestamp("2020-08-04")], ["new_cases", "new_tests"]] = np.nan

    # Outlier dates in PA
    data.loc[
        idx[
            "PA",
            [
                pd.Timestamp("2020-06-03"),
                pd.Timestamp("2020-04-21"),
                pd.Timestamp("2020-05-20"),
            ],
        ],
        ["new_cases", "new_tests"],
    ] = np.nan

    data.loc[idx["HI", pd.Timestamp("2020-08-07")], ["new_cases", "new_tests"]] = np.nan
    data.loc[idx["TX", pd.Timestamp("2020-08-08")], ["new_cases", "new_tests"]] = np.nan
    data.loc[idx["TX", pd.Timestamp("2020-08-11")], ["new_cases", "new_tests"]] = np.nan

    data.loc[idx["DE", pd.Timestamp("2020-08-14")], ["new_cases", "new_tests"]] = np.nan

    data.loc[idx["SD", pd.Timestamp("2020-08-26")], ["new_cases", "new_tests"]] = np.nan

    data.loc[idx["WA", pd.Timestamp("2020-09-22"):pd.Timestamp("2020-09-24")], ["new_cases", "new_tests"]] = np.nan

    # Zero out any rows where positive tests equal or exceed total reported tests
    # Do not act on Wyoming as they report positive==total most days
    filtering_date = pd.Timestamp('2020-07-27')
    zero_filter = (data.positive >= data.total) & \
        (data.index.get_level_values('date') >= filtering_date) & \
        (~data.index.get_level_values('region').isin(['WY']))
    data.loc[zero_filter, ["new_cases", "new_tests"]] = np.nan
    return data


def process_covidtracking_data(df_raw: pd.DataFrame):
    """ Processes raw COVIDTracking data to be in a form for the GenerativeModel.
        In many cases, we need to correct data errors or obvious outliers."""
    df_corrected = apply_corrections(df_raw.copy())
    df_corrected["predicted_new_tests"] = df_corrected["new_tests"]
    df_corrected["new_tests"] = df_raw["new_tests"]
    # actual forecasting currently not implemented...
    forecasting_results = {
    }
    # calculate the sum over all states ... again
    df_all = df_corrected.sum(level='date', min_count=40)
    df_all.insert(0, column='region', value='all')
    df_all = df_all.reset_index().set_index(['region', 'date'])
    df_merged = pd.concat([df_corrected.drop(["all"]), df_all]).sort_index()
    return df_merged, forecasting_results


US_NAME_POPULATION = {
    'AK': ('Alaska', 731545),
    'AL': ('Alabama', 4903185),
    'AR': ('Arkansas', 3017825),
    #'AS': ('American Samoa', 55641),
    'AZ': ('Arizona', 7278717),
    'CA': ('California', 39512223),
    'CO': ('Colorado', 5758736),
    'CT': ('Connecticut', 3565287),
    'DC': ('District of Columbia', 705749),
    'DE': ('Delaware', 973764),
    'FL': ('Florida', 21477737),
    'GA': ('Georgia', 10617423),
    #'GU': ('Guam', 165718),
    'HI': ('Hawaii', 1415872),
    'IA': ('Iowa', 3155070),
    'ID': ('Idaho', 1787147),
    'IL': ('Illinois', 12671821),
    'IN': ('Indiana', 6732219),
    'KS': ('Kansas', 2913314),
    'KY': ('Kentucky', 4467673),
    'LA': ('Louisiana', 4648794),
    'MA': ('Massachusetts', 6949503),
    'MD': ('Maryland', 6045680),
    'ME': ('Maine', 1344212),
    'MI': ('Michigan', 9986857),
    'MN': ('Minnesota', 5639632),
    'MO': ('Missouri', 6137428),
    'MS': ('Mississippi', 2976149),
    'MT': ('Montana', 1068778),
    'NC': ('North Carolina', 10488084),
    'ND': ('North Dakota', 762062),
    'NE': ('Nebraska', 1934408),
    'NH': ('New Hampshire', 1359711),
    #'NI': ('Northern Mariana Islands', 55194),
    'NJ': ('New Jersey', 8882190),
    'NM': ('New Mexico', 2096829),
    'NV': ('Nevada', 3080156),
    'NY': ('New York', 19453561),
    'OH': ('Ohio', 11689100),
    'OK': ('Oklahoma', 3956971),
    'OR': ('Oregon', 4217737),
    'PA': ('Pennsylvania', 12801989),
    #'PR': ('Puerto Rico', 3193694),
    'RI': ('Rhode Island', 1059361),
    'SC': ('South Carolina', 5148714),
    'SD': ('South Dakota', 884659),
    'TN': ('Tennessee', 6833174),
    'TX': ('Texas', 28995881),
    'UT': ('Utah', 3205958),
    'VA': ('Virginia', 8535519),
    #'VI': ('U.S. Virgin Islands', 104914),
    'VT': ('Vermont', 623989),
    'WA': ('Washington', 7614893),
    'WI': ('Wisconsin', 5822434),
    'WV': ('West Virginia', 1792065),
    'WY': ('Wyoming', 578759),
}
US_REGION_NAMES = {
    code : name
    for code, (name, _) in US_NAME_POPULATION.items()
}
US_REGION_POPULATION = {
    code : pop
    for code, (_, pop) in US_NAME_POPULATION.items()
}
US_REGION_NAMES["all"] = "United States of America"
US_REGION_POPULATION["all"] = sum(US_REGION_POPULATION.values())


data.set_country_support(
    country_alpha2="US",
    compute_zone=data.Zone.America,
    region_name=US_REGION_NAMES,
    region_population=US_REGION_POPULATION,
    fn_load=get_raw_covidtracking_data,
    fn_process=process_covidtracking_data,
)