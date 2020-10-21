import io
import logging
import requests

import numpy
import pandas

from typing import Dict, Tuple, Union

from .. import data, preprocessing

_log = logging.getLogger(__file__)


def get_regions_metadata() -> [
    Tuple[Dict[str, str], Dict[str, float]]

]:
    """
    Link to regions' population: https://www.insee.fr/fr/statistiques/1893198
    Link to regions' codes: https://www.insee.fr/fr/information/2114819#titre-bloc-29

    Returns
    -------
    Tuple of dictionaries: one mapping regions' codes to regions' names, the other mapping regions'
    codes to regions' population.
    """
    REGIONS_RAW_DATA = {
        "Name": [
            "Auvergne-Rhône-Alpes",
            "Bourgogne-Franche-Comté",
            "Bretagne",
            "Centre-Val-de-Loire",
            "Corse",
            "Grand Est",
            "Hauts-de-France",
            "Île-de-France",
            "Normandie",
            "Nouvelle-Aquitaine",
            "Occitanie",
            "Pays de la Loire",
            "Provence-Alpes-Côte d'Azur",
            "Guadeloupe",
            "Martinique",
            "Guyane",
            "La Réunion",
            "Mayotte",
        ],
        "Code": [
            "84",
            "27",
            "53",
            "24",
            "94",
            "44",
            "32",
            "11",
            "28",
            "75",
            "76",
            "52",
            "93",
            "01",
            "02",
            "03",
            "04",
            "06",
        ],
        "Total": [
            "8 032 377",
            "2 783 039",
            "3 340 379",
            "2 559 073",
            "344 679",
            "5 511 747",
            "5 962 662",
            "12 278 210",
            "3 303 500",
            "5 999 982",
            "5 924 858",
            "3 801 797",
            "5 055 651",
            "376 879",
            "358 749",
            "290 691",
            "859 959",
            "279 471",
        ],
    }

    reg_pop_codes = pandas.DataFrame.from_dict(REGIONS_RAW_DATA).set_index("Code")

    reg_pop_codes["Population"] = reg_pop_codes.Total.str.replace(" ", "_").astype(
        float
    )
    reg_pop_codes = reg_pop_codes.drop("Total", axis=1)

    reg_pop_codes.loc["all", "Population"] = reg_pop_codes.Population.sum()
    reg_pop_codes.loc["all", "Name"] = "France"

    return reg_pop_codes.to_dict()["Name"], reg_pop_codes.to_dict()["Population"]


def get_data_FR(run_date: pandas.Timestamp) -> pandas.DataFrame:
    """
    Retrieve daily CSV from https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux
    # -resultats-des-tests-virologiques-covid-19 for all French regions.
    Limitations:
    * Data by regions only contain tests for which residence regions of tested
    people could be known. Hence, countrywide data contain more tests than sum of all
    regions.
    * Data transmission can sometimes excess 9 days. Indicators are updated daily on test
    results reception.

    Parameters
    ----------
    run_date : pandas.Timestamp
        use the data as it was released on that day

    Returns
    -------
    result : pandas.DataFrame
        [region, date]-indexed table that has rows for every region & date combination in [
        2020-05-13, run_date - 1].
        Contains columns "new_cases" and "new_tests" that are the number of NEW positives /
        total tests for each (day-region) couple.
        "all" region is the sum over all regions.
    """
    content = requests.get(
        "https://www.data.gouv.fr/fr/datasets/r/001aca18-df6a-45c8-89e6-f82d689e6c01",
        verify=False,
    ).content
    data = pandas.read_csv(
        io.StringIO(content.decode("utf-8")),
        sep=";",
        dtype={"reg": str},
        parse_dates=["jour"],
        usecols=["reg", "jour", "P", "T", "cl_age90"],
    ).rename(
        columns={
            "reg": "region",
            "jour": "date",
            "cl_age90": "ageclass",
            "P": "new_cases",
            "T": "new_tests",
        }
    )
    # Drop data by age class ('0' age class is the sum of all age classes) and truncate data after
    # run_date
    data = (
        data[data.ageclass == 0]
        .drop("ageclass", axis=1)
        .set_index("date")
        .sort_index()
        .truncate(after=run_date - pandas.DateOffset(1))
        .reset_index()
        .set_index(["region", "date"])
        .sort_index()
    )
    # compute and append national data, and restrict to existing regions to get rid of data
    # errors that creep in from the original link
    df_all = data.reset_index(level=1).groupby("date").sum().reset_index()
    df_all["region"] = "all"
    true_region_codes = get_regions_metadata()["Name"].keys()
    data = (
        data.append(df_all.set_index(["region", "date"]))
        .loc[true_region_codes]
        .sort_index()
    )

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


def forecast_FR(df: pandas.DataFrame) -> Tuple[pandas.DataFrame, dict]:
    """
    Applies test count interpolation/extrapolation to French data.

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
        df, "FR"
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


regions_names, regions_population = get_regions_metadata()
data.set_country_support(
    country_alpha2="FR",
    compute_zone=data.Zone.Europe,
    region_name=regions_names,
    region_population=regions_population,
    fn_load=get_data_FR,
    fn_process=forecast_FR,
)
