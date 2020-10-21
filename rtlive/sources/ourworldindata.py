import datetime
import iso3166
import pandas
import typing


def download_owid(run_date: pandas.Timestamp = None) -> pandas.DataFrame:
    """ Downloads the OurWorldInData COVID-19 dataset.

    Parameters
    ----------
    run_date : optional, pandas.Timestamp
        the date for which to download the data
        THIS IS CURRENTLY NOT IMPLEMENTED

    Raises
    ------
    NotImplementedError
        when a run_date earlier than today is passed
    """
    if run_date.date() > datetime.date.today():
        raise ValueError("Run date is in the future. Nice try.")
    if run_date.date() < datetime.date.today():
        # TODO: implement downloading of historic data
        raise NotImplementedError(
            "Downloading with a run_date is not yet supported. "
            f"Today: {datetime.date.today()}, run_date: {run_date}"
        )

    df_raw = pandas.read_csv(
        "https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-data.csv?raw=true",
        parse_dates=["date"],
    ).rename(columns={"iso_code": "iso_alpha3"})
    df_raw["iso_alpha2"] = [
        iso3166.countries.get(alpha3).alpha2 if alpha3 in iso3166.countries else None
        for alpha3 in df_raw.iso_alpha3
    ]
    df_raw["region"] = "all"
    return df_raw.set_index(["iso_alpha2", "region", "date"])


def create_loader_function(
    country_alpha2: str
) -> typing.Callable[[pandas.Timestamp], pandas.DataFrame]:
    """ Creates a data loader functions for a country in the OurWorldInData dataset.

    Parameters
    ----------
    country_alpha2 : str
        ISO-3166 alpha 2 country code

    Returns
    -------
    loader_function : callable
        the data loader function for the specified country
    """
    def loader_fun(run_date: pandas.Timestamp):
        df = download_owid(run_date)[
            [
                "total_cases",
                "new_cases",
                "total_deaths",
                "new_deaths",
                "total_tests",
                "new_tests",
                "tests_units",
            ]
        ].xs(country_alpha2)
        # TODO: inter- and extrapolation of gaps in total_tests and new_tests columns
        return df
    return loader_fun
