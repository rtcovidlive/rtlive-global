import logging
import numpy
import os
import pandas
import pathlib
import requests
import scipy.stats
import tarfile


_log = logging.getLogger(__file__)
_DP_DATA = pathlib.Path(pathlib.Path(__file__).parent.parent, "data")
if not _DP_DATA.exists():
    _log.warning("Data directory at %s does not exist yet. Creating...")
    _DP_DATA.mkdir()

__all__ = [
    "delay_distribution",
]


def _download_patient_data(file_path=None):
    """ Downloads patient data to data directory
        from: https://stackoverflow.com/questions/16694907/ """
    if not file_path:
        file_path = os.path.join(os.path.dirname(__file__), "../data/patients.tar.gz")
    url = "https://github.com/beoutbreakprepared/nCoV2019/raw/master/latest_data/latestdata.tar.gz"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)


def _read_patient_data(file_path=None, max_delay=60) -> pandas.DataFrame:
    """ Finds every valid delay between symptom onset and report confirmation
        from the patient line list and returns all the delay samples. """
    if not file_path:
        file_path = os.path.join(os.path.dirname(__file__), "../data/patients.tar.gz")
    patients = pd.read_csv(
        file_path,
        parse_dates=False,
        usecols=["country", "date_onset_symptoms", "date_confirmation"],
        low_memory=False,
    )

    patients.columns = ["Country", "Onset", "Confirmed"]
    patients.Country = patients.Country.astype("category")

    # There's an errant reversed date
    patients = patients.replace("01.31.2020", "31.01.2020")
    patients = patients.replace("31.04.2020", "01.05.2020")

    # Only keep if both values are present
    patients = patients.dropna()

    # Must have strings that look like individual dates
    # "2020.03.09" is 10 chars long
    is_ten_char = lambda x: x.str.len().eq(10)
    patients = patients[is_ten_char(patients.Confirmed) & is_ten_char(patients.Onset)]

    # Convert both to datetimes
    patients.Confirmed = pd.to_datetime(
        patients.Confirmed, format="%d.%m.%Y", errors="coerce"
    )
    patients.Onset = pd.to_datetime(patients.Onset, format="%d.%m.%Y", errors="coerce")

    # Only keep records where confirmed > onset
    patients = patients[patients.Confirmed > patients.Onset]

    # Mexico has many cases that are all confirmed on the same day regardless
    # of onset date, so we filter it out.
    patients = patients[patients.Country.ne("Mexico")]

    # Remove any onset dates from the last two weeks to account for all the
    # people who haven't been confirmed yet.
    patients = patients[patients.Onset < patients.Onset.max() - pd.Timedelta(days=14)]

    return patients


def _extract_test_delays_from_patient_data(file_path=None, max_delay=60):
    patients = _read_patient_data(file_path=file_path, max_delay=max_delay)
    delays = (patients.Confirmed - patients.Onset).dt.days
    delays = delays.reset_index(drop=True)
    delays = delays[delays.le(max_delay)]
    return delays


def delay_distribution(incubation_days=5) -> numpy.ndarray:
    """ Returns the empirical delay distribution between symptom onset and confirmed positive case. """

    # The literature suggests roughly 5 days of incubation before becoming
    # having symptoms. See:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7081172/
    p_delay_path = pathlib.Path(_DP_DATA, "p_delay.csv")
    if p_delay_path.exists():
        _log.info("Loading precomputed p_delay distribution from %s", p_delay_path)
        p_delay = pandas.read_csv(p_delay_path, squeeze=True)
    else:
        _log.info("Precomputing testing delay distribution from patient data")
        delays = _extract_test_delays_from_patient_data()
        p_delay = delays.value_counts().sort_index()
        new_range = numpy.arange(0, p_delay.index.max() + 1)
        p_delay = p_delay.reindex(new_range, fill_value=0)
        p_delay /= p_delay.sum()
        p_delay = (
            pandas.Series(numpy.zeros(incubation_days))
            .append(p_delay, ignore_index=True)
            .rename("p_delay")
        )
        p_delay.to_csv(pathlib.Path(_DP_DATA, "p_delay.csv"), index=False)

    return p_delay.values
