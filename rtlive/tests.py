import datetime
import numpy
import pandas
import pytest

import arviz
import pymc3

from . import assumptions
from . import data
from . import model


class TestData:
    def test_mock_country(self):
        def test_load(run_date):
            df = pandas.DataFrame(
                index=pandas.date_range("2020-03-01", "2020-10-01", freq="D", name="date"),
                columns=["region", "new_cases", "new_tests"]
            )
            df["region"] = "all"
            df["new_tests"] = 500 + numpy.arange(len(df)) * numpy.random.randint(10, 100, size=len(df))
            numpy.random.seed(123)
            df["new_cases"] = numpy.random.randint(df["new_tests"]/200, df["new_tests"]/100)
            df.at["2020-07-01":"2020-07-15", "new_tests"] = numpy.nan
            return df.reset_index().set_index(["region", "date"])

        def test_process(df):
            results = {}
            for region in df.reset_index().region.unique():
                interpolation = df.xs(region).new_tests.interpolate("linear")
                df.loc[pandas.IndexSlice[region, :], "predicted_new_tests"] = interpolation.values
                results[region] = (
                    interpolation,
                    None, # Prophet object
                    None, # Prophet result dataframe
                    None, # named holidays
                )
            return df, results

        data.set_country_support(
            country_alpha2="DE",
            compute_zone=data.Zone.Europe,
            region_name={
                "all": "Test country",
            },
            region_population={
                "all": 1_234_567,
            },
            fn_load=test_load,
            fn_process=test_process,
        )
        assert "DE" in data.SUPPORTED_COUNTRIES
        df = data.get_data("DE", datetime.datetime.today())
        df, forecasts = data.process_testcounts("DE", df)
        assert isinstance(df, pandas.DataFrame)
        assert isinstance(forecasts, dict)

    def test_unsupported_country(self):
        with pytest.raises(KeyError, match="not in the collection"):
            data.get_data("XY", datetime.datetime.today())
        with pytest.raises(KeyError, match="Unknown ISO-3166 alpha 2"):
            data.set_country_support(
                country_alpha2="XY",
                compute_zone=data.Zone.Europe,
                region_name={
                    "all": "Test country",
                },
                region_population={
                    "all": 1_234_567,
                },
                fn_load=None,
                fn_process=None,
            )


class TestModel:
    def test_build(self):
        from rtlive.sources import data_ch

        country_alpha2 = 'CH'
        df_raw = data.get_data(
            country_alpha2, datetime.datetime.today()
        )
        df_processed, _ = data.process_testcounts(
            country=country_alpha2,
            df_raw=df_raw,
        )
        pmodel = model.build_model(
            observed=df_processed.xs("all"), 
            p_generation_time=assumptions.generation_time(),
            test_col="predicted_new_tests",
            p_delay=assumptions.delay_distribution(),
            buffer_days=20
        )
        assert isinstance(pmodel, pymc3.Model)
        # important coordinates
        assert "date" in pmodel.coords
        assert "nonzero_date" in pmodel.coords
        # important random variables
        expected_vars = set(['r_t', 'seed', 'infections', 'test_adjusted_positive', 'exposure', 'positive', 'alpha'])
        missing_vars = expected_vars.difference(set(pmodel.named_vars.keys()))
        assert not missing_vars, f'Missing variables: {missing_vars}'

    def test_sample_and_idata(self):
        from rtlive.sources import data_ch

        country_alpha2 = 'CH'
        df_raw = data.get_data(
            country_alpha2, datetime.datetime.today()
        )
        df_processed, _ = data.process_testcounts(
            country=country_alpha2,
            df_raw=df_raw,
        )
        pmodel = model.build_model(
            observed=df_processed.xs("all"), 
            p_generation_time=assumptions.generation_time(),
            test_col="predicted_new_tests",
            p_delay=assumptions.delay_distribution(),
            buffer_days=20
        )
        idata = model.sample(
            pmodel, cores=1, chains=2, tune=5, draws=7
        )
        assert isinstance(idata, arviz.InferenceData)
        # check posterior
        assert idata.posterior.attrs["model_version"] == model.__version__
        assert "chain" in idata.posterior.coords
        assert "draw" in idata.posterior.coords
        assert "date" in idata.posterior.coords
        expected_vars = set(["r_t", "seed", "infections", "test_adjusted_positive", "exposure", "positive", "alpha"])
        missing_vars = expected_vars.difference(set(idata.posterior.keys()))
        assert not missing_vars, f'Missing {missing_vars} from posterior group'
        # check observed_data
        assert "nonzero_date" in idata.observed_data.coords
        expected_vars = set(["nonzero_positive"])
        missing_vars = expected_vars.difference(set(idata.observed_data.keys()))
        assert not missing_vars, f'Missing {missing_vars} from observed_data group'
        # check constant_data
        assert "date" in idata.constant_data.coords
        assert "nonzero_date" in idata.constant_data.coords
        expected_vars = set(["exposure", "tests", "observed_positive", "nonzero_observed_positive"])
        missing_vars = expected_vars.difference(set(idata.constant_data.keys()))
        assert not missing_vars, f'Missing {missing_vars} from constant_data group'
