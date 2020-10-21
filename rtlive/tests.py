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
            df["new_tests"] = 100 + numpy.arange(len(df)) * numpy.random.randint(10, 100, size=len(df))
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
            country_alpha2="CH",
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
        assert "CH" in data.SUPPORTED_COUNTRIES
        df = data.get_data("CH", datetime.datetime.today())
        df, forecasts = data.process_testcounts("CH", df)
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


class TestDataGeneralized:
    def test_get_unsupported(self):
        with pytest.raises(KeyError):
            covid.data.get_data(country="not_a_country", run_date=pandas.Timestamp("2020-06-20"))

    def test_get_us(self):
        import covid.data_us
        assert "us" in covid.data.LOADERS
        run_date = pandas.Timestamp('2020-06-25')
        result = covid.data.get_data("us", run_date)
        assert isinstance(result, pandas.DataFrame)
        assert result.index.names == ("region", "date")
        assert result.xs('NY').index[-1] < run_date
        assert result.xs('NY').index[-1] == (run_date - pandas.DateOffset(1))
        assert "positive" in result.columns
        assert "total" in result.columns


class TestGenerative:
    def test_build(self):
        df_raw = covid.data.get_raw_covidtracking_data()
        df_processed = covid.data.process_covidtracking_data(df_raw, pandas.Timestamp('2020-06-25'))
        model = covid.models.generative.GenerativeModel(
            region='NY',
            observed=df_processed.xs('NY')
        )
        pmodel = model.build()
        assert isinstance(pmodel, pymc3.Model)
        # important coordinates
        assert "date" in pmodel.coords
        assert "nonzero_date" in pmodel.coords
        # important random variables
        expected_vars = set(['r_t', 'seed', 'infections', 'test_adjusted_positive', 'exposure', 'positive', 'alpha'])
        missing_vars = expected_vars.difference(set(pmodel.named_vars.keys()))
        assert not missing_vars, f'Missing variables: {missing_vars}'

    def test_sample_and_idata(self):
        df_raw = covid.data.get_raw_covidtracking_data()
        df_processed = covid.data.process_covidtracking_data(df_raw, pandas.Timestamp('2020-06-25'))
        model = covid.models.generative.GenerativeModel(
            region='NY',
            observed=df_processed.xs('NY')
        )
        model.build()
        model.sample(
            cores=1, chains=2, tune=5, draws=7
        )
        assert model.trace is not None
        idata = model.inference_data
        assert isinstance(idata, arviz.InferenceData)
        # check posterior
        assert idata.posterior.attrs["model_version"] == model.version
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
