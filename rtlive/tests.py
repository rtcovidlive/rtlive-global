import datetime
from matplotlib import pyplot
import numpy
import os
import pandas
import pathlib
import pytest
import xarray

import arviz
import pymc3

from . import assumptions
from . import data
from . import export
from . import model
from . import plotting


IDATA_FILENAMES = [
    "DE_2020-10-26_all_v1.0.2.nc",
    "DE_2020-10-26_all_v1.1.0.nc",
]


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

        # mock with Antarctica
        # (Soviet Union or German Democratic Republic are not supported by ISO-3166 package)
        data.set_country_support(
            country_alpha2="AQ",
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
        assert "AQ" in data.SUPPORTED_COUNTRIES
        df = data.get_data("AQ", datetime.datetime.today())
        df, forecasts = data.process_testcounts("AQ", df)
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
        assert "date_with_cases" in pmodel.coords
        assert "date_with_testcounts" in pmodel.coords
        assert "date_with_data" in pmodel.coords
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
        assert "date_with_data" in idata.observed_data.coords
        expected_vars = set(["likelihood"])
        missing_vars = expected_vars.difference(set(idata.observed_data.keys()))
        assert not missing_vars, f'Missing {missing_vars} from observed_data group'
        # check constant_data
        assert "date_with_cases" in idata.constant_data.coords
        assert "date_with_testcounts" in idata.constant_data.coords
        expected_vars = set(["exposure", "tests", "observed_positive", "observed_positive_where_data"])
        missing_vars = expected_vars.difference(set(idata.constant_data.keys()))
        assert not missing_vars, f'Missing {missing_vars} from constant_data group'

    @pytest.mark.parametrize("filename", IDATA_FILENAMES)
    def test_get_case_curves(self, filename):
        fp = pathlib.Path(pathlib.Path(__file__).parent, "testdata", filename)
        idata = arviz.from_netcdf(str(fp))
        assert isinstance(idata, arviz.InferenceData)
        case_curves = model.get_case_curves(idata)
        assert isinstance(case_curves, tuple)
        for obj in case_curves:
            assert isinstance(obj, xarray.DataArray)
            assert obj.coords.dims == ("date", "sample")

    @pytest.mark.parametrize("filename", IDATA_FILENAMES)
    def test_get_scale_factor(self, filename):
        fp = pathlib.Path(pathlib.Path(__file__).parent, "testdata", filename)
        idata = arviz.from_netcdf(str(fp))
        assert isinstance(idata, arviz.InferenceData)
        scale_factor = model.get_scale_factor(idata)
        assert isinstance(scale_factor, xarray.DataArray)
        assert scale_factor.coords.dims == ("sample",)
        assert 1000 < scale_factor.mean() < 200_000


class TestSources:
    @pytest.mark.parametrize("fp_submodule", pathlib.Path(os.path.join(os.path.dirname(__file__), "sources")).glob("*.py"))
    def test_imports(self, fp_submodule):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            f'rtlive.sources.{fp_submodule.stem}',
            str(fp_submodule)
        )
        imported_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imported_module)


class TestPlotting:
    @pytest.mark.parametrize("filename", IDATA_FILENAMES)
    @pytest.mark.parametrize("plot_positive", [None, "all", "unobserved"])
    def test_plot_details(self, filename, plot_positive):
        fp = pathlib.Path(pathlib.Path(__file__).parent, "testdata", filename)
        idata = arviz.from_netcdf(str(fp))
        assert isinstance(idata, arviz.InferenceData)
        fig, axs = plotting.plot_details(
            idata,
            plot_positive=plot_positive
        )
        pyplot.close()

    @pytest.mark.parametrize("filename", IDATA_FILENAMES)
    def test_plot_thumbnail(self, filename):
        fp = pathlib.Path(pathlib.Path(__file__).parent, "testdata", filename)
        idata = arviz.from_netcdf(str(fp))
        assert isinstance(idata, arviz.InferenceData)
        fig, axs = plotting.plot_thumbnail(idata)
        pyplot.close()


class TestExport:
    def test_summarize_median_and_hdi(self):
        numpy.random.seed(345234)
        samples = xarray.DataArray(data=numpy.random.uniform(low=0.8, high=1.8, size=1000))

        result = export.summarize_median_and_hdi(samples, prefix="blub")
        numpy.testing.assert_allclose(result["blub"], numpy.median(samples))
        numpy.testing.assert_allclose(result["blub_lower"], 0.85, atol=0.02)
        numpy.testing.assert_allclose(result["blub_upper"], 1.75, atol=0.02)

    def test_summarize_r_t(self):
        numpy.random.seed(1234)
        samples = xarray.DataArray(data=numpy.random.uniform(low=0.8, high=1.8, size=1000))
        result = export.summarize_r_t(samples)
        numpy.testing.assert_allclose(result["r_t_threshold_probability"], 0.8, atol=0.02)
        numpy.testing.assert_allclose(result["r_t"], numpy.median(samples))
        numpy.testing.assert_allclose(result["r_t_lower"], 0.83, atol=0.02)
        numpy.testing.assert_allclose(result["r_t_upper"], 1.73, atol=0.02)

    def test_summarize_infections(self):
        numpy.random.seed(356435)
        samples = xarray.DataArray(data=numpy.random.normal(200, 30, size=300_000))
        result = export.summarize_infections(samples, population=200_000, hdi_prob=0.9545)
        result
        numpy.testing.assert_allclose(result["infections_by_100k"], 100, rtol=0.01)
        numpy.testing.assert_allclose(result["infections_by_100k_lower"], 70, rtol=0.01)
        numpy.testing.assert_allclose(result["infections_by_100k_upper"], 130, rtol=0.01)
        numpy.testing.assert_allclose(result["infections_absolute"], 200, rtol=0.01)
        numpy.testing.assert_allclose(result["infections_absolute_lower"], 140, rtol=0.01)
        numpy.testing.assert_allclose(result["infections_absolute_upper"], 260, rtol=0.01)