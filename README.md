# Country-agnostic modeling of R<sub>t</sub>
This repository contains code for data processing and modeling of COVID-19 reproduction number R<sub>t</sub>
using the model that was developed for https://rt.live.

While the implementation of the model is completely country-independent, the code has a high-level interface that allows
for easy plug-in support of new countries.

The [rt.live](https://rt.live) site itself (for the United States) runs the model hosted at [rtcovidlive/covid-model](https://github.com/rtcovidlive/covid-model), on which this further work was based. Other sites like [rtlive.de](https://rtlive.de) run the code in this repository.

## Where to find...
+ [Explanation of the model](notebooks/Tutorial_model.ipynb)
+ [Details on data loading/preprocessing](notebooks/Tutorial_dataloading.ipynb)


## Adding Country Support
We learned that artifacts in data often require manual intervention to be fixed.
At the same time, data loading and processing must be fully automated to support running the data processing
for tens or hundreds of regions every day.

In this repository, we implemented a generalized data loading & processing interface that allows for:
+ supporting countries at national AND OR regional level
+ implementing country-specific interpolation / extrapolation / data cleaning routines
+ manual outlier removal & corrections

Contributions to add/improve country support are very welcome!

## Contributing
To contribute data sources, fix data outliers, or improve data quality for a specific country,
please open a PR for the corresponding `data_xy.py` file in [rtlive/sources](rtlive/sources).

Furthermore, we welcome contributions regarding...
+ testing
+ robustness against data outliers
+ computational performance
+ model insight

### How to run the code and notebooks
To be able to contribute, you'll need to be able to run the code and notebooks on this repo. To that end, we strongly recommend [installing the  Anaconda distribution](https://www.anaconda.com/products/individual) for python. This will make sure all the packages and necessary compilers come from the same source, and it will allow you to create dedicated virtual environments to run your code safely for each project independently.

Once Anaconda is installed, use the terminal and make sure you're in the root of the rtlive-global repository. Then, follow the following steps (still in the terminal):

- Create the virtual environment corresponding to this project: `conda env create -f environment.yml`. This will use the `environment.yml` file that is in the repo and install all the packages automatically. The first line of the `yml` file sets the new environment's name (here, `rtlive`)
- Activate the new environment: `conda activate rtlive`
- Verify that the new environment was installed correctly: `conda env list`
- If you want to run the notebooks, you have to tell Jupyter about this new environment. Still on the terminal and with the virtual env activated, just do: `python -m ipykernel install --user --name rtlive`
- Now, just type `jupyter lab` to launch and run the notebooks :tada

See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for detailed instructions if you have issues.

## Citing
To reference this project in a scientific article:
```
Kevin Systrom, Thomas Vladek and Mike Krieger. Rt.live (2020). GitHub repository, https://github.com/rtcovidlive/covid-model
```
or with the respective BibTeX entry:
```
@misc{rtlive2020,
  author = {Systrom, Kevin and Vladek, Thomas and Krieger, Mike},
  title = {Project Title},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rtcovidlive/covid-model}},
  commit = {...}
}
```