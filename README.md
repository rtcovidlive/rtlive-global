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
