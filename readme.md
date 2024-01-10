# "Machine learning for the detection of atmospheric blocking events during European summer"

## corresponding thesis
tbd

## setup

### environment

this repository is using [poetry](https://python-poetry.org/) for package management. to get running follow the install instructions for poetry. setup your dependencies with
-  `poetry install`

To activate the created environment use `poetry shell`. You can also execute specific commands directly in the environment without activating by using `poetry run`.

### data

to get running the correct data from era5 observational dataset with geopotential height (z500) is required. the data is transformed to be suitable for the learning approaches used in this project.

expert labels are used from [Thomas et al](https://doi.org/10.5194/wcd-2-581-2021) and paired to the daily training data