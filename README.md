# Quantifying solidification of metallic alloys with scikit-image

Materials presented at [BIDS ImageXD
2021](https://bids.berkeley.edu/events/imagexd-2021).

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cgusb/solidification-tracking.git/main?urlpath=%2Fvoila%2Frender%2Fslideshow.ipynb?voila-template=reveal) (view Voilà slideshow)

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/cgusb/solidification-tracking/main?filepath=slideshow.ipynb) (run Jupyter notebook)

## Setup

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or
[Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

    $ mamba env create -f environment.yml
    $ conda activate solidification-tracking
    $ voila --strip_sources=False --template=reveal slideshow.ipynb
