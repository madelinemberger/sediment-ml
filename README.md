# sediment-ml
Nearshore image classification

# Getting Started

- Install poetry, a python package and virtual environment manager
  - On macOS, in the terminal, run the command `brew install poetry` (assuming `homebrew` is installed. If not, [install homebrew](https://brew.sh/))
  - General installation instructions: [install poetry](https://python-poetry.org/docs/#installation)
- Clone this repository by running `git clone git@github.com:madelinemberger/sediment-ml.git` or [downloading the zip](https://github.com/madelinemberger/sediment-ml/archive/refs/heads/main.zip)
- Navigate into this repository
- Run the command `poetry install` to set up the virtual environment for this repository

# Running scripts

- After setting up the environment with `poetry` (see [Getting Started](#getting-started)), you can run a script like this:
  - run the command `poetry run python path/to/script.py`

# Running notebooks
- After setting up the environment, you can launch Jupyter Notebook from within the environment like this:
  - run the command `poetry run jupyter notebook`

# Script Information

## [`scripts/coastline_tile_pipeline.py`](https://github.com/madelinemberger/sediment-ml/blob/main/scripts/coastline_tile_pipeline.py)

This script does the following: Given a .tif file and a shapefile bundle with coastline information that spans the region of the .tif file, this script will
- tile the .tif into NxN regions (default=512x512 pixels);
- determine which of these regions intersect the coastline;
- convert the intersecting regions to PNGs and save these files;
- create a map image showing the intersecting tiles as PNGs and the coastline contour, along with the .tif boundary

Example output from this script can be seen in [`examples/coastline_tiles_plot.png`](https://github.com/madelinemberger/sediment-ml/blob/main/examples/coastline_tiles_plot.png) and below:

![Coastline-intersecting tiles highlighted](https://raw.githubusercontent.com/madelinemberger/sediment-ml/main/examples/coastline_tiles_plot.png)

## [`scripts/coastline_tile_pipeline_adjacent.py`](https://github.com/madelinemberger/sediment-ml/blob/main/scripts/coastline_tile_pipeline_adjacent.py)

This script does the following: Given a .tif and a shapefile bundle with coastline information that spans the region of the .tif file, this script will
- tile the .tif in NxN regions (default=512x512 pixels),
- determine which of these regions intersect the coastline,
- determine which of the regions *adjacent to* intersecting regions are *not* contained within the landmass,
- convert the intersecting regions and adjacent, non-land regions to PNGs and save these files,
- create a map image showing the intersecting tiles as red-boundary PNGs, the adjacent non-land tiles as orange-boundary PNGs, and the coastline contour as a blue line, along with the .tif boundary as a black line

Example output from this script can be seen in [`examples/coastline_tiles_plot_extended.png`](https://github.com/madelinemberger/sediment-ml/blob/main/examples/coastline_tiles_plot_extended.png) and below:

![Coastline-intersecting tiles, and adjacent tiles, highlighted](https://raw.githubusercontent.com/madelinemberger/sediment-ml/main/examples/coastline_tiles_plot_extended.png)




