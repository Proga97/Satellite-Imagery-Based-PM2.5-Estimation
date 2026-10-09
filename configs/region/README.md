# Adding a region

One file per region in this folder, named `<region>.yaml`, listed under `regions:` in
`configs/pipeline.yaml`. The pipeline needs only two fields:

```yaml
name: texas
epa_state_code: "48"   # two-digit EPA AQS state code, as a string
```

The state code decides which monitors are read from the EPA files; stations are
selected by their coverage (`labels.min_week_coverage` over `labels.coverage_years`).
The UTM zone of each station is derived from its longitude, so nothing else is needed.

EPA state codes: Alabama 01, Alaska 02, Arizona 04, Arkansas 05, California 06,
Colorado 08, Connecticut 09, Delaware 10, DC 11, Florida 12, Georgia 13, Hawaii 15,
Idaho 16, Illinois 17, Indiana 18, Iowa 19, Kansas 20, Kentucky 21, Louisiana 22,
Maine 23, Maryland 24, Massachusetts 25, Michigan 26, Minnesota 27, Mississippi 28,
Missouri 29, Montana 30, Nebraska 31, Nevada 32, New Hampshire 33, New Jersey 34,
New Mexico 35, New York 36, North Carolina 37, North Dakota 38, Ohio 39, Oklahoma 40,
Oregon 41, Pennsylvania 42, Rhode Island 44, South Carolina 45, South Dakota 46,
Tennessee 47, Texas 48, Utah 49, Vermont 50, Virginia 51, Washington 53,
West Virginia 54, Wisconsin 55, Wyoming 56, Puerto Rico 72.

Outside the United States the label stage (`01_build_labels.py`, `08_hour_labels.py`)
has to be replaced: it reads the EPA AQS file format. Everything after it only needs
a `stations.parquet` (station_id, lat, lon, site_name, region) and hourly values with
UTC timestamps.
