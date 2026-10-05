# Fast crop stage and crop cycle models

This package provides fast re-implementations of two NASA Harvest / Agmatix models that read crop growth from an NDVI time series:

- [crop-cycle-detection](https://github.com/nasaharvest/crop-cycle-detection) finds the season start, green-up, NDVI peak and season end.
- [crop-stage-detection](https://github.com/nasaharvest/crop-stage-detection) assigns a stage from A to E: bare soil, green-up, peak, senescence, post-harvest.

The re-implementations give **the same results as the original code** and run **about 100–600 times faster**, fast enough to process a million pixels in under a minute. They work on any NDVI time series: a field, a pixel or a point.

The package also includes a pipeline that uses them to make **gridded maps** (30 m, 1 km and 9 km) of crop stage and crop cycle dates for corn and soybean, from Sentinel-2 + Landsat NDVI in Google Earth Engine. It has been run and checked for Boone County, Iowa (2023 and 2024) and Story County, Iowa (2024).

## Using the fast models on your own data

Copy `fast_models.py`, `fast_cycle.py` and `fast_stage.py`. They only need numpy, pandas, numba and scipy (numba uses scipy for linear algebra):

```bash
pip install numpy pandas numba scipy
```

This example uses the sample field from crop-cycle-detection (`sample_data/example_ndvi.csv`, Aug 2024 – Aug 2026):

```python
import pandas as pd
from fast_models import crop_cycle, crop_stages

df = pd.read_csv("example_ndvi.csv")           # columns: date, NDVI

crop_cycle(df["date"], df["NDVI"], year=2025)
# {'season_start': Timestamp('2025-05-27'), 'greenup': Timestamp('2025-06-09'),
#  'ndvi_peak': Timestamp('2025-08-18'), 'ndvi_peak_value': 0.84,
#  'season_end': Timestamp('2025-10-09'), 'season_length': 135.0, 'n_seasons': 1.0}

crop_stages(df["date"], df["NDVI"], as_of=["2025-05-01", "2025-06-15", "2025-08-01", "2025-09-15"])
# ['A', 'C', 'D', 'D']
```

- **`crop_cycle(dates, ndvi, year)`** runs the crop cycle model and returns that year's main season: the season with the highest NDVI peak in April–October, plus how many such seasons there were. Include some months before and after the season so the model can find the low points on both sides. A series that covers only the growing season (for example March–November) usually gives no season, in the original model too.
- **`crop_stages(dates, ndvi, as_of, lookback_days=150)`** returns the stage on each `as_of` date, using the observations from that date and the previous 150 days. This is the same as calling crop-stage-detection's `run_crop_stage_from_dataframe` on that window.
- **Input rules:** dates are used at day precision and must be unique, so combine same-day observations first. Missing NDVI values are dropped.
- **Parameters:** both use the models' default settings. For other settings, use the original code.

## Speed and validation

In a single process on an Apple M4 Pro:

| | Original code | Fast version | Speedup |
|---|---|---|---|
| Crop cycle model, per series | ~11–19 ms | ~0.1–0.19 ms | ~65–150× |
| Crop stage model, 34 weekly dates per series | ~92–95 ms | ~0.14–0.16 ms | ~600–650× |

`fast_cycle.py` and `fast_stage.py` repeat each step of the original as compiled numba code:

- **Shared smoothing:** daily resampling, PCHIP gap filling (using scipy's slope formulas) and Whittaker smoothing with λ = 6000, solved as a banded system.
- **Cycle:** the local-regression outlier and spike removal, scipy's `argrelmax`/`argrelmin` edge rules, trough-to-trough seasons, the start / end / green-up rules, duplicate-season removal and overlap clipping.
- **Stage:** the constant-velocity Kalman trend and the threshold and peak rules for A–E, for all requested dates in one call.

Comparisons with the original code:

| Data | Checked | Mismatches |
|---|---|---|
| Sample NDVI files from both NASA Harvest repos (per-pass field series) | cycle for every year; stage on 155 weekly dates | 0 |
| Boone County 2024, 1 km | all 2,581 series: 7 cycle outputs and 87,754 stage-weeks | 0 |
| Boone County 2024, 30 m | 20,000 random pixels (cycle); 102,000 stage-weeks | 0 |
| Boone County 2023 (30 m) and Story County 2024 (1 km) | 5,646 series, cycle and stage | 0 |
| `fast_models.py` on Boone 2024 series treated as dated observations | 1,500 cycle years and 31,856 stage dates | 0 |
| Smoothed NDVI | 200 series | agrees within 1e-12 |
| Gridded products | all 18 Boone files at 9 km, 1 km and 30 m | identical values |

These also hold with the latest package releases at the time of testing (including pandas 3.0 and numpy 2.5). To repeat the checks, run `validate_fast_models.py`, `validate_fast_cycle.py` and `validate_fast_stage.py` (they need the NASA Harvest repos, see below). Re-run them after updating either repo.

## Gridded maps

### Setup

Tested with Python 3.13, both with the package versions in `requirements.txt` and with the latest releases at the time. You also need a Google Cloud project registered for Earth Engine.

```bash
pip install -r requirements.txt
earthengine authenticate

# The two NASA Harvest repos, pinned to the versions this package was tested with
mkdir -p external && cd external
git clone https://github.com/nasaharvest/crop-cycle-detection && git -C crop-cycle-detection checkout 293f906
git clone https://github.com/nasaharvest/crop-stage-detection && git -C crop-stage-detection checkout 2005f53
cd ..
```

If you already have the repos somewhere else, set `HARVEST_REPOS` to the folder that contains both of them instead.

### Run

```bash
# 1. NDVI from Earth Engine (Boone County: about 1–2 min at 1 km, 2–4 min at 30 m)
python gee_ndvi.py direct   --project <your-ee-project> --year 2024 --region boone
python gee_ndvi.py direct30 --project <your-ee-project> --year 2024 --region boone

# 2. Both models on every pixel (Boone, all three grids, 1.09M pixels at 30 m: about 1 min on 11 cores)
python run_models.py --year 2024 --region boone --grid 9km 1km 30m

# 3. Figures
python plot_maps.py --year 2024 --region boone
```

Use `--workers N` to set the number of CPU cores for `run_models.py`. It defaults to all but one. Add `--cycle-impl original` or `--stage-impl original` to use the original NASA Harvest code instead of the fast version.

### Adding a region

```bash
python gee_ndvi.py add-region --project <your-ee-project> --state Iowa --county Story
```

This looks up the county in the US Census TIGER boundaries, fits a window on the shared 1 km grid around it, and saves it to `regions.json` under the county's name in lower case (here `story`). After that, use `--region story` with every script. Names, postal codes (`IA`) and FIPS codes (`--state 19 --county 169`) all work, and `--name` sets a different key.

Direct mode supports up to 100 × 100 km, so a whole state needs to be run county by county.

### How it works

```
gee_ndvi.py ──► data/*.npz ──► run_models.py ──► products/<region>/<grid>/*.tif ──► plot_maps.py ──► figures/*.png
                                    │
                                    ├─ cycle: fast_cycle.py (default) or crop-cycle-detection (--cycle-impl original)
                                    └─ stage: fast_stage.py (default) or crop-stage-detection (--stage-impl original)
```

| File | Role |
|---|---|
| `fast_models.py` | date-based entry points: `crop_cycle`, `crop_stages` |
| `fast_cycle.py` | numba version of the crop-cycle-detection model |
| `fast_stage.py` | numba version of the crop-stage-detection model |
| `common.py` | grids, regions, 5-day bins, NASS week dates, file paths |
| `gee_ndvi.py` | builds and downloads NDVI from Earth Engine; `add-region` |
| `run_models.py` | runs both models on every pixel and writes the GeoTIFFs |
| `plot_maps.py` | map and timeline figures |
| `validate_fast_models.py` | compares `fast_models.py` with the original code |
| `validate_fast_cycle.py` | compares `fast_cycle.py` with the original code on gridded data |
| `validate_fast_stage.py` | compares `fast_stage.py` with the original code on gridded data |

#### NDVI

The NDVI recipe follows `gee_fetch.py` in crop-stage-detection:

- **Sentinel-2:** `COPERNICUS/S2_SR_HARMONIZED`, 10 m, B8/B4. Pixels are kept where Cloud Score+ `cs` > 0.6.
- **Landsat 8 and 9:** Collection 2 Level-2, 30 m, SR_B5/SR_B4. Pixels are kept where `QA_PIXEL` is 21824 or 21888 (clear).
- **Crop pixels:** USDA CDL of the same year (corn = 1, soybean = 5), eroded by one CDL pixel to drop field edges. CDL is 10 m from 2024 on and 30 m before, so the edge buffer is 10 m or 30 m and earlier years keep fewer pixels.

All clear observations from the three sensors are averaged per 30 m pixel in **5-day bins**. The bins run from Nov 1 of the previous year to Jan 31 of the next year, so the cycle model can find the winter lows on both sides of the season.

- **1 km:** the mean NDVI of each crop's 30 m pixels inside the cell. A bin is used only if at least 50% of the cell's crop pixels are clear, and a cell needs at least 5% crop cover.
- **9 km:** a weighted mean of the 1 km cells on the USDA NASS gridded Crop Progress & Condition grid. Pass `--cpc-ref <file.tif>` to `run_models.py` to align with a specific CPC file.

Pixels with fewer than 10 usable bins are skipped.

#### Differences from the published workflow

- Harvest averages NDVI over field polygons and keeps each satellite pass, preferring Sentinel-2. The gridded pipeline uses grid pixels and 5-day mean composites with no cross-sensor adjustment.
- Sentinel-2 is resampled to 30 m before use.
- The stage model is evaluated at each NASS week-ending Sunday (ISO weeks 14–47), using only bins complete by that date and a 150-day lookback (the model default).

### Outputs

Everything is written to `products/<region>/<grid>/` as GeoTIFFs in EPSG:5070 (NAD83 / CONUS Albers).

**`cycle_<crop>_<year>.tif`** (float32, NaN = not modeled)

| Band | Meaning |
|---|---|
| `season_start_doy` | NDVI starts rising (after planting) |
| `greenup_doy` | fastest green-up |
| `ndvi_peak_doy` | date of peak NDVI |
| `ndvi_peak_value` | peak NDVI |
| `season_end_doy` | NDVI back down after senescence or harvest |
| `season_length` | end minus start, in days |
| `n_seasons` | number of seasons with a peak in Apr–Oct of `<year>` |

Dates are day of year counted from Jan 1 of `<year>`. If a pixel has more than one season, bands 1–6 describe the one with the highest peak.

**`stage_<crop>_<year>_weekly.tif`** (uint8) has one band per NASS week, named like `w30_2024-07-28`. Values: 1–5 = stage A–E, 0 = insufficient data, 255 = not a crop pixel.

**`stage_<crop>_<year>_summary.csv`** gives the share of crop pixels in each stage per week.

## Known limitations

- **Accuracy:** the models were validated by their authors on about 53,000 fields in North Carolina, not in Iowa. The gridded products have not yet been compared with NASS crop progress.
- **Stage C:** the stage model sets its peak threshold from the NDVI seen so far, so stage C means "near the highest NDVI so far," not physiological maturity.
- **Season start:** this marks the NDVI rise, which typically comes 2–3 weeks after planting.
- **Snow:** snow passes both cloud masks and shows up as winter NDVI near 0.
- **Region size:** the gridded pipeline only handles county-sized regions. For a whole state, the models would take roughly an hour at 30 m, but the NDVI download is county by county (about 4 minutes and 275 MB per county).

## Authors

Meijian Yang, Columbia University & NASA GISS: the fast re-implementations (`fast_cycle.py`, `fast_stage.py`, `fast_models.py`), the gridded pipeline and NDVI processing.

The crop cycle and crop stage models are by Ran Pelta (Agmatix / GrowersTech, NASA Harvest).

## Citation and license

Please cite the models:

- Pelta, R. (2026). *Crop Cycle Detection*. NASA Harvest / Agmatix. https://github.com/nasaharvest/crop-cycle-detection
- Pelta, R. (2026). *Crop Stage Detection*. NASA Harvest / Agmatix. https://github.com/nasaharvest/crop-stage-detection

Licensed under Apache 2.0 (see `LICENSE`). `fast_cycle.py` and `fast_stage.py` are modified re-implementations of crop-cycle-detection and crop-stage-detection; see `NOTICE` for attribution.
