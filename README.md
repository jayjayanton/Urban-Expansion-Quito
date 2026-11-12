# Urban Expansion Quito

## Study Areas
Nayón, Cumbayá, Tumbaco, Puembo (Pifo excluded)

## Goals
- Sentinel-2 L2A acquisition
- Cloud masking and composites
- NDVI/NDBI analysis
- Historical land-cover and urban expansion
- Future PyTorch CNN investigation

## Status
Phase 1: Annual Sentinel-2 composite acquisition demonstrated for the Tumbaco test AOI. Quality assessment and urban-change validation remain pending.

## Notes
10m resolution limits small parcel detection.

## Sentinel Hub Evalscript methodology and references

The planned Sentinel-2 processing pipeline uses a **custom Evalscript V3** submitted through the Sentinel Hub Process API. The script is project-specific; it is **not** an official prevalidated monthly compositing algorithm.

Evalscript V3 requires `setup()` to define input bands and output format and `evaluatePixel()` to process the selected pixel samples. Our proposed approach uses spectral bands B02, B03, B04, B08, B11, B12, the Scene Classification Layer (SCL), and `dataMask` to create cloud-screened reflectance composites for Python-defined annual, quarterly, or monthly intervals and observation counts. The use of `ORBIT` mosaicking and the resulting temporal aggregation will be checked against source acquisitions before full-scale processing. SWIR bands B11/B12 have native 20 m resolution, even when exported on a 10 m grid.

**Official documentation:**
- [Sentinel Hub Evalscript V3 specification](https://docs.sentinel-hub.com/api/latest/evalscript/v3/) — `setup`, `evaluatePixel`, input/output configuration, mosaicking, and sample types.
- [Copernicus Data Space Evalscript documentation](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Evalscript.html) — CDSE-specific usage and processing behavior.
- [Copernicus Evalscript examples](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Evalscript/Examples.html) — examples of NDVI, multitemporal calculations, and change detection.

**Validation pending:** cloud/shadow masking, missing-data treatment, valid-observation counts, temporal consistency, and georeferencing. Annual composite examples are documented below; no validated urban-change model outputs are included yet.

## Installation and dependencies

Python 3.10 or newer is recommended. Clone the repository and open a terminal in its root directory:

```bash
git clone https://github.com/jayjayanton/Urban-Expansion-Quito.git
cd Urban-Expansion-Quito
python -m venv .venv
```

Activate the virtual environment:

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

Install the project's Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the imports:

```bash
python -c "import numpy, rasterio, dotenv, sentinelhub; print('Dependencies OK')"
```

The `requirements.txt` file includes `numpy`, `rasterio`, `python-dotenv`, and `sentinelhub`. Python's built-in `os` and `pathlib` modules require no separate installation.

### API credentials

Create a local `.env` file in the repository root containing the following **placeholders replaced with your own Sentinel Hub OAuth client credentials**:

```dotenv
SH_CLIENT_ID=your_client_id
SH_CLIENT_SECRET=your_client_secret
```

Never commit `.env` or share the client secret. The project `.gitignore` should exclude credentials and local ArcGIS Pro workspace files. Check imagery sizes before committing rasters to GitHub.

After setup, run the acquisition script from the repository root once it has been tested and its path confirmed. Large GeoTIFF outputs are intended to remain local rather than being committed to GitHub.

## Step 1 — Sentinel-2 annual composite examples (2016 vs 2025)

The first reproducible acquisition milestone is to retrieve Sentinel-2 Level-2A observations for a small test AOI near Tumbaco, screen observations using the Scene Classification Layer (SCL), and calculate **per-band annual median surface reflectance** with a custom Sentinel Hub Evalscript. Python saves six reflectance bands (B02, B03, B04, B08, B11, B12) plus a valid-observation-count band in a GeoTIFF.

**Visual comparison (true color: B04/B03/B02):**

| 2016 annual composite | 2025 annual composite |
|:---:|:---:|
| ![Tumbaco 2016 annual Sentinel-2 RGB](assets/sentinel2_2016_rgb.png) | ![Tumbaco 2025 annual Sentinel-2 RGB](assets/sentinel2_2025_rgb.png) |

The images are display-oriented PNG exports derived from the scientific multiband GeoTIFFs; the original spectral values are retained in local GeoTIFFs for subsequent analysis. Both PNGs use a common RGB stretch for visual comparability. This is an **illustrative before/after comparison**, not yet a validated map of urban expansion. Differences may reflect seasonal sampling, atmosphere, residual cloud contamination, or land-cover change.

**Image assets:** Place `sentinel2_2016_rgb.png` and `sentinel2_2025_rgb.png` in `assets/` for the table above to render on GitHub. Large source GeoTIFFs are intentionally excluded from version control.


## Step 2 — Acquisition frequency benchmark methodology

The project now provides **three separate entrypoints** using one shared acquisition engine and the same cloud-screening/median Evalscript V3. The evalscript was renamed from `scripts/monthly_composite.js` to `scripts/median_composite.js` because it does **not** select dates. Python defines the temporal interval; the evalscript selects bands B02/B03/B04/B08/B11/B12, filters observations using SCL classes 4/5/6/7 and dataMask, and returns six per-band median reflectances plus a valid orbit-level observation count. It is not an arithmetic average or a single selected acquisition.

### Common benchmark geometry

All three benchmark scripts use the same **5,000 m × 5,000 m square (25 km²)** centered near Tumbaco at longitude -78.41, latitude -0.22. The center is projected to **EPSG:32717** before constructing the exact metric square; the output is **500 × 500 pixels at 10 m**. This controlled tile is intentionally smaller than the broader Quito-valleys study AOI. Change `CENTER_LON_LAT` and `SIDE_METERS` in `scripts/acquisition_common.py` to adjust the benchmark. Full regional tiling and mosaicking are planned, not yet implemented.

### Run each strategy

From the repository root (after installing `requirements.txt` and configuring local OAuth credentials):

```powershell
python scripts/download_annual.py
python scripts/download_quarterly.py
python scripts/download_monthly.py
```

| Strategy | Temporal request | Requests per year per tile | Output directory |
|---|---|---:|---|
| Annual | January–December | 1 | `data/benchmarks/annual/<year>/` |
| Quarterly | Q1, Q2, Q3, Q4 | 4 | `data/benchmarks/quarterly/<year>/` |
| Monthly | Calendar months | 12 | `data/benchmarks/monthly/<year>/` |

The shared engine currently runs **2016 only** with `YEARS = [2016]`. The full historical loop is preserved as a commented line (`# YEARS = range(2016, 2026)`) in `scripts/acquisition_common.py`. To run all years after validating the 2016 benchmark, uncomment the full range and comment out the 2016-only line. Each output is saved separately, with the date interval, processing duration, valid-pixel coverage and evalscript SHA-256 in its metadata. Previously completed files are skipped only if the raster geometry and relevant processing metadata match. API throttling or processing limits can still occur, particularly for annual requests.

### Benchmark interpretation and next stages

Compare request success/failure, elapsed processing time, output size, valid coverage, and observation counts across identical AOIs and years. **Monthly/quarterly medians cannot simply be averaged or median-combined to recover the exact annual median of all source observations**. A separate, documented temporal aggregation method will be required for local annual products. Likewise, a future `mosaic_tiles.py` will spatially combine aligned tiles for the wider study area; it is not part of this initial benchmark. No claim of improved efficiency is made until timings and data quality are measured.

For training, align ESA WorldCover 2020/2021 to the Sentinel-2 10 m grid using nearest-neighbor resampling; WorldCover built-up class 50 can provide candidate binary labels, but independent reference checks remain necessary. WorldCover's 2020/2021 maps are not annual labels for the entire 2016–2025 period.

