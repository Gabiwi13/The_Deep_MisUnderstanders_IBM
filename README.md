# The Deep MisUnderstanders · IBM Z Datathon 2026

Predicts housing abandonment risk in Ciudad Juárez using INEGI Census 2020 and IMIP data (urban growth, maquilas, crime, property values). Runs on IBM LinuxONE with real-time, explainable alerts for lenders and city planners.

## Problem statement

**Ciudad Juárez has 55,000 vacant homes, yet credit, development and home-buying decisions are made without knowing which neighborhoods are likely to empty out or how safe they are.**

### Context

- **Scale:** 55,595 vacant homes in the 2020 Census (about 1 in 8), while about 35,000 new homes are planned, mostly on the periphery (Orquiz, 2025).
- **Impact:** vacant homes concentrate crime (Fuentes Flores, 2016), lower nearby property values and leave public infrastructure unused. Families face insecurity, long commutes and homes that lose value.
- **Cause:** two decades of cheap peripheral housing built far from jobs and services (Pérez Pulido & Romo Aguilar, 2022; Herrera-Terrazas et al., 2019).

### What our data shows

1. **Two kinds of abandonment.** The aging historic center has a high share of empty homes (20 to 30%) but few in total. Distant 2006 to 2008 developments, 20 to 26 km out, have thousands of empty homes, with blocks reaching 75 to 92% empty.
2. **Nearby jobs reduce vacancy.** More maquilas within 3 km means fewer empty homes.
3. **Vacancy and crime go together.** The link is weak but consistent. Crime reports grew from about 5,700 (2020) to about 10,700 (2025).
4. **Location and access explain abandonment more than price.**

### The gap

- **Lenders and planners** approve mortgages and developments without a forward-looking abandonment risk.
- **Buyers and people moving to Juárez** have no single place to check whether a neighborhood is emptying out, whether crime is rising, and how far it is from jobs and services.
- **Research looks backward** (2010 data, case studies), and the data that exists (Census, urban growth, jobs, crime, services) is scattered.

### Our question

> For any neighborhood in Juárez, can we estimate the risk that its homes end up vacant, explain why, show its crime trend and access to jobs and services, and update all of it as new data arrives?

### Users

| User | Decision | What they get |
|---|---|---|
| Lenders (INFONAVIT, banks) | Approve a mortgage | Abandonment risk and reasons |
| City planners (IMIP, municipality) | Approve a development, prioritize rehabilitation | Risk map and alerts as new data arrives |
| Buyers and newcomers | Where to buy or rent | Neighborhood profile: risk, crime trend, top crime types, distance to jobs, BRT, schools, health |

### Scope

- **In:** Juárez urban area, neighborhood level (AGEB and block), public data only.
- **Out:** individual household data, price forecasting, other cities (future scaling).

### Responsible use

- Transparent profiles with sources, not "red zone" maps, to avoid stigmatizing neighborhoods.
- Crime shown as an association, not a cause, and per km² so low-population commercial areas are not inflated.
- Crime report growth may partly reflect reporting changes; we say "reports grew", not "crime grew".

### Success in 24 hours

1. A model that beats a simple baseline, validated by neighborhood.
2. Explainable risk per area, e.g. *"High risk: built 2007, 24 km from center, 1 maquila within 3 km, crime up since 2020."*
3. Real-time demo on LinuxONE: monthly crime records update scores and trigger alerts.
4. A neighborhood profile a buyer can read.
5. Pitch line: *"55,000 empty homes. We show where the next ones will be, before the credit, the permit or the purchase."*

**Theme:** Real-Time AI for Critical Decisions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name ibm-datathon-juarez --display-name "Python (IBM Datathon Juárez)"
jupyter lab
```

Open `notebooks/` and pick the **Python (IBM Datathon Juárez)** kernel.

## Data

Everything is already in `data/raw/`. To download it again from the sources:

```bash
python scripts/download_data.py
```

| File | Source |
|---|---|
| `juarez_censo2020_ageb_manzana.csv`, `diccionario_variables.csv` | INEGI, Censo de Población y Vivienda 2020 (AGEB and block, Juárez only) |
| `inegi_marco_geo_2020_juarez.gpkg` | INEGI, Marco Geoestadístico 2020 (layers `ageb`, `manzana`) |
| `imip/crecimiento_historico.geojson` | IMIP Juárez, historic urban growth 1856 to 2023 |
| `imip/*.geojson` | IMIP Juárez SIGMUN public services: crime 2020/2024/2025/2026, homicides, maquilas, industrial parks, property values 2025, wellbeing levels, BRT, schools, health units |

## Notebooks

| Notebook | Content |
|---|---|
| `01_exploracion.ipynb` | Explores what is associated with vacant housing by AGEB, from `data/ageb_features.csv` (no geopandas needed) |
| `02_model_selection.ipynb` | Compares candidate models with the protocol below and saves `results/` (run on LinuxONE) |
| `03_risk_model.ipynb` | Final XGBoost model: out-of-fold risk score per block, risk level (high = top 10%, medium = next 20%), three reasons per block; saves `results/block_risk.csv` and `results/xgb_risk_model.json` |

## Running on LinuxONE

pip has no s390x builds of the map libraries, so the work is split on the same instance:

| Step | Where | Needs |
|---|---|---|
| `python3 scripts/prepare_features.py` writes `data/features.csv` and `data/ageb_features.csv` | Instance host (Ubuntu 24.04) | `sudo apt install -y python3-geopandas` |
| All notebooks (exploration, modeling, alerts) | Official Jupyter container ([jupyter-lab-ml](https://github.com/linuxone-community-cloud/jupyter-lab-ml)) | pandas, scikit-learn, XGBoost (in the image) plus `requirements-linuxone.txt` (Snap ML, Python 3.12 or lower) |

`requirements.txt` is for laptops (it includes the map libraries). On LinuxONE use `requirements-linuxone.txt`; notebook 02 installs it automatically when it detects s390x, or run `pip install -r requirements-linuxone.txt` yourself.

Both feature files are committed, so the container can run without the host step. If apt fails, run `prepare_features.py` on a laptop and copy the file with `scp`.

## Model selection

We choose the model by experiment, not by preference. Every candidate uses the same data, validation and decision rule.

- **Data:** `data/features.csv`, 16,452 city blocks with 10 or more homes. Target: % vacant homes. Features use only shares of residents or of inhabited homes, so nothing leaks the target. INEGI masks 1 or 2 vacant homes as `*`; we use 1.5.
- **Validation:** 5-fold cross-validation grouped by neighborhood (AGEB). Models are always scored on neighborhoods they never saw, like a new development.
- **Candidates:** median baseline, Ridge, Random Forest, scikit-learn Gradient Boosting, XGBoost, Snap ML.
- **Metrics, by priority:**
  1. Hit rate in the top 10%: of the blocks flagged as highest risk, the share that really are >= 25% vacant.
  2. Share of vacant homes captured in that top 10%.
  3. AUC.
  4. Stability across folds.
  5. Training and scoring time on LinuxONE.
  6. Explanation per block (yes, partial, no).
- **Decision rule:** best mean hit rate wins. Models within one fold standard deviation of the best are tied; ties are broken by explainability, then scoring speed.
- **Variants with the winner:** target (% vacant or >= 25% yes/no) and crime feature (city percentile, raw count, none).

### Results

Run on IBM LinuxONE (s390x, Python 3.12.15, XGBoost 3.4.1). Snap ML installed but failed to import (`AttributeError`), so it was skipped.

| Model | Hit rate top 10% | Vacant homes captured | AUC | Hit rate std | Fit (s/fold) | Scoring (ms per 1k blocks) | Explains each block |
|---|---|---|---|---|---|---|---|
| Gradient Boosting (sklearn) | 0.488 | 0.167 | 0.842 | 0.063 | 0.53 | 7.1 | partial |
| **XGBoost** | 0.481 | 0.169 | 0.843 | 0.072 | 1.37 | 6.7 | **yes** |
| Random Forest | 0.462 | 0.168 | 0.829 | 0.083 | 25.7 | 33.4 | partial |
| Ridge | 0.341 | 0.123 | 0.732 | 0.059 | 0.08 | 1.2 | yes |
| Baseline (median) | 0.120 | 0.102 | 0.500 | 0.031 | 0.01 | 0.3 | n/a |

Base rate of high-vacancy blocks: 0.113.

**Decision: XGBoost.** Gradient Boosting, XGBoost and Random Forest are within one fold standard deviation of the best, so they tie. XGBoost wins the tie because it explains each block natively (`pred_contribs`). Its hit rate is about 4.3 times the base rate: of the 10% of blocks it flags, 48% really are >= 25% vacant.

**Variants with XGBoost:**

| Variant | Hit rate top 10% | Vacant homes captured | AUC |
|---|---|---|---|
| Regression, no crime | 0.491 | 0.171 | 0.846 |
| Regression, crime percentile | 0.481 | 0.169 | 0.843 |
| Regression, crime raw count | 0.481 | 0.169 | 0.843 |
| Classifier >= 25%, crime percentile | 0.478 | 0.154 | 0.841 |

All variants are within the fold spread. Crime does not add accuracy, so its role is the real-time alerts and the buyer profile, not the risk score. Regression stays: it ranks as well as the classifier and captures more vacant homes.

## References

### Literature

Carrillo, M. J., & Pérez, J. (2025). The communal roots of Mexico's maquila industry: Urbanization, land, and inequality in Ciudad Juárez, 1960–2000. *Latin American Research Review, 60*(3), 648–671. https://doi.org/10.1017/lar.2024.35

Fuentes Flores, C. M. (2016). El impacto de las viviendas deshabitadas en el incremento de delitos (robo a casa habitación y homicidios) en Ciudad Juárez, Chihuahua, 2010. *Frontera Norte, 27*(54), 171–196. https://doi.org/10.17428/rfn.v27i54.581

Herrera-Terrazas, L., Peña Barrera, L., & Ordóñez Hernández, G. (2019). Habitabilidad urbana en fraccionamientos periféricos en Ciudad Juárez Chihuahua, México. In *XI Seminario Internacional de Investigación en Urbanismo*. Universitat Politècnica de Catalunya. https://doi.org/10.5821/siiu.6833

Holguín Ávila, R., & Maycotte Pansza, E. (2024). Invasion of social housing and the perception of loneliness: A case study on subdivisions with uninhabited homes in the southeast of Ciudad Juárez. *Nóesis. Revista de Ciencias Sociales, 33*(66), 57–74. https://doi.org/10.20983/noesis.2024.2.4

Pérez Pulido, L. A., & Romo Aguilar, M. de L. (2022). Planes de desarrollo urbano: Instrumentos de legitimación en la expansión urbana de Ciudad Juárez, Chihuahua. *Estudios Demográficos y Urbanos, 37*(1). https://doi.org/10.24201/edu.v37i1.1924

Velázquez Angulo, G., Maycotte Pansza, E., & Olguín, R. (2026). *La evolución de la vivienda deshabitada en el suroriente de Ciudad Juárez: Una mirada retrospectiva al periodo 2002-2019*. Universidad Autónoma de Ciudad Juárez.

### Press

Orquiz, M. (2025, April 4). Harán 35 mil casas nuevas en Ciudad Juárez, pero hay 56 mil abandonadas. *Norte Digital*. https://nortedigital.mx/haran-35-mil-casas-nuevas-en-ciudad-juarez-pero-hay-56-mil-abandonadas

### Data

Instituto Municipal de Investigación y Planeación de Ciudad Juárez. (2026). *Sistema de Información Geográfica Municipal (SIGMUN)* [Data set]. Retrieved October 17, 2026, from https://sigimip.org.mx/

Instituto Nacional de Estadística y Geografía. (2021). *Censo de Población y Vivienda 2020: Principales resultados por AGEB y manzana urbana* [Data set]. https://www.inegi.org.mx/programas/ccpv/2020/

Instituto Nacional de Estadística y Geografía. (2021). *Marco Geoestadístico, diciembre 2020* [Data set]. https://www.inegi.org.mx/temas/mg/
