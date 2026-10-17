# The Deep MisUnderstanders · IBM Z Datathon 2026

Predicts housing abandonment risk in Ciudad Juárez using INEGI Census 2020 and IMIP data (urban growth, maquilas, crime, property values). Runs on IBM LinuxONE with real-time, explainable alerts for lenders and city planners.

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
| `01_exploracion.ipynb` | Joins Census and IMIP layers by AGEB and explores what is associated with vacant housing |
