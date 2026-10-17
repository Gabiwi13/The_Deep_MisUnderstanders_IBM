"""Re-download every raw dataset into data/raw/. Run from the repo root:
    python scripts/download_data.py
"""
import json
import subprocess
import tempfile
import urllib.parse
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
MUN = "037"  # Juárez, Chihuahua (state 08)

INEGI_CENSO = "https://www.inegi.org.mx/contenidos/programas/ccpv/2020/datosabiertos/ageb_manzana/ageb_mza_urbana_08_cpv2020_csv.zip"
INEGI_MG = "https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/geografia/marcogeo/889463807469/08_chihuahua.zip"
IMIP = "https://sigimip.org.mx/server/rest/services/"
IMIP_CRECIMIENTO = "https://sigimip.org.mx/download/crecimiento/chehistoGEO_F7.geojson"
IMIP_LAYERS = {
    "maquilas_2022": "Industria/FeatureServer/0",
    "parques_industriales": "Industria/FeatureServer/2",
    "delitos_2020": "incidenciadelictiva/FeatureServer/38",
    "delitos_2024": "incidenciadelictiva/FeatureServer/42",
    "delitos_2025": "incidenciadelictiva/FeatureServer/43",
    "delitos_2026": "incidenciadelictiva/FeatureServer/50",
    "victimas_homicidio": "incidenciadelictiva/FeatureServer/17",
    "valor_catastral_zonas_2025": "Valorescatastrales/FeatureServer/3",
    "estaciones_brt": "brt/FeatureServer/0",
    "primarias": "Equipamiento/FeatureServer/1",
    "unidades_medicas": "Equipamiento/FeatureServer/5",
    "niveles_bienestar_2010_2020": "Radiografia2025/FeatureServer/58",
}


def curl(url, out=None):
    # curl instead of urllib: INEGI rejects default Python user agents
    # retries: the IMIP server sometimes hangs a request and answers the next one
    cmd = ["curl", "-sSL", "--fail", "--max-time", "300", "--retry", "4", "--retry-all-errors",
           "--retry-delay", "5", "-A", "Mozilla/5.0", url]
    if out:
        subprocess.run(cmd + ["-o", str(out)], check=True)
        return None
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def censo(tmp):
    z = tmp / "censo.zip"
    curl(INEGI_CENSO, z)
    with zipfile.ZipFile(z) as zf:
        csv = next(n for n in zf.namelist() if n.endswith(".csv") and "conjunto_de_datos" in n)
        dic = next(n for n in zf.namelist() if n.endswith(".csv") and "diccionario" in n)
        df = pd.read_csv(zf.open(csv), dtype=str)
        (RAW / "diccionario_variables.csv").write_bytes(zf.read(dic))
    df[df.MUN == MUN].to_csv(RAW / "juarez_censo2020_ageb_manzana.csv", index=False)


def marco_geo(tmp):
    z = tmp / "mg.zip"
    curl(INEGI_MG, z)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(tmp / "mg", [n for n in zf.namelist() if n.split("/")[-1].startswith(("08a.", "08m."))])
    out = RAW / "inegi_marco_geo_2020_juarez.gpkg"
    out.unlink(missing_ok=True)
    for shp, layer in [("08a.shp", "ageb"), ("08m.shp", "manzana")]:
        g = gpd.read_file(next((tmp / "mg").rglob(shp)))
        g[g.CVE_MUN == MUN].to_file(out, layer=layer, driver="GPKG")


def imip_layer(name, path):
    feats, offset = [], 0
    while True:  # the server returns at most 2000 records per request
        q = urllib.parse.urlencode(dict(where="1=1", outFields="*", outSR=4326, f="geojson",
                                        resultOffset=offset, resultRecordCount=2000))
        page = json.loads(curl(f"{IMIP}{path}/query?{q}")).get("features", [])
        feats += page
        if len(page) < 2000:
            break
        offset += 2000
    (RAW / "imip" / f"{name}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return len(feats)


if __name__ == "__main__":
    (RAW / "imip").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        censo(Path(t))
        print("censo ok")
        marco_geo(Path(t))
        print("marco geoestadistico ok")
    curl(IMIP_CRECIMIENTO, RAW / "imip" / "crecimiento_historico.geojson")
    for name, path in IMIP_LAYERS.items():
        print(name, imip_layer(name, path))
