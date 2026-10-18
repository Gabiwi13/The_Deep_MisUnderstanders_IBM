"""Build data/features.csv (one row per city block, model target and features),
data/ageb_features.csv (one row per neighborhood, notebooks 01 and 04) and
data/crimes_by_ageb.csv (IMIP crime records tagged with their neighborhood, notebook 04).
Run from the repo root:
    python scripts/prepare_features.py

Needs geopandas. On LinuxONE (s390x) pip has no builds, use Ubuntu packages on the host:
    sudo apt install -y python3-geopandas
Written for geopandas >= 0.14 (the Ubuntu 24.04 version).
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMIP = RAW / "imip"
UTM = 32613  # UTM zone 13N, distances in meters
MIN_DWELLINGS = 10
CENTRO = (-106.4845, 31.7395)  # approx. historic center (lon, lat)


def imip(name):
    return gpd.read_file(IMIP / f"{name}.geojson").to_crs(UTM)


def nearest(points, layer, max_distance=None):
    """Position in `layer` of the feature nearest to each point, and the distance in meters.
    Replaces gpd.sjoin_nearest, which geopandas 0.10 (Ubuntu 22.04) only supports with PyGEOS."""
    from shapely.strtree import STRtree
    geoms = list(layer.geometry)
    tree = STRtree(geoms)
    find = tree.nearest_item if hasattr(tree, "nearest_item") else tree.nearest  # shapely 1.8 / >= 2.0
    pos = np.array([find(p) for p in points.geometry])
    d = np.array([p.distance(geoms[i]) for p, i in zip(points.geometry, pos)])
    out = pd.DataFrame({"pos": pos.astype(float), "d": d}, index=points.index)
    if max_distance is not None:
        out.loc[out.d > max_distance, "pos"] = np.nan
    return out


def load_blocks():
    c = pd.read_csv(RAW / "juarez_censo2020_ageb_manzana.csv", dtype=str)
    m = c[(c.MZA != "000") & (c.NOM_LOC != "Total AGEB urbana")].copy()
    keys = ["ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "NOM_LOC", "AGEB", "MZA"]
    masked = m["VIVPAR_DES"] == "*"
    num = [k for k in m.columns if k not in keys]
    m[num] = m[num].apply(pd.to_numeric, errors="coerce")
    m = m.copy()  # defragment after converting ~220 columns
    m["CVEGEO"] = m.ENTIDAD + m.MUN + m.LOC + m.AGEB + m.MZA
    m["AGEB_KEY"] = m.CVEGEO.str[:13]
    m["vacant_masked"] = masked
    m = m[m.TVIVPAR >= MIN_DWELLINGS].copy()
    # INEGI masks values of 1 or 2 as '*': use the midpoint instead of dropping (dropping overstates vacancy)
    vacant = np.where(m.vacant_masked, 1.5, m.VIVPAR_DES)
    m["target_pct_vacant"] = vacant / m.TVIVPAR * 100
    return m


def census_features(m):
    # Shares of residents or of inhabited homes only: anything over total housing stock would leak the target
    f = pd.DataFrame(index=m.index)
    f["avg_occupants"] = m.PROM_OCUP
    f["schooling_years"] = m.GRAPROES
    for col, den in [("POB0_14", "POBTOT"), ("POB65_MAS", "POBTOT"), ("PDER_SS", "POBTOT"), ("PEA", "P_12YMAS")]:
        f[f"pct_{col}"] = m[col] / m[den] * 100
    for col in ["VPH_INTER", "VPH_AUTOM", "VPH_PISOTI", "VPH_C_ELEC", "VPH_AGUADV",
                "VPH_DRENAJ", "VPH_PC", "VPH_CEL", "VPH_1CUART"]:
        f[f"pct_{col}"] = m[col] / m.TVIVPARHAB * 100
    return f


def spatial_features(m, f):
    blocks = gpd.read_file(RAW / "inegi_marco_geo_2020_juarez.gpkg", layer="manzana").to_crs(UTM)
    blocks = blocks[["CVEGEO", "geometry"]].merge(m[["CVEGEO"]].reset_index(), on="CVEGEO").set_index("index")
    m, f = m.loc[blocks.index], f.loc[blocks.index]
    cen = gpd.GeoDataFrame(geometry=blocks.centroid, crs=UTM)
    lonlat = cen.to_crs(4326)
    f["lon"], f["lat"] = lonlat.geometry.x, lonlat.geometry.y

    f["area_ha"] = blocks.area / 1e4
    f["dwellings_per_ha"] = m.VIVTOT / f.area_ha

    growth = imip("crecimiento_historico")[["CRECIMI", "geometry"]]
    j = nearest(cen, growth, max_distance=500)
    f["year_urbanized"] = growth.CRECIMI.reset_index(drop=True).reindex(j.pos).to_numpy()

    centro = gpd.GeoSeries.from_xy([CENTRO[0]], [CENTRO[1]], crs=4326).to_crs(UTM).iloc[0]
    f["km_center"] = cen.distance(centro) / 1000
    for layer, col in [("maquilas_2022", "km_maquila"), ("estaciones_brt", "km_brt"),
                       ("unidades_medicas", "km_health"), ("primarias", "km_school")]:
        f[col] = nearest(cen, imip(layer)[["geometry"]]).d / 1000

    for layer, radius, col in [("maquilas_2022", 3000, "maquilas_3km"), ("delitos_2020", 500, "crimes_500m")]:
        buf = gpd.GeoDataFrame(geometry=cen.buffer(radius).rename_axis(None), crs=UTM)
        hits = gpd.sjoin(imip(layer)[["geometry"]], buf, predicate="within")
        f[col] = hits.groupby("index_right").size().reindex(f.index).fillna(0)
    # rank within the year, so 2025/2026 reports can be fed later without the reporting volume shift
    f["crimes_500m_pct"] = f.crimes_500m.rank(pct=True)

    value = imip("valor_catastral_zonas_2025")[["VALORCATA", "geometry"]]
    j = gpd.sjoin(cen, value, how="left", predicate="within")
    f["property_value"] = pd.to_numeric(j.groupby(level=0).VALORCATA.first(), errors="coerce")
    return m, f


def ageb_features():
    """One row per AGEB (neighborhood) for notebook 01, with centroid lon/lat so it can map without geopandas."""
    c = pd.read_csv(RAW / "juarez_censo2020_ageb_manzana.csv", dtype=str)
    c = c[c.NOM_LOC == "Total AGEB urbana"].copy()
    for k in ["POBTOT", "TVIVPAR", "VIVPAR_DES", "GRAPROES"]:
        c[k] = pd.to_numeric(c[k], errors="coerce")
    c["CVEGEO"] = c.ENTIDAD + c.MUN + c.LOC + c.AGEB

    a = gpd.read_file(RAW / "inegi_marco_geo_2020_juarez.gpkg", layer="ageb").to_crs(UTM)
    a = a.merge(c[["CVEGEO", "POBTOT", "TVIVPAR", "VIVPAR_DES", "GRAPROES"]], on="CVEGEO")
    a = a[a.TVIVPAR >= 50].reset_index(drop=True)
    a["pct_vacant"] = a.VIVPAR_DES / a.TVIVPAR * 100
    a["area_km2"] = a.area / 1e6
    cen = gpd.GeoDataFrame(a[["CVEGEO"]], geometry=a.centroid, crs=UTM)
    lonlat = cen.to_crs(4326)
    a["lon"], a["lat"] = lonlat.geometry.x, lonlat.geometry.y

    def at_centroid(layer, col):
        j = gpd.sjoin(cen, layer[[col, "geometry"]], how="left", predicate="within")
        return j.groupby(level=0)[col].first()

    a["year_urbanized"] = at_centroid(imip("crecimiento_historico"), "CRECIMI")
    a["property_value"] = pd.to_numeric(at_centroid(imip("valor_catastral_zonas_2025"), "VALORCATA"), errors="coerce")
    maq = imip("maquilas_2022")
    a["km_maquila"] = cen.geometry.apply(lambda p: maq.distance(p).min() / 1000)
    a["maquilas_3km"] = cen.geometry.apply(lambda p: (maq.distance(p) < 3000).sum())
    for layer, col in [("estaciones_brt", "km_brt"), ("unidades_medicas", "km_health")]:
        pts = imip(layer)
        a[col] = cen.geometry.apply(lambda p: pts.distance(p).min() / 1000)
    centro = gpd.GeoSeries.from_xy([CENTRO[0]], [CENTRO[1]], crs=4326).to_crs(UTM).iloc[0]
    a["km_center"] = cen.distance(centro) / 1000
    for y in [2020, 2025]:
        n = gpd.sjoin(imip(f"delitos_{y}")[["geometry"]], a[["CVEGEO", "geometry"]], predicate="within").groupby("CVEGEO").size()
        a[f"crimes{y}_per1000"] = a.CVEGEO.map(n).fillna(0) / a.POBTOT * 1000
    return pd.DataFrame(a.drop(columns="geometry"))


def crimes_by_ageb(years=(2020, 2024, 2025, 2026)):
    """Every IMIP crime record tagged with the AGEB it falls in, for notebook 04 (no geopandas there)."""
    agebs = gpd.read_file(RAW / "inegi_marco_geo_2020_juarez.gpkg", layer="ageb").to_crs(UTM)[["CVEGEO", "geometry"]]
    out = []
    for y in years:
        d = imip(f"delitos_{y}")
        j = gpd.sjoin(d, agebs, how="left", predicate="within")
        ll = j.to_crs(4326)
        out.append(pd.DataFrame({
            "year": y, "month": pd.to_numeric(j.MES_NUMERO, errors="coerce"), "day": pd.to_numeric(j.DIA_NUMERO, errors="coerce"),
            "hour": j.HORA, "category": j.ID_GRUPO, "colonia": j.COLONIA, "ageb": j.CVEGEO,
            "lon": ll.geometry.x.round(6), "lat": ll.geometry.y.round(6),
        }))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    m = load_blocks()
    m, f = spatial_features(m, census_features(m))
    out = pd.concat([m[["CVEGEO", "AGEB_KEY", "TVIVPAR", "vacant_masked", "target_pct_vacant"]], f], axis=1)
    path = ROOT / "data" / "features.csv"
    out.to_csv(path, index=False)
    print(f"{path}: {len(out)} blocks, {f.shape[1]} features")

    a = ageb_features()
    path = ROOT / "data" / "ageb_features.csv"
    a.to_csv(path, index=False)
    print(f"{path}: {len(a)} AGEBs")

    cr = crimes_by_ageb()
    path = ROOT / "data" / "crimes_by_ageb.csv"
    cr.to_csv(path, index=False)
    print(f"{path}: {len(cr)} crime records, {cr.ageb.isna().mean():.1%} outside urban AGEBs")
