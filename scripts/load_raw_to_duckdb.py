"""
Sprint 0 — Carga de datos crudos a DuckDB
==========================================
Diseñado específicamente para Tabla_Hist_Asesores:
  - Dos archivos CSV/TXT con misma estructura, períodos distintos
  - Sin encabezado en los archivos fuente
  - Encoding ISO-8859-1 (latin-1)
  - Separador: coma
  - Fechas como enteros YYYYMMDD → se convierten a DATE en DuckDB

Uso (desde la raíz del proyecto):
    uv run python scripts/load_raw_to_duckdb.py

Resultado:
    data/advisor_risk.duckdb
    └── tabla_hist_asesores  (ambos archivos concatenados, ordenados por HOY)
"""

import sys
import duckdb
import pandas as pd
from pathlib import Path
from loguru import logger

# ── Configuración ─────────────────────────────────────────────────────────────
RAW_DIR    = Path("data/raw")
DB_PATH    = Path("data/advisor_risk.duckdb")
TABLE_NAME = "tabla_hist_asesores"

# Nombres de columna en el orden exacto del CREATE TABLE original
# (los archivos fuente NO tienen encabezado)
COLUMN_NAMES = [
    "HOY", "REGION", "CODIGO_SUCURSAL", "SUCURSAL", "COORDINADOR",
    "CODIGO_ASESOR", "ASESOR", "DIA_PAGO", "CICLO", "RANGO_CICLO",
    "GRUPOS", "CLIENTES", "PRESTAMO", "CLIENTES_PRESTAMO",
    "CARTERA_HOY", "CARTERA_SEMANT", "INCREMENTO_CARTERA", "ATRASO",
    "SEMANA_ACTUAL", "SEMANA_ACTUAL_RANGO", "TASA_PROM",
    "PROVISION_HOY", "PROVISION_SEMANT", "GASTO_PROVISION_SEMANAL",
    "PASE1", "PASE15", "PASE30",
    "CLIENTES_NUEVOS", "DESEMBOLSO_CLIENTES_NUEVOS",
    "DIAS_MORA", "DIAS_MORA_RANGO",
    "NUMERO_EMPLEADO", "NOMBRE", "PUESTO", "PRODUCTO",
    "N_SUCURSAL", "AREA", "FECHA_ALTA", "FECHA_BAJA",
    "TIPO_BAJA", "MOTIVO_BAJA", "FECHA_NACIMIENTO", "EDAD",
    "SEXO", "ESTADO_CIVIL", "EXP_MICROFINANZAS", "NIVEL_ESTUDIOS",
]

# Columnas con fechas en formato YYYYMMDD (entero → DATE)
DATE_COLUMNS = ["HOY", "FECHA_ALTA", "FECHA_BAJA", "FECHA_NACIMIENTO"]

# Columnas que son leakage confirmado — se cargan pero se marcan en el log
LEAKAGE_COLUMNS = ["PASE1", "PASE15", "PASE30",
                   "DIAS_MORA_RANGO", "SEMANA_ACTUAL_RANGO"]


# ── Funciones ─────────────────────────────────────────────────────────────────

def find_source_files() -> list[Path]:
    """Encuentra archivos .csv y .txt en data/raw/, ignora temporales de Excel."""
    files = [
        f for f in sorted(RAW_DIR.iterdir())
        if f.suffix.lower() in (".csv", ".txt")
        and not f.name.startswith("~")
        and f.stat().st_size > 0
    ]
    return files


def load_file(filepath: Path) -> pd.DataFrame:
    """
    Carga un archivo fuente con los parámetros conocidos:
      - Sin encabezado
      - Encoding latin-1
      - Separador coma
      - Nombres de columna fijos del DDL original
    """
    df = pd.read_csv(
        filepath,
        header=None,
        names=COLUMN_NAMES,
        encoding="latin-1",
        sep=",",
        low_memory=False,
        dtype=str,          # Todo como string primero; los tipos se aplican después
    )
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica los tipos correctos a cada columna.
    Las fechas YYYYMMDD se convierten a datetime (DuckDB las almacenará como DATE).
    """
    # Enteros
    int_cols = [
        "CODIGO_SUCURSAL", "CODIGO_ASESOR", "DIA_PAGO", "CICLO",
        "GRUPOS", "CLIENTES", "PRESTAMO", "CLIENTES_PRESTAMO",
        "SEMANA_ACTUAL", "PASE1", "PASE15", "PASE30",
        "CLIENTES_NUEVOS", "DESEMBOLSO_CLIENTES_NUEVOS",
        "DIAS_MORA", "NUMERO_EMPLEADO", "N_SUCURSAL", "EDAD",
    ]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Flotantes
    float_cols = [
        "CARTERA_HOY", "CARTERA_SEMANT", "INCREMENTO_CARTERA", "ATRASO",
        "TASA_PROM", "PROVISION_HOY", "PROVISION_SEMANT",
        "GASTO_PROVISION_SEMANAL",
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fechas YYYYMMDD → datetime
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")

    return df


def validate_columns(df: pd.DataFrame, filepath: Path) -> None:
    """Verifica que el archivo tenga exactamente las columnas esperadas."""
    if len(df.columns) != len(COLUMN_NAMES):
        raise ValueError(
            f"{filepath.name}: se esperaban {len(COLUMN_NAMES)} columnas, "
            f"se encontraron {len(df.columns)}."
        )


def print_summary(df: pd.DataFrame, label: str) -> None:
    """Imprime resumen de un DataFrame."""
    logger.info(f"  [{label}] Filas: {len(df):,} | "
                f"HOY mín: {df['HOY'].min().date()} | "
                f"HOY máx: {df['HOY'].max().date()}")
    n_asesores = df["CODIGO_ASESOR"].nunique()
    logger.info(f"  [{label}] Asesores únicos: {n_asesores:,}")


def check_overlap(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """Advierte si los períodos de los dos archivos se solapan."""
    min1, max1 = df1["HOY"].min(), df1["HOY"].max()
    min2, max2 = df2["HOY"].min(), df2["HOY"].max()
    overlap = not (max1 < min2 or max2 < min1)
    if overlap:
        logger.warning(
            "⚠  Los períodos de los dos archivos se SOLAPAN. "
            f"Archivo 1: {min1.date()} → {max1.date()} | "
            f"Archivo 2: {min2.date()} → {max2.date()}. "
            "Verifica que no haya filas duplicadas."
        )
    else:
        logger.info(
            f"  Períodos sin solapamiento ✓  "
            f"({min1.date()} → {max1.date()}) + "
            f"({min2.date()} → {max2.date()})"
        )


def check_duplicates(df: pd.DataFrame) -> None:
    """Detecta filas completamente duplicadas en el dataset combinado."""
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        logger.warning(f"⚠  {n_dupes:,} filas duplicadas detectadas en el dataset combinado.")
    else:
        logger.info("  Sin filas duplicadas ✓")


def print_target_distribution(df: pd.DataFrame) -> None:
    """Muestra la distribución de la variable objetivo DIAS_MORA_RANGO."""
    dist = df["DIAS_MORA_RANGO"].value_counts(dropna=False)
    total = len(df)
    logger.info("  Distribución de DIAS_MORA_RANGO (variable objetivo):")
    for val, count in dist.items():
        pct = count / total * 100
        logger.info(f"    {str(val):<25} {count:>8,} filas  ({pct:5.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )

    logger.info("=" * 65)
    logger.info("Sprint 0 — Carga de Tabla_Hist_Asesores a DuckDB")
    logger.info("=" * 65)

    # ── 1. Localizar archivos ──────────────────────────────────────────────
    if not RAW_DIR.exists():
        logger.error(f"No existe {RAW_DIR}. Créala y coloca tus archivos ahí.")
        sys.exit(1)

    files = find_source_files()

    if len(files) == 0:
        logger.error(f"No se encontraron archivos .csv o .txt en {RAW_DIR}")
        sys.exit(1)

    if len(files) > 2:
        logger.warning(f"Se encontraron {len(files)} archivos. Se usarán todos.")

    logger.info(f"Archivos encontrados: {len(files)}")
    for f in files:
        size_mb = f.stat().st_size / 1_048_576
        logger.info(f"  {f.name}  ({size_mb:.1f} MB)")

    # ── 2. Cargar y validar cada archivo ──────────────────────────────────
    dataframes = []
    for filepath in files:
        logger.info(f"Cargando: {filepath.name} ...")
        df = load_file(filepath)
        validate_columns(df, filepath)
        df = cast_types(df)
        print_summary(df, filepath.name)
        dataframes.append(df)

    # ── 3. Verificar solapamiento de períodos ─────────────────────────────
    if len(dataframes) == 2:
        check_overlap(dataframes[0], dataframes[1])

    # ── 4. Concatenar y ordenar ───────────────────────────────────────────
    logger.info("Concatenando archivos ...")
    combined = pd.concat(dataframes, ignore_index=True)
    combined = combined.sort_values("HOY").reset_index(drop=True)

    check_duplicates(combined)

    logger.info(f"Dataset combinado: {len(combined):,} filas | "
                f"{combined['CODIGO_ASESOR'].nunique():,} asesores únicos")

    # ── 5. Distribución del target ────────────────────────────────────────
    print_target_distribution(combined)

    # ── 6. Guardar en DuckDB ──────────────────────────────────────────────
    logger.info(f"Guardando en DuckDB: {DB_PATH} ...")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(DB_PATH))
    conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    conn.register("_combined", combined)
    conn.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM _combined")
    conn.unregister("_combined")

    # Verificación final
    count      = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    hoy_min    = conn.execute(f"SELECT MIN(HOY) FROM {TABLE_NAME}").fetchone()[0]
    hoy_max    = conn.execute(f"SELECT MAX(HOY) FROM {TABLE_NAME}").fetchone()[0]

    conn.close()

    logger.success("=" * 65)
    logger.success(f"Tabla '{TABLE_NAME}' creada en {DB_PATH}")
    logger.success(f"  Filas totales : {count:,}")
    logger.success(f"  Período       : {hoy_min} → {hoy_max}")
    logger.success("=" * 65)
    logger.info("Columnas de leakage cargadas pero a EXCLUIR del modelo:")
    for col in LEAKAGE_COLUMNS:
        logger.warning(f"  ⚠  {col}")
    logger.info("")
    logger.info("Para explorar en Python:")
    logger.info("  import duckdb")
    logger.info(f"  conn = duckdb.connect('{DB_PATH}')")
    logger.info(f"  conn.execute('SELECT * FROM {TABLE_NAME} LIMIT 5').df()")


if __name__ == "__main__":
    main()
