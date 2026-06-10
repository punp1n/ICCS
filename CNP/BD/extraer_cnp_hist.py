import os
import re
import shlex
import pyodbc
import pandas as pd
from pathlib import Path

# Paths
ENV_FILE = r"C:\Users\Asvaldebenitom\OneDrive - Instituto Nacional de Estadisticas\Seguridad y justicia\ICCS\CNP\BD\.env"
OUTPUT_DIR = r"C:\Users\Asvaldebenitom\OneDrive - Instituto Nacional de Estadisticas\Artículos\Trayectoria_delictual_Chile"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "CNP_hist_2.0.xlsx")

def load_dotenv_file(env_path):
    values = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row or row.startswith("#") or "=" not in row:
                continue
            key, raw = row.split("=", 1)
            key = key.strip()
            raw = raw.strip()
            if not key:
                continue
            if raw:
                parsed = shlex.split(raw)
                values[key] = parsed[0] if parsed else ""
            else:
                values[key] = ""
    return values

def get_connection(env_values):
    requested_driver = env_values.get("SQLSERVER_DRIVER", "ODBC Driver 18 for SQL Server")
    available_drivers = [d for d in pyodbc.drivers() if "SQL Server" in d]
    
    selected_driver = requested_driver
    if requested_driver not in available_drivers:
        if available_drivers:
            # Pick the highest version
            def driver_rank(name):
                m = re.search(r"ODBC Driver\s+(\d+)\s+for SQL Server", name, flags=re.IGNORECASE)
                return int(m.group(1)) if m else -1
            odbc_drivers = sorted([d for d in available_drivers if driver_rank(d) >= 0], key=driver_rank, reverse=True)
            selected_driver = odbc_drivers[0] if odbc_drivers else available_drivers[0]
        else:
            raise RuntimeError("No SQL Server ODBC drivers installed.")
            
    server = env_values["SQLSERVER_HOST"]
    instance = env_values.get("SQLSERVER_INSTANCE", "").strip()
    if instance:
        server = f"{server}\\{instance}"

    conn_str = (
        f"DRIVER={{{selected_driver}}};"
        f"SERVER={server};"
        f"DATABASE={env_values['SQLSERVER_DATABASE']};"
        f"UID={env_values['SQLSERVER_USER']};"
        f"PWD={env_values['SQLSERVER_PASSWORD']};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=30;"
    )
    return pyodbc.connect(conn_str, autocommit=True)

def main():
    print("Loading credentials...")
    env_values = load_dotenv_file(ENV_FILE)
    
    print("Connecting to SQL Server...")
    conn = get_connection(env_values)
    
    query = """
    WITH CTE AS (
        SELECT 
            cp.cum,
            cp.glosa_cum,
            cp.descripcion_delito,
            COALESCE(cp.agrupador_delito, fd.glosa_familia) AS agrupacion,
            ROW_NUMBER() OVER(PARTITION BY cp.cum ORDER BY p.anio DESC, p.ciclo DESC) as rn
        FROM cum.cnp_periodo cp
        JOIN cum.periodo p ON cp.periodo_id = p.periodo_id
        JOIN cum.familia_delito fd ON cp.familia_id = fd.familia_id
    )
    SELECT 
        cum AS CUM,
        glosa_cum AS Glosa,
        descripcion_delito AS Descripcion,
        agrupacion AS Agrupacion
    FROM CTE
    WHERE rn = 1
    ORDER BY cum;
    """
    
    print("Executing query...")
    df = pd.read_sql(query, conn)
    
    print(f"Extraction complete. {len(df)} rows retrieved.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_excel(OUTPUT_FILE, index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
