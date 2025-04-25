import pandas as pd

# Ruta del dataset original (ajusta si es necesario)
ruta_original = "C:/Users/10Pearls/Documents/Eda App/jena_climate_2009_2016.csv"

# Nombre del archivo de salida
ruta_muestra = "sample_data.csv"

# Número de filas para la muestra
n_filas = 10000

# Cargar dataset completo
try:
    df = pd.read_csv(ruta_original)
    print(f"✅ Dataset original cargado correctamente. Total de filas: {len(df)}")
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo en la ruta especificada.")
    exit()

# Crear muestra
df_sample = df.head(n_filas)

# Guardar muestra
df_sample.to_csv(ruta_muestra, index=False)
print(f"🎉 Muestra de {n_filas} filas creada exitosamente como '{ruta_muestra}'")
