#!/usr/bin/env bash

# Uso: ./merge_summaries.sh /ruta/al/directorio output.csv

INPUT_DIR="$1"
OUTPUT_FILE="$2"

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_FILE" ]]; then
    echo "Uso: $0 <directorio> <archivo_salida.csv>"
    exit 1
fi

# Vaciar archivo de salida si existe
> "$OUTPUT_FILE"

# Variable para controlar si ya escribimos la cabecera
HEADER_WRITTEN=false

# Recorrer subdirectorios
for d in "$INPUT_DIR"/*/ ; do
    SUMMARY_FILE="${d}summary.csv"

    if [[ -f "$SUMMARY_FILE" ]]; then
        echo "Procesando: $SUMMARY_FILE"

        # Leer la cabecera si aún no se ha escrito
        if [[ "$HEADER_WRITTEN" = false ]]; then
            head -n 1 "$SUMMARY_FILE" >> "$OUTPUT_FILE"
            HEADER_WRITTEN=true
        fi

        # Extraer la segunda línea (fila de datos)
        sed -n '2p' "$SUMMARY_FILE" >> "$OUTPUT_FILE"
    else
        echo "⚠️  No existe summary.csv en $d"
    fi
done

echo "✔️ Completado. Archivo generado: $OUTPUT_FILE"