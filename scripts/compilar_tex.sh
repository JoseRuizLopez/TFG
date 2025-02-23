#!/bin/bash

# Obtener la ruta absoluta del directorio donde está este script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFG_DIR="$(dirname "$SCRIPT_DIR")"

# Directorios dentro del proyecto
DOCS_DIR="$TFG_DIR/docs"
OUT_DIR="$DOCS_DIR/out"

# Nombre del archivo sin extensión
ARCHIVO="proyecto"
TEX_FILE="$DOCS_DIR/$ARCHIVO.tex"

# Crear directorio de salida si no existe
mkdir -p "$OUT_DIR"

# Moverse al directorio de documentos
cd "$DOCS_DIR" || exit

# Compilar el archivo .tex por primera vez
pdflatex -output-directory="$OUT_DIR" "$TEX_FILE"

# Ejecutar BibTeX para procesar la bibliografía
cd "$OUT_DIR" || exit
bibtex "$ARCHIVO"

# Regresar al directorio de documentos y compilar dos veces más
cd "$DOCS_DIR"
pdflatex -output-directory="$OUT_DIR" "$TEX_FILE"
pdflatex -output-directory="$OUT_DIR" "$TEX_FILE"

# Limpiar archivos temporales
echo "Limpiando archivos temporales..."
rm -f "$OUT_DIR/$ARCHIVO.aux" "$OUT_DIR/$ARCHIVO.bbl" "$OUT_DIR/$ARCHIVO.blg" "$OUT_DIR/$ARCHIVO.log" "$OUT_DIR/$ARCHIVO.out"

echo "✅ Compilación completa. El archivo PDF generado es: $OUT_DIR/$ARCHIVO.pdf"
