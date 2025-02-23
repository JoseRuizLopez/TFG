#!/bin/bash

# Nombre del archivo sin extensión
ARCHIVO="proyecto"

# Compilar el archivo .tex por primera vez
pdflatex "$ARCHIVO.tex"

# Ejecutar BibTeX para procesar la bibliografía
bibtex "$ARCHIVO"

# Compilar dos veces más para actualizar las referencias
pdflatex "$ARCHIVO.tex"
pdflatex "$ARCHIVO.tex"

# Limpiar archivos temporales
echo "Limpiando archivos temporales..."
rm -f "$ARCHIVO.aux" "$ARCHIVO.bbl" "$ARCHIVO.blg" "$ARCHIVO.log" "$ARCHIVO.out"

echo "✅ Compilación completa. El archivo PDF generado es: $ARCHIVO.pdf"
