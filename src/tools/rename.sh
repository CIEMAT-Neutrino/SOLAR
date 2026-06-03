#!/bin/bash
# Copy all files in given path to another but rename them to change "shielded" to "nominal"
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    exit 1
fi
SOURCE_DIR="$1"
DEST_DIR="$2"
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi
mkdir -p "$DEST_DIR"
for file in "$SOURCE_DIR"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        new_filename="${filename//shielded/nominal}"
        cp "$file" "$DEST_DIR/$new_filename"
    fi
done
echo "Files copied and renamed from $SOURCE_DIR to $DEST_DIR"