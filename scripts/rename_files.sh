#!/bin/bash
# This script renames all files in the provided path (with flag -p) that contain the string inputted by the user with flag -rm and replaces it with the string inputted by the user with flag -w

# If flag is -h or --help, print the help message
if [[ $1 == "-h" || $1 == "--help" ]]; then
    echo "Usage: ./rename_files.sh -p <path> -r <string_to_remove> -w <string_to_replace>"
    exit 0
fi

# Parse the command-line arguments
while getopts ":p:r:w:" opt; do
    case $opt in
        p) path="$OPTARG";;
        r) remove="$OPTARG";;
        w) with="$OPTARG";;
        \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
        :) echo "Option -$OPTARG requires an argument." >&2; exit 1;;
    esac
done

# Print a summary of the provided arguments
echo "Summary of arguments:"
echo "Path: $path"
echo "Remove: $remove"
echo "With: $with"

# If path flag is not provided, apply to current directory
if [[ -z $path ]]; then
    path="."
fi

final_exit_code=0
# Loop over all files in the provided path
for file_path in $path*; do
    # Check if the file is named with the string to remove (only the filename, not any part of the path)
    file=$(basename "$file_path")
    if [[ $file == *"$remove"* ]]; then
        echo "File to rename: $file"
        # Replace the string with the string to replace
        new_file=$(echo "$file" | sed "s/$remove/$with/")
        # Rename the file
        mv "$path$file" "$path$new_file"
        # Catch the exit code of mv
        exit_code=$?
        # If the exit code is not 0, print an error message
        if [[ $exit_code -ne 0 ]]; then
            final_exit_code=$exit_code
            echo "Error renaming file $file"
        fi
    fi
done

# If the final exit code is 0, print a success message
if [[ $final_exit_code -eq 0 ]]; then
    echo "All files renamed successfully"
fi

# Exit with the final exit code
exit $final_exit_code
