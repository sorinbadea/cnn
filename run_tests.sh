folder_path="${1:-.}"
if [ $# -lt 1 ]; then
    echo "Error: At least one argument is required"
    echo "Usage: $0 <folder_path for test images>"
    exit 1
fi
if [ ! -d "$folder_path" ]; then
    echo "Error: '$folder_path' is not a valid directory."
    exit 1
fi
for file in "$folder_path"/*; do
    if [ -f "$file" ]; then
        echo "Processing: $file"
        python main.py -a "$file"
    fi
done
