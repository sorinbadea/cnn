folder_path="${1:-.}" 
for file in "$folder_path"/*; do
    if [ -f "$file" ]; then
        echo "Processing: $file"
        python main.py -a "$file"
    fi
done
