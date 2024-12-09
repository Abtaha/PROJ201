import os
import shutil

# Source and destination directories
source_dir = "data"
destination_dir = "events 2"

# Create destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.mkdir(destination_dir)

# Initialize a dictionary to hold files categorized by ID
file_groups = {}

# Categorize files by common ID
for file in os.listdir(source_dir):
    if file.endswith(".csv"):
        parts = file.split("_")
        if len(parts) > 1:  # Ensure there is an ID part
            identifier = parts[1]
            if identifier not in file_groups:
                file_groups[identifier] = []
            file_groups[identifier].append(os.path.join(source_dir, file))


# Copy files with the same ID to the destination folder with new names
for i, (identifier, files) in enumerate(file_groups.items()):
    for j, file in enumerate(files):
        # Create new filenames based on the group index (e.g., 1.csv, 1s.csv)
        suffix = "s" if "sb" in file else ""
        new_filename = f"{i + 1}{suffix}.csv"
        destination_path = os.path.join(destination_dir, new_filename)
        print(new_filename, file)

        # Copy the file to the destination folder
        shutil.copy(file, destination_path)
