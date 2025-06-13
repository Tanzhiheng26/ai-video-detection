import os
import random
import shutil

def move_n_random_files(src_folder, dst_folder, n):
    # Get all files in the source folder
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    
    if n > len(files):
        raise ValueError(f"Not enough files in source folder: requested {n}, but only {len(files)} available.")
    
    # Choose n files randomly
    selected_files = random.sample(files, n)
    
    # Make sure destination folder exists
    os.makedirs(dst_folder, exist_ok=True)

    # Move files
    for f in selected_files:
        src_path = os.path.join(src_folder, f)
        dst_path = os.path.join(dst_folder, f)
        shutil.move(src_path, dst_path)
        print(f"Moved: {f}")
