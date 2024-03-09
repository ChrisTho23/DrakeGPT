import kaggle
import pandas as pd
import os
import shutil

from config import DATA_DIR, DATA

def load_dataset():
    print("Downloading data from Kaggle...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset='deepshah16/song-lyrics-dataset', 
        path=DATA_DIR,
        unzip=True,
    )
    print("Data downloaded successfully.")

    csv_file_path = os.path.join(DATA_DIR, 'csv', 'Drake.csv')
    data = pd.read_csv(csv_file_path)

    # Extract the 'Lyric' column and concatenate all lyrics
    lyrics = data['Lyric'].str.cat(sep='\n')

    print("Length of dataset in characters: ", len(lyrics))
    print(f"First 200 character of the text:\n{lyrics[:200]}")

    # Save the lyrics to a text file
    txt_file_path = DATA["input"]
    with open(txt_file_path, 'w', encoding='utf-8') as file:
        file.write(lyrics)

    print(f"Lyrics saved to {txt_file_path}")

    # Function to safely remove a directory
    def remove_directory(directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Directory '{directory}' has been removed.")
        else:
            print(f"Directory '{directory}' does not exist or has already been removed.")

    # Remove the 'csv' and 'json files' directories
    csv_dir_path = os.path.join(DATA_DIR, 'csv')
    json_dir_path = os.path.join(DATA_DIR, 'json files')

    remove_directory(csv_dir_path)
    remove_directory(json_dir_path)

    print("Finished data import...")



