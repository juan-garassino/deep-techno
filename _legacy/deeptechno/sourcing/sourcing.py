import tensorflow as tf
import os
import csv
import pandas as pd
import zipfile
import tensorflow as tf

from deepTechno.manager.manager import Manager

def download_maestro_dataset(data_dir):
    """
    Download the MAESTRO dataset if it doesn't exist locally.

    Args:
    - data_dir (str): Directory to save the dataset.

    Returns:
    - str: Path to the downloaded dataset.
    """
    # Check if the data directory exists
    if not os.path.exists(data_dir):
        # Download and extract the dataset
        data_dir = tf.keras.utils.get_file(
            'maestro-v2.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
            extract=True,
            cache_dir=data_dir,
            cache_subdir='maestro-v2.0.0'
        )
        return data_dir
    else:
        print("Dataset directory exists in", data_dir)
        # dataset_folder = os.path.join(data_dir, "maestro-v2.0.0")
        if os.path.exists(data_dir):
            # Check if the dataset folder is empty
            if not os.listdir(data_dir):
                print("Dataset directory is empty. Downloading the dataset...")
                # Download and extract the dataset
                dataset_zip_path = tf.keras.utils.get_file(
                    'maestro-v2.0.0-midi.zip',
                    origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
                    extract=True,
                    cache_dir=data_dir,
                    cache_subdir='maestro-v2.0.0'
                )
                # Unzip the downloaded dataset
                with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                return data_dir
            else:
                return data_dir
        else:
            print("Error: Dataset folder not found in the specified directory:", data_dir)
            return None

def find_midi_files(dataset_folder, csv_filename):
    """
    Find all MIDI files within subfolders of the dataset folder and save their directories to a CSV file.

    Args:
    - dataset_folder (str): Path to the dataset folder.
    - csv_filename (str): Name of the CSV file to save the directories.

    Returns:
    - list: List of directories of the found MIDI files.
    """
    # List to store the directories of MIDI files
    midi_files = []

    # Recursively search for MIDI files within subfolders
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(".midi") or file.endswith(".mid"):
                midi_path = os.path.join(root, file)
                midi_files.append(midi_path)

    # Construct full path to the CSV file
    csv_file = os.path.join(dataset_folder, csv_filename)

    # Check if MIDI files are found before creating the CSV file
    if midi_files:
        # Save the directories to a CSV file
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["midi_file"])  # Write header
            for midi_file in midi_files:
                writer.writerow([midi_file])
    else:
        print("No MIDI files found in the dataset folder.")

    return midi_files, csv_file

def split_csv(csv_file, train_csv, val_csv, test_csv, train_p=0.6, val_p=0.2, test_p=0.2, data_dir="."):
    """
    Split the CSV file into three separate CSV files for train, validation, and test sets.

    Args:
    - csv_file (str): Path to the input CSV file.
    - train_csv (str): Name of the CSV file for the train set.
    - val_csv (str): Name of the CSV file for the validation set.
    - test_csv (str): Name of the CSV file for the test set.
    - train_p (float): Proportion of data to include in the train set.
    - val_p (float): Proportion of data to include in the validation set.
    - test_p (float): Proportion of data to include in the test set.
    - data_dir (str): Directory where the CSV files will be saved.
    """
    # Read the original CSV file
    csv_file = os.path.join(data_dir, csv_file)
    df = pd.read_csv(csv_file)

    # Shuffle the rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate the number of samples for each split
    num_samples = len(df)
    num_train = int(num_samples * train_p)
    num_val = int(num_samples * val_p)
    num_test = num_samples - num_train - num_val

    # Split the data into train, validation, and test sets
    train_set = df.iloc[:num_train]
    val_set = df.iloc[num_train:num_train + num_val]
    test_set = df.iloc[num_train + num_val:]

    # Construct full paths for the CSV files
    train_csv = os.path.join(data_dir, train_csv)
    val_csv = os.path.join(data_dir, val_csv)
    test_csv = os.path.join(data_dir, test_csv)

    # Save the split datasets to CSV files
    train_set.to_csv(train_csv, index=False)
    val_set.to_csv(val_csv, index=False)
    test_set.to_csv(test_csv, index=False)

    print("Split data into train, validation, and test sets successfully.")

def create_sequences(dataset: tf.data.Dataset, seq_length: int, vocab_size = 128, key_order = []):
  """Returns TF Dataset of sequence and label examples."""
  seq_length = seq_length + 1

  # Take 1 extra for the labels
  windows = dataset.window(seq_length, shift=1, stride=1,
                              drop_remainder=True)

  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)

  # Normalize note pitch
  def scale_pitch(x):
    x = x/[vocab_size,1.0,1.0]
    return x

  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

    return scale_pitch(inputs), labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

if __name__ == "__main__":
    
    args = Manager().parse_args()
    
    data_dir = os.path.join(os.environ.get('HOME'), args.root + args.data_dir)
    
    temps = download_maestro_dataset(data_dir)
    
    print("Dataset path:", data_dir)

    midi_files, csv_file = find_midi_files(data_dir, "my-maestro.csv")

    print("First few rows of the CSV file:")
    
    split_csv("my-maestro.csv", "train-maestro.csv", "val-maestro.csv", "test-maestro.csv", train_p=0.6, val_p=0.2, test_p=0.2, data_dir=data_dir)

    Manager.play_midi(midi_files[1])
    # print("first",  midi_files[1])