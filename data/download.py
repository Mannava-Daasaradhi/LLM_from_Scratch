import os
import urllib.request

def prepare_data():
    # Setup our target URLs and cross-platform paths
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_dir = "data"
    input_path = os.path.join(data_dir, "input.txt")
    train_path = os.path.join(data_dir, "train.txt")
    val_path = os.path.join(data_dir, "val.txt")

    # Step 1 & 2: Safe Folder Creation & Downloading
    os.makedirs(data_dir, exist_ok=True) 

    if not os.path.exists(input_path):
        print("Downloading file...")
        # Using the built-in library instead of wget/subprocess
        urllib.request.urlretrieve(url, input_path)
        print("Download complete.")
    else:
        print("File already exists. Skipping download.")

    # Step 3: Reading the Content
    # Explicitly using utf-8 ensures Windows and Mac/Linux count characters identically
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Step 4: Calculating the Split Index
    # Multiply by 0.9 and convert to an integer for a clean slice
    split_index = int(len(text) * 0.9)

    # Step 5: Slicing the Data
    train_data = text[:split_index]
    val_data = text[split_index:]

    # Step 6: Saving the New Files
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(train_data)
        
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write(val_data)

    # Step 7: Formatted Output
    # The :, inside the curly braces automatically adds the comma separators
    print(f"Train: {len(train_data):,} chars | Val: {len(val_data):,} chars")

if __name__ == "__main__":
    prepare_data()