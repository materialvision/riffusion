import os
import re
import argparse
import shutil
from collections import defaultdict

def rename_files_and_generate_individual_texts(input_folder, new_base_name):
    # List and sort all files in the directory
    files = sorted(os.listdir(input_folder))
    
    # Initialize the counter
    counter = 0
    unique_names = set()

    for filename in files:
        if os.path.isfile(os.path.join(input_folder, filename)):  # Ensure it's a file
            base, extension = os.path.splitext(filename)
            new_name_part = base.split('-')[0]  # Get the part before the first underscore

            # Create new file name with counter
            new_filename = f"{new_base_name}-({counter}){extension}"
            
            # Rename the file
            old_path = os.path.join(input_folder, filename)
            new_path = os.path.join(input_folder, new_filename)
            os.rename(old_path, new_path)
            
            # Create a text file containing the new name part
            text_filename = os.path.join(input_folder, f"{new_base_name}-({counter}).txt")
            with open(text_filename, "w") as txt_file:
                txt_file.write(new_name_part + '\n')
            
            # Add the name to the set if it hasn't been added yet
            unique_names.add(new_name_part)

            counter += 1
    # Create a consolidated text file with all unique name parts
    consolidated_text_filename = os.path.join(input_folder, f"{new_base_name}_all_unique_names.txt")
    with open(consolidated_text_filename, "w") as consolidated_txt_file:
        for name in unique_names:
            consolidated_txt_file.write(name + '\n')

def sanitize_filename(name):
    # Convert to lowercase and replace spaces and special characters with underscores
    name = re.sub(r'[^\w]', '_', name.lower())
    return name

def copy_and_rename_files(input_folder, output_folder, custom_word=None):
    # Ensure that the input folder exists
    if not os.path.isdir(input_folder):
        print(f"The directory '{input_folder}' does not exist.")
        return
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to track occurrences of each filename prefix
    prefix_count = defaultdict(int)

    # Iterate over each file in the folder
    for filename in os.listdir(input_folder):
        # Skip directories
        if os.path.isdir(os.path.join(input_folder, filename)):
            continue
        
        # Split the filename and extension
        name, extension = os.path.splitext(filename)
        
        # Sanitize the file name to remove spaces and special characters
        sanitized_name = sanitize_filename(name)

        # Find the index of the first underscore
        underscore_index = sanitized_name.find('_')
        
        if underscore_index != -1:
            # Extract the prefix (before the first underscore)
            prefix = sanitized_name[:underscore_index]

            # Create the new name based on the presence of custom_word
            if custom_word:
                new_name = f"{prefix}_{custom_word}-({prefix_count[prefix]}){extension}"
            else:
                new_name = f"{prefix}-({prefix_count[prefix]}){extension}"

            # Construct full paths correctly
            old_path = os.path.join(input_folder, filename)
            new_path = os.path.join(output_folder, new_name)

            # Copy the file to the output directory
            shutil.copy2(old_path, new_path)
            print(f"Copied '{filename}' to '{new_name}' in '{output_folder}'")

            # Increment the count for this prefix
            prefix_count[prefix] += 1
        else:
            print(f"No underscore found in '{filename}', skipping...")

#make a def that will make two folders named captions and instance_images and move all the .txt files into captions and all the .png files into instance_images.
def move_files_to_folders(input_folder, output_folder):
    captions_folder = os.path.join(output_folder, 'captions')
    instance_images_folder = os.path.join(output_folder, 'instance_images')
    os.makedirs(captions_folder, exist_ok=True)
    os.makedirs(instance_images_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            shutil.move(os.path.join(input_folder, filename), os.path.join(captions_folder, filename))
        elif filename.endswith('.png'):
            shutil.move(os.path.join(input_folder, filename), os.path.join(instance_images_folder, filename))

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Copy and rename files in a folder by sanitizing names and optionally replacing text after the first underscore with a custom word, '
                                                 'appending incremental numbers for duplicates, and saving them to an output folder.')
    parser.add_argument('folder', type=str, help='Path to the input folder containing files to rename.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder where renamed files will be saved.')
    parser.add_argument('word', type=str, nargs='?', help='Word that will be the new filename')

    # Parse arguments
    args = parser.parse_args()

    # Call the renaming function with provided arguments
    copy_and_rename_files(args.folder, args.output_folder)

    rename_files_and_generate_individual_texts(args.output_folder, args.word)

    move_files_to_folders(args.output_folder, args.output_folder)
if __name__ == "__main__":
    main()


    