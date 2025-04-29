import os

def print_directory_tree(startpath, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []

    for root, dirs, files in os.walk(startpath, topdown=True):
        # Modify dirs in-place to exclude specified directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        # Calculate the depth of the current directory
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')

        # Print the file names
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

# Example usage:
# Replace '.' with the path to your project directory if needed
# Specify the directories you want to exclude in the exclude_dirs list
print_directory_tree('.', exclude_dirs=['train', 'test', "images", "depricated", "DelftBikes"])
