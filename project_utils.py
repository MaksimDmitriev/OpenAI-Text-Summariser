import os


def read_file(file_path):
    """Reads a file and returns its contents as a string."""
    # Check if file exists (optional)
    print(file_path)
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None

    with open(file_path, 'r') as file:
        content = file.read()
    return content
