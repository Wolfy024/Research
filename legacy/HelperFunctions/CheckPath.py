def check_path(folder):
    """
    Checks if the path is a directory and if we have permission to
    read and access the files. It returns the list of files.
    :param folder: The path to the folder
    :return: List of files
    """

    import os
    try:
        files = os.listdir(folder)

    except FileNotFoundError:
        print(f"Path '{folder}' does not exist.")
        return

    except NotADirectoryError:
        print(f"'{folder}' is not a directory.")
        return

    except PermissionError:
        print(f"Permission denied for accessing '{folder}'.")
        return

    if not files:
        print("No files found.")
        return

    return files
