def remove_extra_files(folder_1: str, folder_2: str = None, extensions: list = None) -> None:
    """
    Remove files that are not common between two folders or match the specified extensions.

    :param folder_1: The path to the first folder.
    :param folder_2: The path to the second folder (optional).
    :param extensions: A list of file extensions to consider for removal ['.jpg', '.png'] (optional).
    :return: None
    """
    import os
    from HelperFunctions import CheckPath

    files_1 = CheckPath.check_path(folder_1)

    if folder_2:
        files_2 = CheckPath.check_path(folder_2)
        files_to_remove = set(files_1) - set(files_2)
    else:
        files_to_remove = set(files_1)
    if extensions:
        files_to_remove = [f for f in files_to_remove if f.endswith(extensions)]
    for file in files_to_remove:
        file_path = os.path.join(folder_1, file)
        try:
            os.remove(file_path)
            print(f"File {file_path} removed")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")