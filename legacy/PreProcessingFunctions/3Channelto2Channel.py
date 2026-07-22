def convert(input_folder, output_folder):
    """
    :param input_folder: Path to folder 1 containing images.
    :param output_folder: Path to folder 2 where you want to keep the converted images.
    :return: None
    """
    from HelperFunctions import CheckPath
    import cv2
    import os

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = CheckPath.check_path(input_folder)

    for i in files:
        img = cv2.imread(input_folder + "/" + i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_folder + "/" + i, img)
        print('Image ' + i + ' converted to 2 channel')
    return
