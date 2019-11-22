import os

from PIL import Image


def convert_to_greyscale(path, save_path):
    img = Image.open(path).convert('LA')
    img.save(save_path)


def convert_folder_images_to_greyscale(path, op_path):
    for index, image_path in enumerate(os.listdir(path)):
        # create the full input path and read the file
        input_path = os.path.join(path, image_path)
        if ".DS_Store" in input_path:
            continue
        output_path = os.path.join(op_path, str(index)+".png")
        convert_to_greyscale(input_path, output_path)
