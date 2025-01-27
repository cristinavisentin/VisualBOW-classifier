import os
import cv2

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isdir(path):
            path = folder + "/" + filename
            for cat in os.listdir(path):
                img = cv2.imread(path + "/" + cat, 0)
                if img is not None:
                    images.append({
                        'label': filename, 
                        'image': enlarge_image(resize_image(img))
                    })
    return images

def enlarge_image(image, scale=1.2):
    height, width = image.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale) 
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return image

def resize_image(image, target_size=(256, 256)):
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return image