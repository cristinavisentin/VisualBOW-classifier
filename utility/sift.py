import cv2

def extract_sift_descriptors(images, n_sift_features=500):
    sift = cv2.SIFT_create(n_sift_features)
    descriptor_list = []
    for item in images:
        label = item['label']
        kp, des = sift.detectAndCompute(item['image'], None)
        if des is not None:
            descriptor_list.append({
                'label': label,
                'sift descriptor': des,
                'keypoints': kp
            })
    return descriptor_list

def extract_single_image_sift_descriptor(image, n_sift_features=500):
    sift = cv2.SIFT_create(n_sift_features)
    __, descriptors = sift.detectAndCompute(image, None)
    return descriptors
