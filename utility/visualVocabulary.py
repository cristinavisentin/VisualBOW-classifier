import numpy as np
from sklearn.cluster import KMeans
import torch

from utility.sift import extract_sift_descriptors, extract_single_image_sift_descriptor

class VisualVocabulary:
    def __init__(self, n_clusters, n_sift_features, dim_subset):
        self.n_clusters = n_clusters
        self.dim_subset = dim_subset
        self.n_sift_features = n_sift_features
        self.vocabulary = None
        self.kmeans = None

    def create_vocabulary(self, images):
        sift_descriptors = extract_sift_descriptors(images, self.n_sift_features)
        count = 0
        for item in sift_descriptors:
            for i in item['sift descriptor']:
                count += 1
        self.vocabulary = self._kmeans(sift_descriptors)
        quantized_train_images = self.quantize_images(sift_descriptors)
        UNC_train_images = self.UNC_images(sift_descriptors)
        pyramid_train_histogram = self.compute_pyramid_descriptor(images, 2)
        return self.vocabulary, quantized_train_images, UNC_train_images, pyramid_train_histogram
    
    def _kmeans(self, descriptors):
        vector_descriptors = self._get_descriptor_vectors(descriptors)
        self.kmeans = KMeans(n_clusters = self.n_clusters, n_init = 10, random_state=42)
        self.kmeans.fit(vector_descriptors)
        return self.kmeans.cluster_centers_

    def quantize_images(self, descriptors):
        quantized_images = []
        for item in descriptors:
            label = item['label']
            des = item['sift descriptor']
            visual_words_indexes = self.kmeans.predict(des)   
            quantized_images.append({
                'label': label,
                'vw': visual_words_indexes
            })
        return quantized_images
    
    def _get_descriptor_vectors(self, descriptors):
        all_descriptors = [item['sift descriptor'] for item in descriptors]
        descriptor_vector = np.vstack(all_descriptors)
        if self.dim_subset is not None and self.dim_subset < descriptor_vector.shape[0]:
            random_indices = np.random.choice(descriptor_vector.shape[0], self.dim_subset, replace=False)
            descriptor_vector = descriptor_vector[random_indices]
        return descriptor_vector
    
    def UNC_images(self, descriptors):
        UNC_images = []
        cluster_centers_gpu = torch.tensor(self.kmeans.cluster_centers_, device='cuda', dtype=torch.float32)
        for item in descriptors:
            label = item['label']
            descriptor_gpu = torch.tensor(item['sift descriptor'], device='cuda', dtype=torch.float32)
            distances = torch.cdist(descriptor_gpu, cluster_centers_gpu)
            gaussian_scores = self.gaussian_kernel(distances, 150)
            den = torch.sum(gaussian_scores, dim=1, keepdim=True)
            UNC = torch.sum(gaussian_scores / den, dim=0) / 128
            UNC_images.append({
                'label': label,
                'vw': UNC.cpu().tolist()
            })
        return UNC_images

    def gaussian_kernel(self, x, sigma):
        return (1 / (torch.sqrt(torch.tensor(2 * torch.pi, device='cuda')) * sigma)) * torch.exp(-0.5 * (x / sigma) ** 2)

    def compute_pyramid_descriptor(self, images, levels):
        normalized_descriptors = []
        for item in images:
            image = item['image']
            descriptors = []
            h, w = image.shape[:2]
            for level in range(levels + 1):
                num_cells = 2**level
                cell_h, cell_w = h // num_cells, w // num_cells
                for i in range(num_cells):
                    for j in range(num_cells):
                        y_start, y_end = i * cell_h, (i + 1) * cell_h
                        x_start, x_end = j * cell_w, (j + 1) * cell_w
                        cell = image[y_start:y_end, x_start:x_end]
                        features = extract_single_image_sift_descriptor(cell, self.n_sift_features)
                        quantized_features = self.assign_to_visual_words(features)  
                        histogram, _ = np.histogram(quantized_features, bins=self.n_clusters, range=(0, self.n_sift_features))
                        descriptors.append(histogram)
            concatenated_descriptor = np.hstack(descriptors)
            normalized_descriptor_his = concatenated_descriptor / np.linalg.norm(concatenated_descriptor)
            normalized_descriptors.append({
                'label': item['label'],
                'histogram': normalized_descriptor_his
            })
        return normalized_descriptors
    
    def assign_to_visual_words(self, features):
        if features is None or len(features) == 0:
            return np.array([])
        return self.kmeans.predict(features)