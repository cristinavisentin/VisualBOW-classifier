import numpy as np

def create_histograms(quantized_images, visual_words):
    histograms = []
    for item in quantized_images:
        label = item['label']
        his = np.zeros(len(visual_words))
        for vw_index in item['vw']:
            his[vw_index] += 1
        histograms.append({
            'label': label,
            'histogram': his
        })
    return histograms

def calculate_idf(histograms, n_clusters):
    k = n_clusters
    N = len(histograms)
    df = [0] * k
    for item in histograms:
        for i, value in enumerate(item['histogram']):
            if value > 0:
                df[i] += 1

    df = np.array(df)
    idf = np.log(N / df)
    return idf

def tfidf_histograms(histograms, idf):
    tfidf = []
    for item in histograms:
        label = item['label']
        histogram = np.array(item['histogram'])
        tf = histogram / np.sum(histogram)
        tf_idf = tf * idf
        tf_idf_normalized = tf_idf / np.linalg.norm(tf_idf, ord=1)
        tfidf.append({
            'label': label, 
            'tf_idf': tf_idf_normalized
        })
    return tfidf

def create_UNC_histogram(UNC_images, visual_words):
    histograms = []
    for item in UNC_images:
        label = item['label']
        vws = item['vw']
        his = np.zeros(len(visual_words))
        for i in range(0,len(his)):               
            his[i] += vws[i]
        histograms.append({
            'label': label,
            'histogram': his
        })
    return histograms