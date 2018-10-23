from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np
import os
import glob

if __name__ == "__main__":
    print("Load model")
    kmeans = joblib.load("exp/k_means.joblib")
    #print("Loading model")
    for folder in os.listdir("features/vgg16_fc2/test/"):
        folder_path = os.path.join("features/vgg16_fc2/test/",folder) 
        for file in os.listdir(folder_path):
            if not file.startswith('.'):
                data = np.load(os.path.join(folder_path, file))
                group = kmeans.predict(data)
                token = file.split("+")
                class_name = token[0]
                file_name = token[1].replace(".npy","")
                print("File ", file_name, " | class : ", class_name, " | group : ", group)