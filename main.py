import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

if __name__ == '__main__':
    nivel_artistic = input("Introdu nivelul artistic dorit (intre 0 si 255) ")
    nivel_artistic = int(nivel_artistic)
    if nivel_artistic not in range(0, 256):
        raise ValueError("Nivelul artistic nu se afla in intervalul cerut")

    nivel_artistic = 255 - nivel_artistic

    imagine = plt.imread("imagine.jpg")
    size = imagine.shape
    imagine = np.array(imagine)

    imagine = imagine.reshape(size[0] * size[1], 3)

    kmeans = KMeans(nivel_artistic)
    labels = kmeans.fit_predict(imagine)
    medii = kmeans.cluster_centers_

    dictionar_medii = dict()
    for id, medie in enumerate(medii):
        dictionar_medii[id] = list(medie)

    imagine_noua = np.array([dictionar_medii[elem] for elem in labels])

    plt.title("Arta contemporana\nculori folosite: " + str(nivel_artistic))
    imagine_noua = imagine_noua.reshape(size[0], size[1], 3)
    plt.imshow(np.uint8(imagine_noua))
    plt.show()