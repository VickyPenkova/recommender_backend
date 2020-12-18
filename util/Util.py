import cv2
import matplotlib.pyplot as plt


def show_image(img):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()