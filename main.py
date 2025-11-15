import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Używamy backend bez wyświetlacza
from matplotlib import pyplot as plt

# -----------------------------
# 1) WCZYTANIE OBRAZU (URL lub lokalnie)
# -----------------------------

# Tworzenie przykładowego obrazu testowego (gradient kolorowy)
print("Tworzenie przykładowego obrazu testowego...")
height, width = 400, 600
img = np.zeros((height, width, 3), dtype=np.uint8)

# Tworzenie gradientu kolorowego
for i in range(height):
    for j in range(width):
        img[i, j] = [int(255 * i / height), int(255 * j / width), 128]

print(f"Utworzono obraz o wymiarach: {img.shape}")

# Zapisanie oryginalnego obrazu
plt.figure(figsize=(6,6))
plt.title("Oryginalny obraz")
plt.imshow(img)
plt.axis('off')
plt.savefig('original_image.png', bbox_inches='tight')
print("Zapisano: original_image.png")

# -----------------------------
# 2) ZMNIEJSZENIE ROZDZIELCZOŚCI O 50%
# -----------------------------
height, width = img.shape[:2]
new_size = (width // 2, height // 2)
img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

# -----------------------------
# 3) ZMIANA NA GRAYSCALE
# -----------------------------
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

# -----------------------------
# 4) OBRÓT O 90 STOPNI W PRAWO
# -----------------------------
img_rotated = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)

# -----------------------------
# 5) ZAPISANIE OBRAZU WYNIKOWEGO
# -----------------------------
plt.figure(figsize=(6,6))
plt.title("Obraz po przetwarzaniu")
plt.imshow(img_rotated, cmap="gray")
plt.axis('off')
plt.savefig('processed_image.png', bbox_inches='tight')
print("Zapisano: processed_image.png")

# -----------------------------
# 6) WYŚWIETLENIE MACIERZY OBRAZU
# -----------------------------
print("Macierz obrazu (fragment):")
print(img_rotated)

print("\nRozmiar macierzy:", img_rotated.shape)
