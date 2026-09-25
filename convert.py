from PIL import Image
import numpy as np

img = Image.open("my_drawing.jpg").convert("L")
img = img.resize((128, 128), Image.Resampling.LANCZOS)
img_array = np.array(img).astype(np.float64) / 255.0
dataset_img = []
dataset_img.append(img_array.flatten())
with open("custom_image.bin", "wb") as f:
    f.write(*dataset_img)
