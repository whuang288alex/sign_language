from PIL import Image
import os

img = Image.open(os.path.join("captured","handd.png")).convert('L')
img.save('./captured/handd.jpg')