from PIL import Image
img = Image.open('rem3.png').convert('LA')
img.save('grayscale.png')
