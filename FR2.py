from PIL import Image#pip install pillow
import pytesseract
import cv2 #pip install opencv-python
import os
import numpy as np


image = 'C:\\Users\\IMatveev\\PycharmProjects\\FR2\\1.png'
print(image)
preprocess = "thresh"
# загрузить образ и преобразовать его в оттенки серого
image = cv2.imread(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# проверьте, следует ли применять пороговое значение для предварительной обработки изображения
if preprocess == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# если нужно медианное размытие, чтобы удалить шум
elif preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)
img_erode = cv2.erode(gray, np.ones((3, 3), np.uint8), iterations=1)
contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(contours)
output = image.copy()
letters = []
for idx, contour in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(contour)
    # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
    # hierarchy[i][0]: the index of the next contour of the same level
    # hierarchy[i][1]: the index of the previous contour of the same level
    # hierarchy[i][2]: the index of the first child
    # hierarchy[i][3]: the index of the parent
    if hierarchy[0][idx][3] == 0:
        cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
        letter_crop = gray[y:y + h, x:x + w]
        # print(letter_crop.shape)
        # Resize letter canvas to square
        size_max = max(w, h)
        letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
        if w > h:
            # Enlarge image top-bottom
            # ------
            # ======
            # ------
            y_pos = size_max // 2 - h // 2
            letter_square[y_pos:y_pos + h, 0:w] = letter_crop
        elif w < h:
            # Enlarge image left-right
            # --||--
            x_pos = size_max // 2 - w // 2
            letter_square[0:h, x_pos:x_pos + w] = letter_crop
        else:
            letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
        letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))
        # Sort array in place by X-coordinate
        letters.sort(key=lambda x: x[0], reverse=False)

cv2.imshow("output", output)
cv2.waitKey(0)

# # сохраним временную картинку в оттенках серого, чтобы можно было применить к ней OCR
# filename = "{}.png".format(os.getpid())
# cv2.imwrite(filename, gray)
#
# # загрузка изображения в виде объекта image Pillow, применение OCR, а затем удаление временного файла
# text = pytesseract.image_to_string(Image.open(filename))
# os.remove(filename)
# print(text)
#
# # показать выходные изображения
# cv2.imshow("Image", image)
# cv2.imshow("Output", gray)
