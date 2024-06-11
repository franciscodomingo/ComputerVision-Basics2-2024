import cv2
import numpy as np
import matplotlib.pyplot as plt

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)


placa_bw = cv2.imread('placa.png', cv2.IMREAD_GRAYSCALE)
placa_color = cv2.imread('placa.png', cv2.IMREAD_COLOR)
imshow(placa_bw)


_, thresh = cv2.threshold(placa_bw, 140, 255, cv2.THRESH_BINARY_INV)
imshow(thresh, title='Imagen Umbralizada')


_, thresh_chip = cv2.threshold(placa_bw, 90, 255, cv2.THRESH_BINARY_INV)
thresh_chip = 255 - thresh_chip
imshow(thresh_chip)



# --- CANNY ---------------------------------------------------------------------------------------
f_blur = cv2.GaussianBlur(thresh, ksize=(2,2), sigmaX=1.5)
gcan = cv2.Canny(f_blur, threshold1=0.4*255, threshold2=0.7*255)
imshow(gcan)

f_blur_chip = cv2.GaussianBlur(thresh_chip, ksize=(5,5), sigmaX=1.5)
gcan_chip = cv2.Canny(f_blur_chip, threshold1=0.4*255, threshold2=0.7*255)
imshow(gcan_chip)





########################################################################################################################

# CLAUSURA
A = gcan_chip
B = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
chip_cerrado = cv2.morphologyEx(A, cv2.MORPH_CLOSE, B, iterations=5)
imshow(chip_cerrado)


# ------------------------------------------
# --- Rellenado de huecos -----------------------------------------------------
# ------------------------------------------
def imreconstruct(marker, mask, kernel=None):
    if kernel==None:
        kernel = np.ones((5,5), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                         # Finalizacion?
            break                                                           #
        marker = expanded_intersection        
    return expanded_intersection

def imfillhole(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)                                                   # Genero mascara para...
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
    marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh

chip_relleno = imfillhole(chip_cerrado)
imshow(chip_relleno)





############
# COMPONENTES CONECTADAS
############

connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(chip_relleno, connectivity, cv2.CV_32S)
imshow(img=labels)
