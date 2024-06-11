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

def preproc(img):
    # BRILLO
    # Convertir la imagen de BGR a RGB (matplotlib utiliza RGB)
    placa_color_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convertir la imagen a float32 para evitar problemas de saturación
    placa_float = placa_color_rgb.astype(np.float32)
    # Agregarle brillo (se suma un valor constante a todos los píxeles)
    brillo_incremento = 60.0
    placa_brillante = placa_float + brillo_incremento
    # Asegurarse de que los valores estén en el rango correcto [0, 255]
    placa_brillante = np.clip(placa_brillante, 0, 255)
    # Convertir la imagen de vuelta a uint8
    img_proc = placa_brillante.astype(np.uint8)

    # CONTRASTE
    # Aplicar corrección gamma
    gamma = 1.3  # Valor de gamma (mayor que 1 para incrementar el contraste)
    img_proc = ((img_proc / 255.0) ** gamma) * 255.0
    img_proc = np.clip(img_proc, 0, 255).astype(np.uint8)

    # ESCALA DE GRISES
    img_gray = cv2.cvtColor(img_proc, cv2.COLOR_RGB2GRAY)

    return img_gray


placa_bw = cv2.imread('placa.png', cv2.IMREAD_GRAYSCALE)
placa_color = cv2.imread('placa.png', cv2.IMREAD_COLOR)
#imshow(placa_color, color_img=True)


img_preproc = preproc(placa_color)
imshow(img_preproc)


_, thresh = cv2.threshold(img_preproc, 190, 255, cv2.THRESH_BINARY_INV)
imshow(thresh, title='Imagen Umbralizada')






########################################################################################################################
# CAPACITORES
########################################################################################################################

cthresh = 255 - thresh
imshow(cthresh)


##########
# COMPONENTES CONEXAS
##########

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cthresh)
imshow(labels)

H, W = cthresh.shape[:2]
# --- Defino parametros para la clasificación -------------------------------------------
RHO_TH = 0.23    # Factor de forma (rho)
AREA_TH = 6500   # Umbral de area
aux = np.zeros_like(labels)
labeled_image = cv2.merge([aux, aux, aux])

capacitores = []
# --- Clasificación ---------------------------------------------------------------------
# Clasifico en base al factor de forma
for i in range(1, num_labels):

    # --- Remuevo las celulas que tocan el borde de la imagen -----------------
    if (stats[i, cv2.CC_STAT_LEFT] == 0 or 
        stats[i, cv2.CC_STAT_TOP] == 0 or 
        stats[i, cv2.CC_STAT_HEIGHT] + stats[i, cv2.CC_STAT_TOP] == H or 
        stats[i, cv2.CC_STAT_WIDTH] + stats[i, cv2.CC_STAT_LEFT] == W):
        continue

    # --- Remuevo celulas con area chica --------------------------------------
    if (stats[i, cv2.CC_STAT_AREA] < AREA_TH):
        continue

    # --- Selecciono el objeto actual -----------------------------------------
    obj = (labels == i).astype(np.uint8)

    # --- Calculo Rho ---------------------------------------------------------
    ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(ext_contours[0])
    perimeter = cv2.arcLength(ext_contours[0], True)
    rho = 4 * np.pi * area/(perimeter**2)
    flag_circular = rho > RHO_TH

    # --- Calculo cantidad de huecos ------------------------------------------
    all_contours, _ = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    holes = len(all_contours) - 1

    # --- Muestro por pantalla el resultado -----------------------------------
    print(f"Objeto {i:2d} --> Circular: {flag_circular}  /  Huecos: {holes}  /  Rho: {rho}  / Area: {stats[i, cv2.CC_STAT_AREA]}")

    # --- Clasifico -----------------------------------------------------------
    if flag_circular:
        capacitores.append(i)
        if holes == 1:
            labeled_image[obj == 1, 0] = 255    # Circular con 1 hueco
        else:
            labeled_image[obj == 1, 1] = 255    # Circular con mas de 1 hueco
    else:
        pass
        #labeled_image[obj == 1, 2] = 255        # No circular

plt.figure(); plt.imshow(labeled_image); plt.show(block=False)


#######
# VISUALIZACION SOBRE IMAGEN ORIGINAL
#######


# Crear una copia de la imagen original para dibujar sobre ella
output_img = placa_color.copy()

# Dibujar rectángulos y marcar centroides
for i in capacitores:
    # Obtener los datos del componente
    x, y, w, h, area = stats[i]
    cx, cy = centroids[i]
    
    # Dibujar el rectángulo en verde
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Marcar el centroide en rojo
    cv2.circle(output_img, (int(cx), int(cy)), 5, (0, 0, 255), -1)

# Mostrar la imagen resultante
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.title('Imagen con Bounding Boxes y Centroides')
plt.axis('off')
plt.show()








########################################################################################################################
# RESISTENCIAS
########################################################################################################################












########################################################################################################################
# CHIP
########################################################################################################################

def preproc_chip(img):
    # BRILLO
    # Convertir la imagen de BGR a RGB (matplotlib utiliza RGB)
    placa_color_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convertir la imagen a float32 para evitar problemas de saturación
    placa_float = placa_color_rgb.astype(np.float32)
    # Agregarle brillo (se suma un valor constante a todos los píxeles)
    brillo_incremento = 80.0
    placa_brillante = placa_float + brillo_incremento
    # Asegurarse de que los valores estén en el rango correcto [0, 255]
    placa_brillante = np.clip(placa_brillante, 0, 255)
    # Convertir la imagen de vuelta a uint8
    img_proc = placa_brillante.astype(np.uint8)

    # CONTRASTE
    # Aplicar corrección gamma
    gamma = 1.8  # Valor de gamma (mayor que 1 para incrementar el contraste)
    img_proc = ((img_proc / 255.0) ** gamma) * 255.0
    img_proc = np.clip(img_proc, 0, 255).astype(np.uint8)

    return img_proc

placa_proc = preproc_chip(placa_bw)
imshow(placa_proc)

# Umbralado
_, thresh_chip = cv2.threshold(placa_proc, 100, 255, cv2.THRESH_BINARY_INV)
thresh_chip = 255 - thresh_chip
imshow(thresh_chip)

# Canny
f_blur_chip = cv2.GaussianBlur(thresh_chip, ksize=(5,5), sigmaX=1.5)
gcan_chip = cv2.Canny(f_blur_chip, threshold1=0.04*255, threshold2=0.7*255)
imshow(gcan_chip)


# --- Gradiente Morfológico (Morphological Gradient)
L = 3
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (L, L) )
f_mg = cv2.morphologyEx(gcan_chip, cv2.MORPH_GRADIENT, kernel)
imshow(f_mg)



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

chip_relleno = imfillhole(f_mg)
imshow(chip_relleno)



############
# COMPONENTES CONECTADAS
############

connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(chip_relleno, connectivity, cv2.CV_32S)
imshow(img=labels)

# Dibujar el rectángulo y el centroide
output_image = cv2.cvtColor(placa_color, cv2.COLOR_GRAY2BGR)  # Convertir a BGR para poder dibujar en color

# Obtener las estadísticas y el centroide de la etiqueta deseada
x, y, w, h, area = stats[276]
cx, cy = centroids[276]

# Dibujar el rectángulo (en color verde)
cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Dibujar el centroide (en color rojo)
cv2.circle(output_image, (int(cx), int(cy)), 5, (0, 0, 255), -1)

# Mostrar la imagen resultante
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
