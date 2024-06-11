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


############
# VEAMOS UNA DE LAS IMÁGENES
############

# Ruta de la imagen
imagen_path = 'Patentes/img01.png'

# Cargar la imagen a color
img_color = cv2.imread(imagen_path, cv2.IMREAD_COLOR)
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Desactivar los ejes
plt.show()

# Escala de Grises
img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
imshow(img)




############
# VEAMOS LOS HISTOGRAMAS DE LAS PRIMERAS 9 IMÁGENES
############

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.ravel()  # Aplanar la matriz de ejes para un acceso fácill

# Iterar sobre las 9 imágenes y sus correspondientes subplots
for i in range(1, 10):
    # Leer la imagen
    imagen_path = f'Patentes/img0{i}.png'
    imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    
    if imagen is None:
        print(f"Error al cargar la imagen {imagen_path}")
        continue
    
    # Calcular el histograma
    hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])
    
    # Ploteo del histograma en el subplot correspondiente
    axes[i-1].plot(hist, color='black')
    axes[i-1].set_title(f'Imagen {i}')
    axes[i-1].set_xlim([0, 256])

# Ajustar el diseño de la figura
plt.tight_layout()
plt.show()








############
# ¿Podemos identificar el "celeste" de la patente en todas las imágenes?
############

# Nos encontramos con el problema de que había celestes en las ópticas, así que vamos a sumar la condicion
# de que una ventana de 5x5 10 píxeles por debajo de la ubicación sea lo suficientemente oscura
# representando lo negro entre las letras y los números.

def black_amount(img_bw, ubi):
    x, y = ubi
    # Definir los límites de la ventana alrededor de la ubicación (ubi)
    ymin = max(y - 2, 0)
    ymax = min(y + 2, img_bw.shape[0] - 1)
    xmin = max(x - 2, 0)
    xmax = min(x + 2, img_bw.shape[1] - 1)

    # Extraer la región de interés (ventana)
    ventana = img_bw[ymin:ymax + 1, xmin:xmax + 1]

    # Calcular el promedio de intensidad de la ventana
    promedio = np.mean(ventana)
    return promedio

def saturar_negros(img):
    img_saturada = np.copy(img)
    img_saturada[img < 50] = 0
    return img_saturada

def encontrar_pixel_color(img_color, img_bw, color_rgb, tolerancia=15):
    # Obtener las dimensiones de la imagen
    alto, ancho, _ = img_color.shape

    # Recorrer la imagen píxel por píxel
    # Agregamos estos 100s para no buscar en los bordes.
    for y in range(100,alto):
        for x in range(100,ancho):
            pixel_color = img_color[y, x]
            azul = pixel_color[0]
            verde = pixel_color[1]
            rojo = pixel_color[2]
            if azul > verde*1.15 and verde>rojo*1.15 and verde>145 and rojo>115 and black_amount(img_bw,(x,y+10))<20 and black_amount(img_bw,(x,y))>50:
                return x, y  # Devolver la ubicación x, y si se encuentra el color

    return None  #

# Color celeste a buscar en RGB (en el plot están al revés)
color_a_buscar = (242, 209, 179)

imagenes_color = []
imagenes_bw = []
centros = []

# Buscar el color en la imagen
for i in range(1,10):
    img_color = cv2.imread(f'Patentes/img0{i}.png', cv2.IMREAD_COLOR)
    img_bw = saturar_negros(cv2.imread(f'Patentes/img0{i}.png', cv2.IMREAD_GRAYSCALE))
    ubicacion = encontrar_pixel_color(img_color, img_bw, color_a_buscar)
    centros.append(ubicacion)
    imagenes_color.append(img_color)
    imagenes_bw.append(img_bw)
for i in range(10,13):
    img_color = cv2.imread(f'Patentes/img{i}.png', cv2.IMREAD_COLOR)
    img_bw = saturar_negros(cv2.imread(f'Patentes/img{i}.png', cv2.IMREAD_GRAYSCALE))
    ubicacion = encontrar_pixel_color(img_color, img_bw, color_a_buscar)
    centros.append(ubicacion)
    imagenes_color.append(img_color)
    imagenes_bw.append(img_bw)



print(centros)



############
# ¿Realmente estamos viendo los centros de las patentes?
############

# Lista para almacenar los recortes de las imágenes
recortes = []
recortes_bw = []
i=0
# Iterar sobre cada imagen y su punto correspondiente
for img_color, img_bw, centro in zip(imagenes_color, imagenes_bw, centros):
    i=i+1
    if centro is None:
        print(f"Problemas al leer la imagen {i}.")
        continue
    # Obtener las coordenadas para el recorte
    x1 = centro[0] - 60
    y1 = centro[1] - 45
    x2 = centro[0] + 60
    y2 = centro[1] + 45

    # Recortar la imagen
    recorte = img_color[y1:y2, x1:x2]

    # Dibujar un círculo verde alrededor del centro relativo
    cv2.circle(recorte, (60, 45), 1, (0, 255, 0), 1)

    # Dibujar un círculo verde alrededor del centro relativo
    cv2.rectangle(recorte, (58, 53), (62, 57), (0, 0, 255), 1)

    # Agregar el recorte a la lista de recortes
    recortes.append(recorte)
    recorte_bw = img_bw[y1:y2, x1:x2]
    recortes_bw.append(recorte_bw)

# Crear el plot para mostrar los recortes
fig, axs = plt.subplots(3, 4, figsize=(12, 9))

# Iterar sobre los recortes y mostrarlos en el plot
for i, recorte in enumerate(recortes):
    fila = i // 4
    columna = i % 4
    axs[fila, columna].imshow(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
    axs[fila, columna].axis('off')

# Ajustar los márgenes del plot
plt.tight_layout()
plt.show()










############
# ¿Cómo podemos reducir el rango dinámico para que el centro de las patentes siempre mapee a 0/negro y las letras a 255/blanco?
############

# "Negro" en img01 = 27
# "Blanco" en img01 = 199
# "Negro" en img02 = 34
# "Blanco" en img02 = 167

# Reduzcamos el rango dinámico para que todos los valores menores a 50 mapeen a 0, y todos los mayores a 160 a 255.
# Saturación

# Aplicar la saturación de los valores
def saturar(img, min_val, max_val):
    img_saturada = np.copy(img)
    img_saturada[img < min_val] = min_val
    img_saturada[img > max_val] = max_val
    return img_saturada

# Definir los valores mínimos y máximos
min_val = 60
max_val = 120

recortes_saturados = []
for recorte in recortes_bw:
    # Saturar imagen
    recorte_saturado = saturar(recorte, min_val, max_val)
    
    # Calcular el mínimo y el máximo de la imagen por si no coinciden
    min_val = np.min(recorte_saturado)
    max_val = np.max(recorte_saturado)
    
    # Expandir el contraste de la imagen
    img_expandida = cv2.convertScaleAbs(recorte_saturado, alpha=255.0/(max_val - min_val), beta=-255.0*min_val/(max_val - min_val))
    
    recortes_saturados.append(img_expandida)


# Crear el plot para mostrar los recortes
fig, axs = plt.subplots(3, 4, figsize=(12, 9))

# Iterar sobre los recortes y mostrarlos en el plot
for i, recorte in enumerate(recortes_saturados):
    fila = i // 4
    columna = i % 4
    axs[fila, columna].imshow(recorte, cmap='gray')
    axs[fila, columna].axis('off')

# Ajustar los márgenes del plot
plt.tight_layout()
plt.show()








############
# Analicemos como podemos pre-procesar la imagen para poder segmentar visualmente lo mejor posible las patentes
############
imagen_saturada = recortes_saturados[0]


# Filtrado
ddepth = cv2.CV_64F  # Formato salida

# Calcular gradiente en X y Y usando Sobel
grad_x = cv2.Sobel(imagen_saturada, ddepth, 1, 0, ksize=3)
grad_y = cv2.Sobel(imagen_saturada, ddepth, 0, 1, ksize=3)

# Calcular magnitud del gradiente
grad = np.sqrt(grad_x**2 + grad_y**2)
grad_aprox = grad_x**2 + grad_y**2

# Normalizar gradientes
grad_n = cv2.convertScaleAbs(grad)
grad_aprox_n = cv2.convertScaleAbs(grad_aprox)

# Umbralado
# VARIABLE
grad_th = grad >= grad.max() * 0.1
grad_aprox_th = grad_aprox >= grad_aprox.max() * 0.01

# Segunda versión del cálculo del gradiente
grad2_x = cv2.Sobel(imagen_saturada, ddepth, 1, 0, ksize=3)
grad2_y = cv2.Sobel(imagen_saturada, ddepth, 0, 1, ksize=3)
abs_grad2_x = cv2.convertScaleAbs(grad2_x)
abs_grad2_y = cv2.convertScaleAbs(grad2_y)
grad2 = cv2.addWeighted(abs_grad2_x, 0.5, abs_grad2_y, 0.5, 0)

# Umbralado para la segunda versión
# VARIABLE
grad2_th = grad2 >= grad2.max() * 0.51

# Visualización
plt.figure()
ax = plt.subplot(321)
imshow(grad, new_fig=False, title="Gradiente - Magnitud")
plt.subplot(322, sharex=ax, sharey=ax), imshow(grad_th, new_fig=False, title="Gradiente - Magnitud + Umbralado")
plt.subplot(323, sharex=ax, sharey=ax), imshow(grad_aprox, new_fig=False, title="Gradiente - Magnitud aprox.")
plt.subplot(324, sharex=ax, sharey=ax), imshow(grad_aprox_th, new_fig=False, title="Gradiente - Magnitud aprox. + Umbralado")
plt.subplot(325, sharex=ax, sharey=ax), imshow(grad2, new_fig=False, title="Gradiente version 2")
plt.subplot(326, sharex=ax, sharey=ax), imshow(grad2_th, new_fig=False, title="Gradiente version 2 + Umbralado")
plt.show(block=False)

# Nos quedamos con Gradiente - Magnitud aprox. + Umbralado
imagen_umbralada = grad_aprox_th.astype(np.uint8) * 255

# Chequeamos que sea matriz binaria
print(np.unique(imagen_umbralada))




#########
# GUARDEMOS LOS RECORTES PROCESADOS
#########

def procesado(recorte):
    # Filtrado
    ddepth = cv2.CV_64F  # Formato salida

    # Calcular gradiente en X y Y usando Sobel
    grad_x = cv2.Sobel(recorte, ddepth, 1, 0, ksize=3)
    grad_y = cv2.Sobel(recorte, ddepth, 0, 1, ksize=3)
    grad_aprox = grad_x**2 + grad_y**2
    grad_aprox_th = grad_aprox >= grad_aprox.max() * 0.01

    return grad_aprox_th.astype(np.uint8) * 255


recortes_procesados = []
for recorte in recortes_saturados:
    recortes_procesados.append(procesado(recorte))


# Crear el plot para mostrar los recortes
fig, axs = plt.subplots(3, 4, figsize=(12, 9))
# Iterar sobre los recortes y mostrarlos en el plot
for i, recorte in enumerate(recortes_procesados):
    fila = i // 4
    columna = i % 4
    axs[fila, columna].imshow(recorte, cmap='gray')
    axs[fila, columna].axis('off')

# Ajustar los márgenes del plot
plt.tight_layout()
plt.show()






############
# Veamos qué tan bien podemos identificar o segmentar la imagen con CANNY
############
ejemplo = recortes_saturados[0]
ejemplo_original = recortes[0]

# --- CANNY ---------------------------------------------------------------------------------------
f_blur = cv2.GaussianBlur(ejemplo, ksize=(3, 3), sigmaX=1.5)
gcan = cv2.Canny(f_blur, threshold1=0.02*255, threshold2=0.07*255)
imshow(gcan)


############
# Este algoritmo se comporta bastante mejor, guardemos todos los recortes procesados con CANNY
############

recortes_canny = []
for recorte in recortes_saturados:
    f_blur = cv2.GaussianBlur(recorte, ksize=(3, 3), sigmaX=1.5)
    gcan = cv2.Canny(f_blur, threshold1=0.02*255, threshold2=0.07*255)
    recortes_canny.append(gcan)

# Crear el plot para mostrar los recortes
fig, axs = plt.subplots(3, 4, figsize=(12, 9))
# Iterar sobre los recortes y mostrarlos en el plot
for i, recorte in enumerate(recortes_canny):
    fila = i // 4
    columna = i % 4
    axs[fila, columna].imshow(recorte, cmap='gray')
    axs[fila, columna].axis('off')

# Ajustar los márgenes del plot
plt.tight_layout()
plt.show()


