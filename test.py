import cv2
import numpy as np
import matplotlib.pyplot as plt

def encontrar_pixel_color(img_color, color_rgb, tolerancia=15):
    # Convertir el color a un rango aceptable para comparación
    color_min = (max(color_rgb[0] - tolerancia, 0),
                 max(color_rgb[1] - tolerancia, 0),
                 max(color_rgb[2] - tolerancia, 0))
    color_max = (min(color_rgb[0] + tolerancia, 255),
                 min(color_rgb[1] + tolerancia, 255),
                 min(color_rgb[2] + tolerancia, 255))

    # Obtener las dimensiones de la imagen
    alto, ancho, _ = img_color.shape

    # Recorrer la imagen píxel por píxel
    # Agregamos estos 100s para no buscar en los bordes.
    for y in range(100,alto):
        for x in range(100,ancho):
            pixel_color = img_color[y, x]
            if (color_min[0] <= pixel_color[0] <= color_max[0] and
                color_min[1] <= pixel_color[1] <= color_max[1] and
                color_min[2] <= pixel_color[2] <= color_max[2]):
                return x, y  # Devolver la ubicación x, y si se encuentra el color

    return None  #


# Color a buscar en RGB (en el plot están al revés)
color_a_buscar = (242, 209, 179)

imagenes_color = []
imagenes_bw = []
centros = []

# Buscar el color en la imagen
for i in range(1,10):
    img_color = cv2.imread(f'Patentes/img0{i}.png', cv2.IMREAD_COLOR)
    ubicacion = encontrar_pixel_color(img_color, color_a_buscar)
    centros.append(ubicacion)
    imagenes_color.append(img_color)
for i in range(10,13):
    img_color = cv2.imread(f'Patentes/img{i}.png', cv2.IMREAD_COLOR)
    ubicacion = encontrar_pixel_color(img_color, color_a_buscar)
    centros.append(ubicacion)
    imagenes_color.append(img_color)

print(centros)