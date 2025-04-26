# Cargar imágenes de ejemplo (asumiendo estructura de carpetas COVID, Normal, etc.)
def cargar_imagenes_ejemplo(ruta='data/sample_images', num_por_clase=2):
    imagenes = {}
    for clase in os.listdir(ruta):
        clase_path = os.path.join(ruta, clase)
        if os.path.isdir(clase_path):
            imagenes[clase] = []
            for img_name in os.listdir(clase_path)[:num_por_clase]:
                img_path = os.path.join(clase_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imagenes[clase].append(img)
                except:
                    continue
    return imagenes
