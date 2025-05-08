import os
import cv2
import base64
import numpy as np
from joblib import load
from skimage.feature import hog

MODEL_FILENAME = "modelo_svm.joblib"

def preprocess_image(img):
    """Aplica o mesmo pré-processamento usado no script de treino."""
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        img_rgb = img[:, :, :3]
        white_bg = np.ones_like(img_rgb, dtype=np.uint8) * 255
        img = np.where(alpha[..., None] == 0, white_bg, img_rgb)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 1)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    horizontal_kernel = np.ones((1, 2), np.uint8)
    processed = cv2.morphologyEx(opened, cv2.MORPH_OPEN, horizontal_kernel)

    return processed

def segment_characters_by_split(image, num_chars=6, padding=2):
    """Segmenta horizontalmente com padding."""
    height, width = image.shape
    char_width = width // num_chars
    characters = []

    for i in range(num_chars):
        x_start = max(i * char_width - padding, 0)
        x_end = min((i + 1) * char_width + padding, width)
        char = image[:, x_start:x_end]
        characters.append(char)

    return characters

def decripto(content: str, type: str) -> str:
    if type == "image" and not content.startswith("data:image"):
        content = "data:image/png;base64," + content

    b64 = content.split(",", 1)[1] if content.startswith("data:image") else content
    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Não foi possível decodificar a imagem.")

    # Salvar imagem original temporária
    script_dir = os.path.abspath(os.path.dirname(__file__))
    tmp_dir = os.path.join(script_dir, "resources", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    temp_name = os.path.join(tmp_dir, "last_upload.png")
    cv2.imwrite(temp_name, img)

    # Alinhar ao pipeline de pré-processamento do treino
    processed_img = preprocess_image(img)

    # Segmentar com padding (usado no treino)
    chars = segment_characters_by_split(processed_img, num_chars=6, padding=2)

    # Carregar modelo
    modelo_path = os.path.join(script_dir, MODEL_FILENAME)
    if not os.path.isfile(modelo_path):
        raise FileNotFoundError(f"Modelo não encontrado em {modelo_path}")
    model = load(modelo_path)

    texto = ""
    for sl in chars:
        im = cv2.resize(sl, (20, 20))
        _, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        feats = hog(im, pixels_per_cell=(4, 4), cells_per_block=(1, 1))
        pred = model.predict([feats])[0]
        texto += pred

    # Renomear imagem com o texto reconhecido
    final_name = os.path.join(tmp_dir, f"{texto}.png")
    os.replace(temp_name, final_name)

    return texto
