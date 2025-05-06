import os, cv2, base64, numpy as np
from joblib import load
from skimage.feature import hog

MODEL_FILENAME = "modelo_svm.joblib"

def decripto(content: str, type: str) -> str:
    # Se for imagem bruta em Base64, já vem no formato data:image…
    # Caso ‘type’ seja 'image', convertemos resp.content p/ Base64
    if type == "image" and not content.startswith("data:image"):
        content = "data:image/png;base64," + content

    # Agora o fluxo é igual ao anterior:
    b64 = content.split(",", 1)[1] if content.startswith("data:image") else content
    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Não foi possível decodificar a imagem.")
    # ... todo o seu pipeline de pré-processamento, segmentação e predição ...
    # por fim, return texto


    # 3) Tratamento de canal alfa (se houver)
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        img_bgr = img[:, :, :3]
        white = np.ones_like(img_bgr, dtype=np.uint8) * 255
        img = np.where(alpha[..., None] == 0, white, img_bgr)

    # 4) ----- SALVA A IMAGEM ORIGINAL DECODIFICADA EM resources/tmp dentro de zeta -----
    # Define caminho absoluto ao redor de run.py (datasource/zeta)
    script_dir = os.path.abspath(os.path.dirname(__file__))
    tmp_dir = os.path.join(script_dir, "resources", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Usa nome temporário até decodificar o texto
    temp_name = os.path.join(tmp_dir, "last_upload.png")
    cv2.imwrite(temp_name, img)

    # 5) Pré‑processamento para OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    # 6) Segmentação fixa em 6 fatias horizontais
    h, w = thresh.shape
    num_chars = 6
    char_w = w // num_chars
    slices = []
    padding = 2
    for i in range(num_chars):
        xs = max(i * char_w - padding, 0)
        xe = min((i + 1) * char_w + padding, w)
        slices.append(thresh[:, xs:xe])

    # 7) Carrega o modelo
    modelo_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
    if not os.path.isfile(modelo_path):
        raise FileNotFoundError(f"Modelo não encontrado em {modelo_path}")
    model = load(modelo_path)

    # 8) Para cada fatia, extrai HOG e prediz
    texto = ""
    for sl in slices:
        im = cv2.resize(sl, (20, 20))
        _, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        feats = hog(im, pixels_per_cell=(4, 4), cells_per_block=(1, 1))
        pred = model.predict([feats])[0]
        texto += pred

    # 9) Renomeia o PNG salvo para usar o texto reconhecido
    final_name = os.path.join(tmp_dir, f"{texto}.png")
    os.replace(temp_name, final_name)

    return texto
