import sys
import os
import requests
import base64
import time
import cv2
import string
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from joblib import dump

class CaptchaProcessor:
    def __init__(self):
        pass

    def salvar_imagem_base64(self, base64_str, pasta_destino, nome_arquivo):
        try:
            if base64_str.startswith("data:image"):
                base64_str = base64_str.split(",", 1)[1]
            imgdata = base64.b64decode(base64_str)
            os.makedirs(pasta_destino, exist_ok=True)
            caminho_arquivo = os.path.join(pasta_destino, nome_arquivo)
            with open(caminho_arquivo, 'wb') as f:
                f.write(imgdata)
            print(f"Imagem salva em: {caminho_arquivo}")
        except Exception as e:
            print(f"Erro ao salvar a imagem: {e}")

    def baixar_imagens(self, url_base, quantidade, pasta_destino, usar_base64=False):
        os.makedirs(pasta_destino, exist_ok=True)
        arquivos_existentes = [f for f in os.listdir(pasta_destino) if f.lower().endswith('.png')]
        contador_inicial = len(arquivos_existentes)
        for i in range(quantidade):
            try:
                resposta = requests.get(url_base)
                if resposta.status_code == 200:
                    nome_arquivo = f"captcha_{contador_inicial + i + 1}.png"
                    caminho_arquivo = os.path.join(pasta_destino, nome_arquivo)
                    if usar_base64:
                        if resposta.text.startswith("data:image"):
                            self.salvar_imagem_base64(resposta.text, pasta_destino, nome_arquivo)
                        else:
                            print(f"Imagem {i+1} não está em formato Base64.")
                    else:
                        with open(caminho_arquivo, 'wb') as f:
                            f.write(resposta.content)
                        print(f"Imagem salva em: {caminho_arquivo}")
                else:
                    print(f"Falha ao baixar a imagem {i+1}: Status {resposta.status_code}")
            except Exception as e:
                print(f"Erro ao baixar a imagem {i+1}: {e}")
            time.sleep(3)

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            img = img[:, :, :3]
            white_bg = np.ones_like(img, dtype=np.uint8) * 255
            img = np.where(alpha[..., None] == 0, white_bg, img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 1)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        horizontal_kernel = np.ones((1, 2), np.uint8)
        horizontal_lines_removed = cv2.morphologyEx(opened, cv2.MORPH_OPEN, horizontal_kernel)
        return horizontal_lines_removed

    def augment_image(self, img):
        rows, cols = img.shape
        angle = np.random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows), borderValue=255)
        return rotated

    def segment_characters_by_split(self, image, num_chars=6, padding=2):
        height, width = image.shape
        char_width = width // num_chars
        characters = []
        for i in range(num_chars):
            x_start = max(i * char_width - padding, 0)
            x_end = min((i + 1) * char_width + padding, width)
            char = image[:, x_start:x_end]
            characters.append(char)
        return characters

    def treinar_modelo(self, pasta_segmentos='processed', modelo_saida='modelo_svm.joblib'):
        X, y = [], []
        for nome_arquivo in os.listdir(pasta_segmentos):
            if not nome_arquivo.lower().endswith('.png'):
                continue
            try:
                nome_base = nome_arquivo.split('.')[0]
                partes = nome_base.split('_')
                if len(partes) < 3:
                    print(f"[!] Nome de arquivo inválido: {nome_arquivo}")
                    continue
                rotulo = partes[1].upper()
            except Exception as e:
                print(f"[!] Erro ao extrair rótulo de {nome_arquivo}: {e}")
                continue
            if rotulo not in string.ascii_uppercase:
                continue
            caminho = os.path.join(pasta_segmentos, nome_arquivo)
            imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
            if imagem is None:
                print(f"[!] Erro ao ler imagem: {caminho}")
                continue
            imagem = cv2.resize(imagem, (20, 20))
            _, imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            features = hog(imagem, pixels_per_cell=(4, 4), cells_per_block=(1, 1))
            X.append(features)
            y.append(rotulo)

        if not X:
            print("[!] Nenhuma imagem válida encontrada para treinamento.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo = svm.SVC(kernel='linear', probability=True)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        print(f"Acurácia do modelo: {acuracia * 100:.2f}%")
        dump(modelo, modelo_saida)
        print(f"Modelo salvo em: {modelo_saida}")

    def menu(self):
        while True:
            print("\nSelecione uma opção:")
            print("1. Baixar captchas")
            print("2. Pré-processar imagens")
            print("3. Segmentar caracteres")
            print("4. Aumentar dados (augmentation)")
            print("5. Treinar modelo SVM")
            print("6. Sair")
            escolha = input("Digite o número da opção desejada: ")

            if escolha == '1':
                url = input("Digite a URL para download dos captchas: ")
                quantidade = int(input("Digite a quantidade de captchas a serem baixados: "))
                pasta = input("Digite o nome da pasta de destino[default=captchas]: ") or 'captchas'
                base64_input = input("As imagens estão em formato Base64? (s/n)[defualt=s]: ").lower() or 's'
                usar_base64 = base64_input == 's'
                self.baixar_imagens(url, quantidade, pasta, usar_base64)

            elif escolha == '2':
                pasta = input("Digite o nome da pasta contendo as imagens[default=captchas]: ") or 'captchas'
                pasta_saida = 'preprocess'
                os.makedirs(pasta_saida, exist_ok=True)
                for filename in os.listdir(pasta):
                    if filename.lower().endswith('.png'):
                        caminho_imagem = os.path.join(pasta, filename)
                        imagem_processada = self.preprocess_image(caminho_imagem)
                        caminho_saida = os.path.join(pasta_saida, filename)
                        cv2.imwrite(caminho_saida, imagem_processada)
                        print(f"Imagem pré-processada salva em: {caminho_saida}")

            elif escolha == '3':
                pasta_entrada = input("Digite o nome da pasta contendo as imagens pré-processadas [default=preprocess]: ").strip() or 'preprocess'
                pasta_saida = 'processed'
                os.makedirs(pasta_saida, exist_ok=True)

                for filename in os.listdir(pasta_entrada):
                    if filename.lower().endswith('.png'):
                        caminho_imagem = os.path.join(pasta_entrada, filename)
                        imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
                        nome_base = os.path.splitext(filename)[0]
                        num_chars = len(nome_base)

                        caracteres = self.segment_characters_by_split(imagem, num_chars=num_chars, padding=2)

                        for idx, char in enumerate(caracteres):
                            if idx < len(nome_base):
                                rotulo = nome_base[idx]
                            else:
                                rotulo = f"char{idx+1}"  # fallback seguro

                            # Novo nome com índice da posição para evitar sobrescrita
                            nome_saida = f"{nome_base}_{rotulo}_{idx+1}.png"
                            caminho_saida = os.path.join(pasta_saida, nome_saida)
                            cv2.imwrite(caminho_saida, char)
                            print(f"Caractere '{rotulo}' na posição {idx+1} salvo em: {caminho_saida}")


            elif escolha == '4':
                pasta_entrada = 'processed'
                pasta_saida = 'processed_augmented'
                os.makedirs(pasta_saida, exist_ok=True)
                for filename in os.listdir(pasta_entrada):
                    if filename.lower().endswith('.png'):
                        caminho = os.path.join(pasta_entrada, filename)
                        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            for i in range(3):
                                aug = self.augment_image(img)
                                caminho_saida = os.path.join(pasta_saida, filename)
                                cv2.imwrite(caminho_saida, aug)
                                print(f"Imagem aumentada salva em: {caminho_saida}")

            elif escolha == '5':
                self.treinar_modelo()

            elif escolha == '6':
                print("Encerrando o programa.")
                break
            else:
                print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    CaptchaProcessor().menu()
