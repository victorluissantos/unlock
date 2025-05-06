import os
import importlib.util

def processar_decripto(modelo: str, content: str, tipo: str) -> str:
    caminho_modulo = f"datahouse/{modelo}/run.py"
    if not os.path.isfile(caminho_modulo):
        raise FileNotFoundError(f"Modelo '{modelo}' não encontrado.")

    spec = importlib.util.spec_from_file_location("run_module", caminho_modulo)
    run_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_module)

    if not hasattr(run_module, "decripto"):
        raise AttributeError("Função 'decripto' não encontrada no módulo.")

    # Chama run_module.decripto com o content já preparado
    return run_module.decripto(content, tipo)
