from fastapi import APIRouter, HTTPException
from schemas.decripto import DecriptoRequest
from services.decripto_service import processar_decripto
import requests
import base64

router = APIRouter()

@router.post("/decripto")
async def decripto_endpoint(req: DecriptoRequest):
    # 1) Se o usuário enviou 'url', baixar o recurso primeiro
    if req.url:
        try:
            resp = requests.get(str(req.url), timeout=10)
            resp.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao baixar da URL: {e}")

        # 2) Montar req.content conforme o tipo
        if req.type == "base64":
            # URL que retorna string Base64 pura ou "data:image..."
            req.content = resp.text.strip()
        else:
            # URL que retorna bytes de imagem
            b64 = base64.b64encode(resp.content).decode()
            req.content = f"data:image/png;base64,{b64}"

    # 3) Agora req.content está sempre populado, só chamar o serviço
    try:
        resultado = processar_decripto(req.modelo, req.content, req.type)
        return {"resultado": resultado}
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))