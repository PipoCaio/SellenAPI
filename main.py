from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import xgboost as xgb
import pandas as pd
import pdfplumber
import requests
import io
from datetime import datetime

app = FastAPI(
    title="SellenAPI",
    description="Carregue algumas contas da Equatorial em PDF para uma previsão do próximo mês.",
    version="3.1.0"
)

modelo = xgb.Booster()
modelo.load_model("xgboost_otimizado.json")

MESES_MAP = {
    "JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4, "MAI": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12
}

def obter_temperatura_maceio(mes_alvo: int) -> float:
    """
    Busca a temperatura média diretamente da API da NASA
    para as coordenadas de Maceió (-9.6658, -35.7353).
    """
    # NUNCA MAIS EU CONFIO 100% NA API DA NASA
    medias_maceio_backup = {
        1: 27.2, 2: 27.4, 3: 27.2, 4: 26.7, 5: 25.9, 6: 25.0,
        7: 24.5, 8: 24.5, 9: 25.2, 10: 26.2, 11: 26.8, 12: 27.1
    }

    url = "https://power.larc.nasa.gov/api/temporal/climatology/point"
    
    params = {
        "parameters": "T2M",       
        "community": "RE",         # É MAIS EFICIENTE POR ALGUM MOTIVO
        "longitude": -35.7353,
        "latitude": -9.6658,
        "format": "JSON"
    }
    
    try:
        resposta = requests.get(url, params=params, timeout=3)
        
        if resposta.status_code == 200:
            dados = resposta.json()
            temperaturas_nasa = dados["properties"]["parameter"]["T2M"]
            temp_mes = temperaturas_nasa.get(str(mes_alvo))
            
            if temp_mes is not None:
                print(f"[NASA API] Temperatura para o mês {mes_alvo} capturada via Satélite Mt poderoso e caro: {temp_mes}°C")
                return float(temp_mes)
                
    except Exception as e:
        print(f"Falha na conexão com a NASA ({e}). Ativando contingência local.")

    return medias_maceio_backup.get(mes_alvo, 26.0)

def subtrair_meses(data_base: datetime, meses_a_subtrair: int) -> datetime:
    """
    Auxiliar para voltar no tempo mês a mês de forma segura.
    """
    ano = data_base.year
    mes = data_base.month - meses_a_subtrair
    while mes < 1:
        mes += 12
        ano -= 1
    return datetime(ano, mes, 1)

@app.post("/prever-via-pdf")
async def prever_via_pdf(arquivos: List[UploadFile] = File(...)):
    if len(arquivos) < 1:
        raise HTTPException(status_code=400, detail="Envie ao menos 1 fatura em PDF contendo o histórico.")

    historico_unificado = {}
    for arquivo in arquivos:
        if not arquivo.filename.lower().endswith(".pdf"):
            continue
        conteudo_bytes = await arquivo.read() #n tem pq pegar tudo no armaz
        
        try:
            with pdfplumber.open(io.BytesIO(conteudo_bytes)) as pdf:
                pagina = pdf.pages[0]
                caixa_historico = (440, 317, 592, 464)
                pag_cortada = pagina.within_bbox(caixa_historico)
                texto = pag_cortada.extract_text()
                
                if texto:
                    for line in texto.strip().splitlines():
                        parts = line.strip().split(maxsplit=1)
                        if len(parts) == 2:
                            mes_ano_texto, valor_consumo = parts
                            try:
                                m_str, a_str = mes_ano_texto.split("/")#FUNCIONOU NÃO TIRA
                                m_num = MESES_MAP[m_str.upper()]#FUNCIONOU NÃO MEXE
                                ano_num = int(f"20{a_str}")#FUNCIONOU NEM ENCOSTA
                                
                                data_objeto = datetime(ano_num, m_num, 1)
                                historico_unificado[data_objeto] = int(valor_consumo)
                            except Exception:
                                continue
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao processar o arquivo {arquivo.filename}: {str(e)}")

    if not historico_unificado:
        raise HTTPException(status_code=400, detail="Não foi possível extrair dados válidos da tabela de histórico das faturas.")

    ultima_data_historico = max(historico_unificado.keys())
    if ultima_data_historico.month == 12:
        data_previsao = datetime(ultima_data_historico.year + 1, 1, 1)
    else:
        data_previsao = datetime(ultima_data_historico.year, ultima_data_historico.month + 1, 1)

    mes_anterior_dt = ultima_data_historico
    dois_meses_atras_dt = subtrair_meses(data_previsao, 2)
    tres_meses_atras_dt = subtrair_meses(data_previsao, 3)
    ano_passado_dt = datetime(data_previsao.year - 1, data_previsao.month, 1)

    try:
        consumo_mes_anterior = historico_unificado[mes_anterior_dt]
        consumo_2_meses = historico_unificado.get(dois_meses_atras_dt, consumo_mes_anterior)
        consumo_3_meses = historico_unificado.get(tres_meses_atras_dt, consumo_mes_anterior)
        
        consumo_ano_passado = historico_unificado.get(ano_passado_dt, consumo_mes_anterior)
        media_3_meses = (consumo_mes_anterior + consumo_2_meses + consumo_3_meses) / 3.0
    except KeyError:
        raise HTTPException(status_code=400, detail="Histórico insuficiente nos PDFs para calcular os atrasos de memória requeridos.")

    mes_alvo = data_previsao.month
    estacao_quente = 1 if mes_alvo in [11, 12, 1, 2, 3] else 0
    temp_maceio = obter_temperatura_maceio(mes_alvo)

    features_calculadas = {
        "Mes": mes_alvo,
        "Consumo_Mes_Anterior": float(consumo_mes_anterior),
        "Consumo_Ano_Passado": float(consumo_ano_passado),
        "Media_Ultimos_3_Meses": float(media_3_meses),
        "Estacao_Quente": int(estacao_quente),
        "Temp_Real_Maceio": float(temp_maceio)
    }

    dados_df = pd.DataFrame([features_calculadas])
    if modelo.feature_names:
        dados_df = dados_df[modelo.feature_names]#ele consegue pegar o json agr
        
    dados_api = xgb.DMatrix(dados_df)
    predicao = modelo.predict(dados_api)

    return {
        "status": "sucesso",
        "mes_da_previsao": f"{mes_alvo}/{data_previsao.year}",
        "features_extraidas_dos_pdfs": features_calculadas,
        "previsao_consumo_kwh": float(predicao[0])
    }