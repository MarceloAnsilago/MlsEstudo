import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from supabase import create_client, Client
import time
import random

# === CONFIGURAÇÕES ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("🔐 Variáveis de ambiente SUPABASE_URL e SUPABASE_KEY não configuradas")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === PARÂMETROS ===
PERIODO_DIAS = 5  # Últimos dias de cotações a buscar
MERCADO_SUFFIX = ".SA"  # Sufixo do Yahoo Finance para ações brasileiras

# === PASSO 1: Buscar ações cadastradas ===
res = supabase.table("acoes").select("acao").execute()
tickers_raw = [row["acao"] for row in res.data]

if not tickers_raw:
    print("⚠️ Nenhuma ação cadastrada na tabela 'acoes'")
    exit()

# === PASSO 2: Formatando tickers ===
tickers_yahoo = [f"{ticker}{MERCADO_SUFFIX}" if not ticker.endswith(MERCADO_SUFFIX) else ticker for ticker in tickers_raw]

# === PASSO 3: Coletar e inserir cotações ===
sucesso, falha = 0, 0

for ticker in tickers_yahoo:
    print(f"📥 Baixando {ticker}...")
    try:
        df = yf.download(ticker, period=f"{PERIODO_DIAS}d", auto_adjust=True, progress=False)

        if df.empty:
            print(f"❌ Nenhuma cotação para {ticker}")
            falha += 1
        else:
            for index, row in df.iterrows():
                payload = {
                    "ticker": ticker.replace(MERCADO_SUFFIX, ""),
                    "data": index.date().isoformat(),
                    "preco_fechamento": float(row["Close"].item()),
                }
                supabase.table("cotacoes").upsert(payload, on_conflict=["ticker", "data"]).execute()

            sucesso += 1

        # Aguarda entre 3 e 6 segundos para evitar rate limit
        time.sleep(random.uniform(3, 6))

    except Exception as e:
        print(f"🚫 Erro ao processar {ticker}: {e}")
        falha += 1

# === RESULTADO FINAL ===
print(f"\n✅ Coleta finalizada: {sucesso} tickers com sucesso, {falha} falharam.")