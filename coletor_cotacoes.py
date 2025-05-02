import os
import yfinance as yf
import pandas as pd
import time
from datetime import datetime
from supabase import create_client, Client

# === CONFIGURAÇÕES ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("🔐 Variáveis de ambiente SUPABASE_URL e SUPABASE_KEY não configuradas")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === PARÂMETROS ===
PERIODO_DIAS = 5
MERCADO_SUFFIX = ".SA"

# === PASSO 1: Buscar ações cadastradas ===
res = supabase.table("acoes").select("acao").execute()
tickers_raw = [row["acao"] for row in res.data]

if not tickers_raw:
    print("⚠️ Nenhuma ação cadastrada na tabela 'acoes'")
    exit()

# === PASSO 2: Formatando tickers ===
tickers_yahoo = [f"{ticker}{MERCADO_SUFFIX}" if not ticker.endswith(MERCADO_SUFFIX) else ticker for ticker in tickers_raw]

# Limitar para evitar rate limit no GitHub Actions (opcional, pode remover depois)
# tickers_yahoo = tickers_yahoo[:5]

# === PASSO 3: Coletar e inserir cotações ===
sucesso, falha = 0, 0

for ticker in tickers_yahoo:
    print(f"\n📥 Baixando {ticker} com período {PERIODO_DIAS}d...")
    try:
        df = yf.download(ticker, period=f"{PERIODO_DIAS}d", auto_adjust=True, progress=False)

        if df.empty:
            print(f"❌ Nenhuma cotação encontrada para {ticker}")
            print(df)
            falha += 1
            continue

        for index, row in df.iterrows():
            payload = {
                "ticker": ticker.replace(MERCADO_SUFFIX, ""),
                "data": index.date().isoformat(),
                "preco_fechamento": float(row["Close"]),
            }

            response = supabase.table("cotacoes").upsert(payload, on_conflict=["ticker", "data"]).execute()
            print(f"📝 Supabase resposta para {payload['ticker']} em {payload['data']}: {response}")

        sucesso += 1

    except Exception as e:
        print(f"🚫 Erro ao processar {ticker}: {e}")
        falha += 1

    time.sleep(2)  # Espera entre as chamadas para evitar bloqueios (ajustável)

# === RESULTADO FINAL ===
print(f"\n✅ Coleta finalizada: {sucesso} tickers com sucesso, {falha} falharam.")
