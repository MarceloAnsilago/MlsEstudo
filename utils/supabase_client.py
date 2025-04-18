# utils/supabase_client.py
from supabase import create_client, Client
from dotenv import load_dotenv
import os

# Carrega automaticamente o .env da raiz
load_dotenv()

def get_supabase_client() -> Client:
    """
    Cria e retorna o cliente Supabase usando variáveis de ambiente.
    Lança um RuntimeError se SUPABASE_URL ou SUPABASE_KEY não estiverem definidas.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError(
            "❌ As credenciais do Supabase não foram definidas. "
            "Por favor, verifique seu arquivo .env na raiz do projeto "
            "(SUPABASE_URL e SUPABASE_KEY)."
        )

    return create_client(url, key)
