# utils/supabase_client.py
from supabase import create_client, Client
from dotenv import load_dotenv
import streamlit as st
import os

# Carrega variáveis locais do .env
load_dotenv()

def get_supabase_client() -> Client:
    """
    Cria e retorna o cliente Supabase, usando st.secrets (Cloud)
    ou variáveis de ambiente (.env local).
    """
    url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError(
            "❌ As credenciais do Supabase não foram definidas. "
            "Por favor, verifique seu arquivo .env local ou secrets no Streamlit Cloud."
        )

    return create_client(url, key)
