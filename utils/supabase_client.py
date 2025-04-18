
from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()  # Carrega variáveis do .env automaticamente
def get_supabase_client() -> Client:
    """
    Cria e retorna o cliente Supabase usando variáveis de ambiente
    ou valores diretos definidos aqui.
    """
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    return create_client(SUPABASE_URL, SUPABASE_KEY)