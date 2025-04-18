
from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()  # Carrega variáveis do .env automaticamente
def get_supabase_client() -> Client:
    """
    Cria e retorna o cliente Supabase usando variáveis de ambiente
    ou valores diretos definidos aqui.
    """
    SUPABASE_URL = os.getenv("SUPABASE_URL", "https://dtghhtizyirativfkmoo.supabase.co")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR0Z2hodGl6eWlyYXRpdmZrbW9vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDU4MzgwMCwiZXhwIjoyMDYwMTU5ODAwfQ.l0SsoNc9xTsnS70z6EbCIcTZZDmGY5xwvY-y34WuAso")

    return create_client(SUPABASE_URL, SUPABASE_KEY)