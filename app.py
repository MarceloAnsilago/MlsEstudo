import sys
from pathlib import Path

# Garante que o diretório do projeto esteja no sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st

# Deve ser a PRIMEIRA linha
st.set_page_config(
    page_title="Gerenciamento de Ações",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Imports depois de set_page_config
import login
import principal

# Se estiver logado, mostra a página principal
if st.session_state.get("usuario"):
    principal.render()
else:
    login.render()
