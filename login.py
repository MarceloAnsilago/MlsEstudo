import streamlit as st
from pathlib import Path
from PIL import Image
from utils.supabase_client import get_supabase_client

def render():
    # ✅ Sidebar escondida apenas nesta tela
    st.markdown("""
        <style>
            [data-testid="stSidebar"], [data-testid="stSidebarNav"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Se o usuário já estiver logado, volta pro app.py redirecionar para principal
    if st.session_state.get("usuario"):
        return

    # ✅ Logo (opcional)
    logo_path = Path("logos/sua_logo.png")
    if logo_path.exists():
        st.image(Image.open(logo_path), width=150)

    st.markdown("## 🔐 Login ou Cadastro")
    supabase = get_supabase_client()
    abas = st.tabs(["Login", "Criar Conta"])

    # ================
    # 🔐 ABA LOGIN
    # ================
    with abas[0]:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.subheader("Entrar")

            nome_login = st.text_input("Nome de Usuário", key="login_nome").strip().lower()
            senha_login = st.text_input("Senha", type="password", key="login_senha").strip()

            if st.button("Entrar"):
                try:
                    data = supabase.table("usuarios").select("*") \
                        .eq("nome", nome_login).eq("senha", senha_login).execute()

                    if data.data:
                        st.session_state["usuario"] = {
                            "id": data.data[0]["id"],
                            "nome": data.data[0]["nome"]
                        }
                        st.success("✅ Login realizado com sucesso!")
                        st.rerun()
                    else:
                        st.error("❌ Nome ou senha incorretos.")
                except Exception as e:
                    st.error(f"Erro ao tentar logar: {e}")

    # ================
    # 🆕 ABA CADASTRO
    # ================
    with abas[1]:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.subheader("Criar Conta")

            nome_novo = st.text_input("Novo Nome de Usuário", key="cadastro_nome").strip().lower()
            senha_nova = st.text_input("Nova Senha", type="password", key="cadastro_senha").strip()

            if st.button("Criar Conta"):
                if nome_novo and senha_nova:
                    try:
                        check = supabase.table("usuarios").select("id") \
                            .eq("nome", nome_novo).execute()

                        if check.data:
                            st.warning("⚠️ Nome de usuário já existe.")
                        else:
                            supabase.table("usuarios").insert({
                                "nome": nome_novo,
                                "senha": senha_nova
                            }).execute()
                            st.success("✅ Conta criada com sucesso! Faça login.")
                    except Exception as e:
                        st.error(f"Erro ao criar conta: {e}")
                else:
                    st.warning("Preencha nome e senha.")
