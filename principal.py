import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridOptionsBuilder
import yfinance as yf
import base64
import os
import plotly.graph_objects as go
from utils.supabase_client import get_supabase_client
from datetime import datetime, time as dt_time
from PIL import Image
from statsmodels.tsa.stattools import coint
import numpy as np
import statsmodels.api as sm
from hurst import compute_Hc
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
import mplfinance as mpf
from io import BytesIO
from pathlib import Path
from streamlit_extras.switch_page_button import switch_page


# Função para carregar o ícone
def carregar_icone(ticker):
    # Verifica o formato do ticker e ajusta o nome do arquivo
    if len(ticker) >= 5 and ticker[4].isdigit():
        ticker_base = ticker[:4]
    else:
        ticker_base = ticker.replace(".SA", "")

    # Caminho do arquivo do ícone
    icon_path = f"logos/{ticker_base}.jpg"

    if os.path.exists(icon_path):
        try:
            with open(icon_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
                return f"data:image/jpg;base64,{encoded_string}"
        except Exception as e:
            return None
    else:
        return None
def operacao_ja_existe(supabase, usuario_id, ativo1, ativo2):
    try:
        response = (
            supabase.table("operacoes")
            .select("ativo_venda, ativo_compra")
            .eq("usuario_id", usuario_id)
            .eq("status", "aberta")
            .execute()
        )
        if not response.data:
            return False

        for op in response.data:
            venda = op["ativo_venda"]
            compra = op["ativo_compra"]

            # Verifica par direto ou invertido
            if {venda, compra} == {ativo1, ativo2}:
                return True
        return False
    except Exception as e:
        st.error(f"Erro ao verificar operações existentes: {e}")
        return True  # Por segurança, bloqueia operação se falhar


def plot_candlestick_e_volume(serie_preco, nome_ativo):
    df = pd.DataFrame(serie_preco.tail(50)).copy()
    df.columns = ['Close']
    df['Open'] = df['Close'].shift(1)
    df['High'] = df[['Close', 'Open']].max(axis=1) * 1.01
    df['Low'] = df[['Close', 'Open']].min(axis=1) * 0.99
    df.dropna(inplace=True)
    df['Volume'] = np.random.randint(1000, 5000, size=len(df))

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=[f"{nome_ativo} — Candlestick", "Volume"]
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlestick"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        marker_color='lightblue',
        name="Volume"
    ), row=2, col=1)

    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)


def mostrar_fluxo_liquido(venda_total: float, compra_total: float) -> float:
        """
        Exibe o fluxo financeiro líquido da operação de long & short.

        Retorna o valor do fluxo líquido para uso posterior.
        """
        resultado_total = venda_total - compra_total

        mensagem = f"**Fluxo Líquido da Operação: R$ {resultado_total:.2f}**"
        explicacao = (
            "Este valor representa o fluxo financeiro inicial da operação de long & short. "
            "Se for positivo, você recebe esse montante ao montar a estratégia. "
            "Se for negativo, você precisa investir esse valor para abrir as posições."
        )

        if resultado_total >= 0:
            st.success(mensagem)
        else:
            st.error(mensagem)

        st.markdown(f"<p style='font-size: 14px; color: #666;'>{explicacao}</p>", unsafe_allow_html=True)

        return resultado_total
def plotar_beta_movel(S1, S2, window=40):
                try:
                    returns_S1 = np.log(S1 / S1.shift(1)).dropna()
                    returns_S2 = np.log(S2 / S2.shift(1)).dropna()

                    betas = []
                    index_values = returns_S1.index[window - 1:]  # Ajustar para a janela

                    for i in range(window, len(returns_S1) + 1):
                        reg = LinearRegression().fit(
                            returns_S2[i - window:i].values.reshape(-1, 1),
                            returns_S1[i - window:i].values
                        )
                        betas.append(reg.coef_[0])

                    beta_movel = pd.Series(betas, index=index_values)

                    # Plotar o gráfico de beta móvel
                    plt.figure(figsize=(10, 5))
                    plt.plot(beta_movel, label=f'Beta Móvel ({window} períodos)')
                    plt.axhline(0, color='black', linestyle='--')
                    plt.title(f'Beta Móvel ({window} períodos)')
                    plt.xlabel('Data')
                    plt.ylabel('Beta')
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(plt)
                except Exception as e:
                    st.error(f"Erro ao calcular ou plotar o beta móvel: {e}")    

def get_base64(file_path):
        with open(os.path.abspath(file_path), "rb") as f:
            return base64.b64encode(f.read()).decode()

        image_base64 = get_base64("logos/cotacoes.png") 

# Função para calcular Half-Life
def half_life_calc(ts):
    lagged = ts.shift(1).fillna(method="bfill")
    delta = ts - lagged
    X = sm.add_constant(lagged.values)
    ar_res = sm.OLS(delta, X).fit()
    half_life = -1 * np.log(2) / ar_res.params[1]
    return half_life

# Função para calcular Hurst exponent
def hurst_exponent(ts):
    H, c, data = compute_Hc(ts, kind='price', simplified=True)
    return H

# Função para calcular Beta Rotation (ang. cof)
def beta_rotation(series_x, series_y, window=40):
    beta_list = []
    try:
        for i in range(0, len(series_x) - window):
            slice_x = series_x[i:i + window]
            slice_y = series_y[i:i + window]
            X = sm.add_constant(slice_x.values)
            mod = sm.OLS(slice_y, X)
            results = mod.fit()
            beta = results.params[1]
            beta_list.append(beta)
    except Exception as e:
        st.error(f"Erro ao calcular beta rotation: {e}")
        raise

    return beta_list[-1]  # Return the most recent beta value

# Função para calcular o beta móvel em uma janela deslizante
def calcular_beta_movel(S1, S2, window=40):
    returns_S1 = np.log(S1 / S1.shift(1)).dropna()
    returns_S2 = np.log(S2 / S2.shift(1)).dropna()

    betas = []
    index_values = returns_S1.index[window-1:]  # Ajustar para a janela

    for i in range(window, len(returns_S1) + 1):
        reg = LinearRegression().fit(returns_S2[i-window:i].values.reshape(-1, 1), returns_S1[i-window:i].values)
        betas.append(reg.coef_[0])

    return pd.Series(betas, index=index_values)

# Exibir o gráfico de beta móvel
def plotar_beta_movel(S1, S2, window=40):
    try:
        returns_S1 = np.log(S1 / S1.shift(1)).dropna()
        returns_S2 = np.log(S2 / S2.shift(1)).dropna()

        betas = []
        index_values = returns_S1.index[window - 1:]  # Ajustar para a janela

        for i in range(window, len(returns_S1) + 1):
            reg = LinearRegression().fit(
                returns_S2[i - window:i].values.reshape(-1, 1),
                returns_S1[i - window:i].values
            )
            betas.append(reg.coef_[0])

        beta_movel = pd.Series(betas, index=index_values)

        # Plotar o gráfico de beta móvel
        plt.figure(figsize=(10, 5))
        plt.plot(beta_movel, label=f'Beta Móvel ({window} períodos)')
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f'Beta Móvel ({window} períodos)')
        plt.xlabel('Data')
        plt.ylabel('Beta')
        plt.legend()
        plt.xticks(rotation=45, fontsize=6)
        plt.grid(True)

        # Reduzir a quantidade de rótulos no eixo X
        ax = plt.gca()             # Pega o axis atual
        ticks = ax.get_xticks()    # Pega os ticks atuais
        ax.set_xticks(ticks[::5])  # Exibe somente 1 a cada 5

        st.pyplot(plt)
    except Exception as e:
        st.error(f"Erro ao calcular ou plotar o beta móvel: {e}")


# Exibir o gráfico de dispersão entre os dois ativos
def plotar_grafico_dispersao(S1, S2):
    plt.figure(figsize=(10, 5))
    plt.scatter(S1, S2)
    plt.title(f'Dispersão entre {S1.name} e {S2.name}')
    plt.xlabel(f'{S1.name}')
    plt.ylabel(f'{S2.name}')
    plt.grid(True)
    st.pyplot(plt)


def obter_preco_atual(ticker):
    dados = yf.download(ticker, period="1d")  # Baixar o dado mais recente
    if not dados.empty:
        return dados['Close'].iloc[-1]  # Retornar o preço de fechamento mais recente
    else:
        return None
# Função para plotar o gráfico do Z-Score
def plotar_grafico_zscore(S1, S2):
    ratios = S1 / S2
    zscore_series = (ratios - ratios.mean()) / ratios.std()

    plt.figure(figsize=(10, 5))
    plt.plot(zscore_series, label='Z-Score')
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(2, color='red', linestyle='--')
    plt.axhline(-2, color='green', linestyle='--')
    plt.legend(loc='best')
    plt.xlabel('Data')
    plt.ylabel('Z-Score')
    plt.xticks(rotation=45, fontsize=6)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    st.pyplot(plt)

    # Função para plotar o gráfico dos preços das ações
def plotar_grafico_precos(S1, S2, ticker1, ticker2):
    plt.figure(figsize=(10, 5))
    plt.plot(S1, label=ticker1)
    plt.plot(S2, label=ticker2)
    plt.legend(loc='best')
    plt.xlabel('Data')
    plt.ylabel('Preço de Fechamento')
    plt.xticks(rotation=45, fontsize=6)
    st.pyplot(plt)


# Função para encontrar pares cointegrados e calcular z-score, half-life, Hurst, ang. cof
def find_cointegrated_pairs(data, zscore_threshold_upper, zscore_threshold_lower):
    n = data.shape[1]
    keys = data.keys()
    pairs = []
    pvalues = []
    zscores = []
    half_lives = []
    hursts = []
    beta_rotations = []

    for i in range(n):
        for j in range(i + 1, n):
            S1 = data[keys[i]].dropna()  # Remover NaNs de S1
            S2 = data[keys[j]].dropna()  # Remover NaNs de S2

            # Garantir que ambas as séries tenham o mesmo comprimento após a remoção dos NaNs
            combined = pd.concat([S1, S2], axis=1).dropna()
            if len(combined) < 2:  # Verificar se ainda há dados suficientes
                continue

            S1 = combined.iloc[:, 0]
            S2 = combined.iloc[:, 1]

            try:
                score, pvalue, _ = coint(S1, S2)
                if pvalue < 0.05:
                    ratios = S1 / S2
                    zscore = (ratios - ratios.mean()) / ratios.std()

                    if zscore.iloc[-1] > zscore_threshold_upper or zscore.iloc[-1] < zscore_threshold_lower:
                        pairs.append((keys[i], keys[j]))
                        pvalues.append(pvalue)
                        zscores.append(zscore.iloc[-1])
                        half_lives.append(half_life_calc(ratios))
                        hursts.append(hurst_exponent(ratios))
                        beta_rotations.append(beta_rotation(S1, S2))

            except Exception as e:
                print(f"Erro ao calcular cointegração para {keys[i]} e {keys[j]}: {e}")
                continue

    return pairs, pvalues, zscores, half_lives, hursts, beta_rotations
# Função para criar cards explicativos das métricas
def criar_card_metrica(nome_metrica, valor_metrica, descricao):
    st.markdown(
        f"""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 10px; text-align: center; background-color: #f9f9f9; height: 250px; margin-bottom: 15px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <h4 style="margin: 0;">{nome_metrica}</h4>
                <hr style="border: none; border-top: 2px solid red; margin: 5px 0 10px 0;">
            </div>
            <div style="flex-grow: 1; display: flex; justify-content: center; align-items: center;">
                <h2 style="margin: 0; font-size: 24px;">{valor_metrica}</h2>
            </div>
            <div style="margin-top: 10px; text-align: center;">
                <p style="font-size: 14px; color: #888; margin-bottom: 4px;">{descricao}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def exibir_metrica_cartao(ticker, ultimo_preco, ultima_data, icone=None):
    icone_html = (
        f'<img src="{icone}" style="max-width: 100px; max-height: 100px; object-fit: contain;">'
        if icone else '<p style="color: red;">Sem Ícone</p>'
    )

    st.markdown(
        f"""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 10px; text-align: center; background-color: #f9f9f9; height: 300px; margin-bottom: 15px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <h2 style="margin: 0;">{ticker}</h2> <!-- Aumentei a fonte do título -->
                <hr style="border: none; border-top: 2px solid red; margin: 5px 0 10px 0;">
            </div>
            <div style="flex-grow: 1; display: flex; justify-content: center; align-items: center;">
                {icone_html} <!-- Ícone com altura ajustada -->
            </div>
            <div style="margin-top: 10px; text-align: center;">
                <h6 style="font-size: 14px; color: #888; margin-bottom: 4px;">Última Cotação ({ultima_data})</h6>
                <h3 style="margin: 0; font-size: 24px;">R$ {ultimo_preco:.2f}</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def format_metric(value, default="N/A", format_str="{:.2f}"):
    if pd.isna(value):
        return default
    return format_str.format(value)


def fetch_from_yfinance(ticker, period):
    try:
        df = yf.download(ticker, period=period, threads=False, progress=False)['Close']
        if not df.empty:
            return df
    except Exception as e:
        print(f"[yfinance] {ticker} falhou: {e}")
    return None




def render():
    if "usuario" not in st.session_state:
        st.warning("⚠️ Faça login primeiro.")
        st.stop()

    if "global_cotacoes" not in st.session_state:
        st.session_state["global_cotacoes"] = pd.DataFrame()

    usuario = st.session_state["usuario"]
    st.sidebar.markdown(f"👤 Usuário: `{usuario['nome']}`")

    if st.sidebar.button("🔒 Sair"):
        st.session_state.clear()
        st.rerun()

    # ✅ Definir a imagem base64 usada nos cabeçalhos das abas
    image_base64 = get_base64("logos/Ações_Acompanhadas.png")

    # Menu Lateral
    with st.sidebar:
    
        logo_path = Path("logos/LogoApp.png")
        if logo_path.exists():
            logo_image = Image.open(logo_path)
            st.image(logo_image, use_container_width=True)

    
      
        selected = option_menu(
            menu_title="Menu Principal",  # required
            options=["Página Inicial", "Cotações", "Análise", "Operações", "Encerradas", "Backtesting"],  # <-- adicionada "Encerradas"
            icons=["house", "currency-exchange", "graph-up-arrow", "briefcase", "archive", "clock-history"],  # <-- adiciona ícone
            menu_icon="cast",  # ícone do menu
            default_index=0,  # seleciona a aba 'Página Inicial'
        )


    st.markdown(f"## Bem-vindo, {usuario['nome']}!")

    # Aba "Ações Acompanhadas"
    if selected == "Página Inicial":
        # st.title("Ações Acompanhadas")
        if st.session_state["global_cotacoes"].empty:
            st.info("Nenhuma cotação carregada. Por favor, carregue as cotações na aba 'Cotações'.")        
        
 

        image_base64 = get_base64("logos/Ações_Acompanhadas.png")

        st.markdown(
            f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{image_base64}" alt="Logo" style="height: 130px; margin-right: 10px;">
                <h1 style="margin: 0;">Ações Acompanhadas</h1>
            </div>
            """,
            unsafe_allow_html=True
        )


        # Verificar se o DataFrame global tem dados
        if "global_cotacoes" in st.session_state and not st.session_state["global_cotacoes"].empty:
            # DataFrame global com as cotações
            cotacoes_df = st.session_state["global_cotacoes"].copy()

            # Verificar se o índice é chamado 'Date' e transformá-lo em coluna
            if "Date" not in cotacoes_df.columns:
                cotacoes_df.reset_index(inplace=True)

            # Garantir que a coluna "Date" exista
            if "Date" in cotacoes_df.columns:
                # Preparar os dados para exibição (última cotação de cada ticker)
                cotacoes_df = cotacoes_df.melt(id_vars=["Date"], var_name="ticker", value_name="fechamento")
                cotacoes_df = cotacoes_df.dropna().sort_values(by=["ticker", "Date"], ascending=[True, False])
                cotacoes_df = cotacoes_df.groupby("ticker").first().reset_index()

                # Exibir as métricas em 5 colunas com espaçamento de 5px
                cols = st.columns(5, gap="small")

                for index, row in cotacoes_df.iterrows():
                    ticker = row['ticker']
                    ultimo_preco = row['fechamento']
                    ultima_data = row['Date']

                    # **Carregar ícone do ticker**
                    icone = carregar_icone(ticker)

                    # Exibir a métrica no formato de cartão
                    with cols[index % 5]:
                        exibir_metrica_cartao(ticker, ultimo_preco, ultima_data, icone)
            else:
                st.error("A coluna de datas ('Date') não foi encontrada no DataFrame.")
        else:
            st.warning("Nenhuma cotação carregada. Por favor, carregue as cotações na aba 'Cotações'.")




  


# ======================
# Aba de Cotações
# ======================
    if selected == "Cotações":
        supabase = get_supabase_client()
        usuario_id = st.session_state["usuario"]["id"]
        cotacoes_base64 = get_base64("logos/cotacoes.png")  # Caminho da imagem

        st.markdown(
                f"""
                <div style="display: flex; align-items: center;">
                    <img src="data:image/png;base64,{cotacoes_base64}"
                        alt="cotacoes"
                        style="height: 125px; margin-right: 20px;">
                    <h1 style="margin: 0;">Cotações</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        # === LAYOUT CENTRALIZADO ===
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            aba_cotacoes = st.tabs([ "📈 Carregar Cotações","📥 Cadastrar Ações"])


        # === 📥 Cadastrar Ações ===
        with aba_cotacoes[0]:
            st.markdown("### Buscar Cotações de Ações")
            
            with st.spinner("Carregando ações cadastradas..."):
                resultado = supabase.table("acoes").select("*").eq("usuario_id", usuario_id).execute()
                acoes = sorted({row['acao'] for row in resultado.data})

            if not acoes:
                st.warning("Você ainda não cadastrou nenhuma ação.")
            else:
                selecionadas = st.multiselect("Selecione as ações:", options=acoes, default=acoes)

                # Input para número de períodos
                num_periodos = st.number_input(
                    "Número de Períodos (em dias):", min_value=1, max_value=365, value=200, step=1
                )
         
                if st.button("📥 Baixar Cotações"):
                    if selecionadas:
                        tickers_yahoo = [f"{ticker}.SA" if not ticker.endswith(".SA") else ticker for ticker in selecionadas]
                        new_cotacoes = pd.DataFrame()

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for idx, ticker in enumerate(tickers_yahoo):
                            status_text.text(f"Buscando cotações para {ticker} via Yahoo Finance...")

                            dados = fetch_from_yfinance(ticker, f"{num_periodos}d")

                            if dados is not None and not dados.empty:
                                dados.name = ticker.replace(".SA", "")
                                new_cotacoes = pd.concat([new_cotacoes, dados], axis=1)
                            else:
                                st.warning(f"Nenhuma cotação encontrada para {ticker}")

                            progress_bar.progress((idx + 1) / len(tickers_yahoo))

                        status_text.empty()
                        progress_bar.empty()

                        if not new_cotacoes.empty:
                            new_cotacoes.index = new_cotacoes.index.strftime("%Y-%m-%d")
                            new_cotacoes.index.name = "Date"

                            st.session_state["global_cotacoes"] = new_cotacoes

                            st.success("✅ Cotações carregadas com sucesso!")
                            st.dataframe(new_cotacoes.reset_index(), use_container_width=True)
                        else:
                            st.warning("Nenhuma cotação foi encontrada para os tickers selecionados.")


                        # === 📈 Carregar Cotações ===
        with aba_cotacoes[1]:

            nova_acao = st.text_input("Digite o ticker da ação (ex: PETR4)")
            
            if st.button("Cadastrar Ação"):
                if nova_acao:
                    supabase.table("acoes").insert({
                        "acao": nova_acao.upper(),
                        "usuario_id": usuario_id
                    }).execute()
                    st.success(f"Ação {nova_acao.upper()} cadastrada com sucesso!")
                else:
                    st.warning("Por favor, digite um ticker válido.")
            st.markdown("### Buscar Cotações de Ações")
         

    if selected == "Análise":
    
        analise_b64 = get_base64("logos/analise.png")
        st.markdown(
            f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{analise_b64}"
                    alt="Análise"
                    style="height: 125px; margin-right: 25px;">
                <h1 style="margin: 0;">Análise</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Seleção de parâmetros para análise
        with st.form(key='analysis_form'):
            numero_periodos = st.number_input(
                "Número de Períodos para Análise",
                min_value=1,
                value=120,
                help="Número de períodos (mais recentes) para considerar na análise de cointegração."
            )
            zscore_threshold_upper = st.number_input("Limite Superior do Z-Score", value=2.0)
            zscore_threshold_lower = st.number_input("Limite Inferior do Z-Score", value=-2.0)
            submit_button = st.form_submit_button(label="🛠 Analisar Pares Cointegrados")

        if submit_button or 'cotacoes_pivot' in st.session_state:
            if submit_button:
                if "global_cotacoes" not in st.session_state or st.session_state["global_cotacoes"].empty:
                    st.error("Nenhuma cotação carregada. Por favor, carregue as cotações antes de realizar a análise.")
                    st.stop()

                cotacoes_df = st.session_state["global_cotacoes"]
                if "Date" in cotacoes_df.columns:
                    cotacoes_df.set_index("Date", inplace=True)

                cotacoes_pivot = cotacoes_df.tail(numero_periodos)
                st.session_state['cotacoes_pivot'] = cotacoes_pivot

            cotacoes_pivot = st.session_state['cotacoes_pivot']
            st.write(f"Número de períodos selecionados para análise: {cotacoes_pivot.shape[0]}")

            st.subheader("Pares Encontrados")

            # Encontrar pares cointegrados e calcular métricas
            # Spinner enquanto os pares são analisados
            with st.spinner("🔍 Analisando cointegração entre os ativos..."):
                pairs, pvalues, zscores, half_lives, hursts, beta_rotations = find_cointegrated_pairs(
                    cotacoes_pivot, zscore_threshold_upper, zscore_threshold_lower
                )

            if pairs:
                for idx, (pair, zscore, pvalue, hurst, beta, half_life) in enumerate(zip(pairs, zscores, pvalues, hursts, beta_rotations, half_lives)):
                    par_str = f"{pair[0]} - {pair[1]}"
                    metricas_str = f"Z-Score: {zscore:.2f} | P-Value: {pvalue:.4f} | Hurst: {hurst:.4f} | Beta: {beta:.4f} | Half-Life: {half_life:.2f}"
                    
                    if st.button(f"{par_str} | {metricas_str}", key=f"btn_{idx}"):
                        st.session_state['par_selecionado'] = pair

                if 'par_selecionado' in st.session_state:
                    pair_selected = st.session_state['par_selecionado']
                    S1 = cotacoes_pivot[pair_selected[0]]
                    S2 = cotacoes_pivot[pair_selected[1]]
                    ratios = S1 / S2
                    zscore_series = (ratios - ratios.mean()) / ratios.std()

                    st.markdown("---")
                    st.markdown(f"<h4 style='text-align: center;'>{pair_selected[0]} - {pair_selected[1]}</h4>", unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Z-Score do Par")
                        # Cria a figura e o eixo
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(zscore_series, label='Z-Score')
                        ticks = ax.get_xticks()  
                        ax.set_xticks(ticks[::5])
                        ax.axhline(0, color='black', linestyle='--')
                        ax.axhline(2, color='red', linestyle='--')
                        ax.axhline(-2, color='green', linestyle='--')
                        ax.axhline(3, color='orange', linestyle='--', label='+3 Desvio (Stop)')
                        ax.axhline(-3, color='orange', linestyle='--', label='-3 Desvio (Stop)')
                        ax.legend(loc='best')
                        ax.set_xlabel('Data')
                        ax.set_ylabel('Z-Score')

                        # Rotaciona e diminui o tamanho das labels do eixo X
                        plt.xticks(rotation=45, fontsize=6)

                        ax.grid(True)

                        # Ajusta automaticamente o layout para evitar sobreposições
                        fig.tight_layout()

                        # Exibe no Streamlit
                        st.pyplot(fig)

                    with col2:
                        st.subheader("Cotação Normalizada")
                        fig, ax = plt.subplots(figsize=(10, 5))

                        # Plota os dois ativos normalizados
                        ax.plot(S1 / S1.iloc[0], label=f"{pair_selected[0]}")
                        ax.plot(S2 / S2.iloc[0], label=f"{pair_selected[1]}")

                        # Ajusta legendas e eixos
                        ax.legend(loc='best')
                        ax.set_xlabel('Data')
                        ax.set_ylabel('Cotação Normalizada')
                        ax.grid(True)

                        # Rotaciona e diminui a fonte das datas
                        plt.xticks(rotation=45, fontsize=6)

                        # Reduz a quantidade de rótulos no eixo X
                        ticks = ax.get_xticks()      # Pega os ticks atuais
                        ax.set_xticks(ticks[::5])    # Exibe somente 1 a cada 5

                        fig.tight_layout()
                        st.pyplot(fig)

                    col3, col4 = st.columns(2)

                    with col3:
                        st.subheader(f"Beta Móvel para {pair_selected[0]} e {pair_selected[1]}")
                        plotar_beta_movel(S1, S2, window=40)

                    with col4:
                        st.subheader(f"Dispersão entre {pair_selected[0]} e {pair_selected[1]}")
                        plotar_grafico_dispersao(S1, S2)


                # === Tabs adicionais antes do Expander ===
                # Determina os ativos antes de usar nas abas
                    current_zscore = zscores[pairs.index(pair_selected)]
                    if current_zscore > 0:
                        stock_to_sell = pair_selected[0]
                        stock_to_buy = pair_selected[1]
                    else:
                        stock_to_sell = pair_selected[1]
                        stock_to_buy = pair_selected[0]


        
                    tab1, tab2 = st.tabs(["📐 Calcular Proporção", "📎 Correlação inversa"])

                    with tab1:
                        with st.expander("📐 Cálculo da Proporção entre os Ativos", expanded=False):

                            # Dados atualizados com base nos papéis selecionados
                            preco_venda = cotacoes_pivot[stock_to_sell].iloc[-1]
                            preco_compra = cotacoes_pivot[stock_to_buy].iloc[-1]
                            ativo_venda = stock_to_sell
                            ativo_compra = stock_to_buy

                            # Colunas superiores com preços e entrada de capital
                            col_precos1, col_precos2 = st.columns(2)

                            with col_precos1:
                                st.markdown(f"### 🔻 Vender (Short): `{ativo_venda}`")
                                st.write(f"Preço atual de **{ativo_venda}**: R$ {preco_venda:.2f}")
                                capital_maximo = st.number_input(
                                    "Capital Total para Venda (R$)",
                                    min_value=100.0,
                                    value=25000.0,
                                    step=100.0,
                                    help="Limite de capital disponível para definir o máximo de lotes de venda"
                                )

                            with col_precos2:
                                st.markdown(f"### 🔺 Comprar (Long): `{ativo_compra}`")
                                st.write(f"Preço atual de **{ativo_compra}**: R$ {preco_compra:.2f}")

                            st.markdown("---")
                            st.subheader("📊 Melhor Proporção com Base no Limite de Venda")

                            melhor_resultado = None
                            max_lotes_venda = int(capital_maximo // (100 * preco_venda))

                            for lotes_venda in range(1, max_lotes_venda + 1):
                                total_venda = lotes_venda * 100 * preco_venda

                                for lotes_compra in range(1, 100):
                                    total_compra = lotes_compra * 100 * preco_compra
                                    residuo = abs(total_venda - total_compra)

                                    if melhor_resultado is None or residuo < melhor_resultado["residuo"]:
                                        melhor_resultado = {
                                            "lotes_venda": lotes_venda,
                                            "lotes_compra": lotes_compra,
                                            "total_venda": total_venda,
                                            "total_compra": total_compra,
                                            "residuo": residuo,
                                            "fluxo_liquido": total_venda - total_compra
                                        }

                            if melhor_resultado:
                                col_result1, col_result2 = st.columns(2)

                                with col_result1:
                                    st.markdown(f"### 🔻 Vender (Short): `{ativo_venda}`")
                                    st.write(f"- Lotes de 100: **{melhor_resultado['lotes_venda']}**")
                                    st.write(f"- Quantidade: **{melhor_resultado['lotes_venda'] * 100} ações**")
                                    st.write(f"- Total Venda: R$ {melhor_resultado['total_venda']:.2f}")

                                with col_result2:
                                    st.markdown(f"### 🔺 Comprar (Long): `{ativo_compra}`")
                                    st.write(f"- Lotes de 100: **{melhor_resultado['lotes_compra']}**")
                                    st.write(f"- Quantidade: **{melhor_resultado['lotes_compra'] * 100} ações**")
                                    st.write(f"- Total Compra: R$ {melhor_resultado['total_compra']:.2f}")

                                st.markdown("---")
                                fluxo = melhor_resultado['fluxo_liquido']

                                if fluxo >= 0:
                                    st.success(f"💰 Fluxo Inicial da Operação: R$ {fluxo:.2f}")
                                else:
                                    st.error(f"📉 Fluxo Inicial da Operação: R$ {fluxo:.2f}")

                                st.markdown(f"📎 Resíduo Absoluto entre os valores: R$ {melhor_resultado['residuo']:.2f}")
                            else:
                                st.warning("❗ Nenhuma combinação de lotes encontrada.")





                    with tab2:
                        st.subheader("📎 Correlação Inversa (Z-Score Invertido)")
                        st.info("📌 Este gráfico mostra o comportamento do par **invertido**, onde agora consideramos o ativo que estava sendo comprado como vendido e vice-versa.")

                        # Inversão dos ativos
                        S_inv_1 = S2  # Antes comprado
                        S_inv_2 = S1  # Antes vendido

                        stock_to_sell = pair_selected[1]  # Agora vendemos o que antes comprávamos
                        stock_to_buy = pair_selected[0]   # Compramos o que antes vendíamos

                        ratio_inv = S_inv_1 / S_inv_2
                        zscore_inv = (ratio_inv - ratio_inv.mean()) / ratio_inv.std()

                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(zscore_inv, label="Z-Score Inverso")
                        ax.axhline(0, color='black', linestyle='--')
                        ax.axhline(2, color='red', linestyle='--', label='+2')
                        ax.axhline(-2, color='green', linestyle='--', label='-2')
                        ax.axhline(3, color='orange', linestyle='--', label='+3 (Stop)')
                        ax.axhline(-3, color='orange', linestyle='--', label='-3 (Stop)')
                        ax.set_title(f"Inversão: {stock_to_sell} / {stock_to_buy}")
                        ax.set_ylabel("Z-Score Inverso")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)

                        # Simulação com ativos invertidos
                        st.markdown("---")
                        st.info("📌 Este gráfico mostra o comportamento do par **invertido**, onde agora consideramos o ativo que estava sendo comprado como vendido e vice-versa.")
                        st.markdown("### Configurar Operação")

                        current_zscore_inv = zscore_inv.iloc[-1]

                        if current_zscore_inv > 0:
                            st.markdown(f"**Legenda de Operação:** Com o Z-Score positivo ({current_zscore_inv:.2f}), recomenda-se **VENDER {stock_to_sell}** e **COMPRAR {stock_to_buy}**.")
                            sell_price = S_inv_1.iloc[-1]
                            buy_price = S_inv_2.iloc[-1]
                        else:
                            st.markdown(f"**Legenda de Operação:** Com o Z-Score negativo ({current_zscore_inv:.2f}), recomenda-se **VENDER {stock_to_buy}** e **COMPRAR {stock_to_sell}**.")
                            # inverte os papéis
                            stock_to_sell, stock_to_buy = stock_to_buy, stock_to_sell
                            sell_price = S_inv_2.iloc[-1]
                            buy_price = S_inv_1.iloc[-1]

                        # Aqui você pode continuar com os controles de simulação iguais ao da aba 1





                    with st.expander("Configurar Operação", expanded=True):
                        current_zscore = zscores[pairs.index(pair_selected)]

                        if current_zscore > 0:
                            st.markdown(
                                f"**Legenda de Operação:** Com o Z-Score positivo ({current_zscore:.2f}), recomenda-se **VENDER {pair_selected[0]}** (ativo sobrevalorizado) e **COMPRAR {pair_selected[1]}** (ativo subvalorizado)."
                            )
                            stock_to_sell = pair_selected[0]
                            stock_to_buy = pair_selected[1]
                            sell_price = S1.iloc[-1]
                            buy_price = S2.iloc[-1]
                        else:
                            st.markdown(
                                f"**Legenda de Operação:** Com o Z-Score negativo ({current_zscore:.2f}), recomenda-se **VENDER {pair_selected[1]}** (ativo sobrevalorizado) e **COMPRAR {pair_selected[0]}** (ativo subvalorizado)."
                            )
                            stock_to_sell = pair_selected[1]
                            stock_to_buy = pair_selected[0]
                            sell_price = S2.iloc[-1]
                            buy_price = S1.iloc[-1]
                    
                        col1, col2 = st.columns(2)

                        # =========================
                        # Coluna 1 - Ação Vendida
                        # =========================
                        with col1:
                            st.subheader(f"Vender Ação: {stock_to_sell}")
                            venda_quantidade = st.number_input("Quantidade para Vender", min_value=100, step=100, value=100, key="venda_quantidade")
                            venda_preco_atual = sell_price
                            venda_total = venda_quantidade * venda_preco_atual

                            st.write(f"Preço Atual: R$ {venda_preco_atual:.2f}")
                            st.write(f"Total Venda: R$ {venda_total:.2f}")
                            
                            slider_venda = st.slider("Movimento (%)", min_value=0, max_value=25, value=5, step=1, key="slider_venda")
                            st.write(f"Simulação de {slider_venda}%")

                            novo_preco_caindo = venda_preco_atual * (1 - slider_venda / 100)
                            lucro_short = (venda_preco_atual - novo_preco_caindo) * venda_quantidade

                            novo_preco_subindo = venda_preco_atual * (1 + slider_venda / 100)
                            preju_short = (venda_preco_atual - novo_preco_subindo) * venda_quantidade

                            st.metric(f"Queda de {slider_venda}% (Lucro Short)", f"R$ {novo_preco_caindo:.2f}", delta=round(lucro_short, 2))
                            st.metric(f"Alta de {slider_venda}% (Prejuízo Short)", f"R$ {novo_preco_subindo:.2f}", delta=round(preju_short, 2))

                        # =========================
                        # Coluna 2 - Ação Comprada
                        # =========================
                        with col2:
                            st.subheader(f"Comprar Ação: {stock_to_buy}")
                            compra_quantidade = st.number_input("Quantidade para Comprar", min_value=100, step=100, value=100, key="compra_quantidade")
                            compra_preco_atual = buy_price
                            compra_total = compra_quantidade * compra_preco_atual

                            st.write(f"Preço Atual: R$ {compra_preco_atual:.2f}")
                            st.write(f"Total Compra: R$ {compra_total:.2f}")

                            slider_compra = st.slider("Movimento (%)", min_value=0, max_value=25, value=5, step=1, key="slider_compra")
                            st.write(f"Simulação de {slider_compra}%")

                            novo_preco_subindo_long = compra_preco_atual * (1 + slider_compra / 100)
                            lucro_long = (novo_preco_subindo_long - compra_preco_atual) * compra_quantidade

                            novo_preco_caindo_long = compra_preco_atual * (1 - slider_compra / 100)
                            preju_long = (novo_preco_caindo_long - compra_preco_atual) * compra_quantidade

                            st.metric(f"Alta de {slider_compra}% (Lucro Long)", f"R$ {novo_preco_subindo_long:.2f}", delta=round(lucro_long, 2))
                            st.metric(f"Queda de {slider_compra}% (Prejuízo Long)", f"R$ {novo_preco_caindo_long:.2f}", delta=round(preju_long, 2))
                        resultado_total = mostrar_fluxo_liquido(venda_total, compra_total)
                        st.markdown("---")
                        preju_total = preju_short + preju_long
                        st.metric("STOP (Soma dos Prejuízos)", f"R$ {preju_total:.2f}", delta=round(preju_total, 2))

            

    
                    st.markdown("---")
                    # if st.button("Salvar Operação como Excel"):
                    col_btn1, col_btn2 = st.columns(2)

                    with col_btn1:
                        if st.button("🚀 Iniciar Operação"):
                            supabase = get_supabase_client()
                            usuario_id = st.session_state["usuario"]["id"]

                            # 🛑 Verifica se já existe operação aberta com este par (ou invertido)
                            if operacao_ja_existe(supabase, usuario_id, stock_to_sell, stock_to_buy):
                                st.warning("⚠️ Já existe uma operação aberta com esse par ou sua inversão.")
                            else:
                                data = {
                                    "usuario_id": usuario_id,
                                    "data_operacao": datetime.now().isoformat(),
                                    "ativo_venda": stock_to_sell,
                                    "ativo_compra": stock_to_buy,
                                    "preco_venda": float(sell_price),
                                    "preco_compra": float(buy_price),
                                    "quantidade_venda": int(venda_quantidade),
                                    "quantidade_compra": int(compra_quantidade),
                                    "resultado_total": float(resultado_total),
                                    "zscore": float(zscores[pairs.index(pair_selected)]),
                                    "p_value": float(pvalues[pairs.index(pair_selected)]),
                                    "hurst": float(hursts[pairs.index(pair_selected)]),
                                    "beta": float(beta_rotations[pairs.index(pair_selected)]),
                                    "half_life": float(half_lives[pairs.index(pair_selected)]),
                                    "status": "aberta"
                                }

                                try:
                                    response = supabase.table("operacoes").insert(data).execute()
                                    if response.data:
                                        st.success("✅ Operação registrada com sucesso no Supabase!")
                                    else:
                                        st.warning("⚠️ A operação foi enviada, mas sem retorno de dados.")
                                except Exception as e:
                                    st.error(f"Erro ao inserir operação no Supabase: {e}")


                    with col_btn2:
                        df_operacao = pd.DataFrame({
                            "Ativo Vendido": [stock_to_sell],
                            "Ativo Comprado": [stock_to_buy],
                            "Preço Venda": [sell_price],
                            "Preço Compra": [buy_price],
                            "Quantidade Vendida": [venda_quantidade],
                            "Quantidade Comprada": [compra_quantidade],
                            "Resultado Total": [resultado_total],
                            "Data Operação": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                        })

                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_operacao.to_excel(writer, index=False, sheet_name="Operacao")
                        output.seek(0)

                        st.download_button(
                            label="📥 Baixar para Excel",
                            data=output,
                            file_name=f"operacao_{stock_to_sell}_{stock_to_buy}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    # Exibir métricas no rodapé
                    st.markdown("---")
                    st.markdown("### 📊 Métricas do Par Selecionado e Recomendações")

                    col1, col2, col3, col4, col5 = st.columns(5)

                    # Z-Score
                    col1.metric("Z-Score", f"{zscores[pairs.index(pair_selected)]:.2f}")
                    col1.caption("📌 Mede o desvio da média. Valores acima de ±2 indicam oportunidade de reversão.")

                    # P-Value
                    col2.metric("P-Value", f"{pvalues[pairs.index(pair_selected)]:.4f}")
                    col2.caption("📌 Probabilidade de cointegração ser aleatória. Valores abaixo de 0.05 são desejáveis.")

                    # Hurst Exponent
                    col3.metric("Hurst", f"{hursts[pairs.index(pair_selected)]:.4f}")
                    col3.caption("📌 Mede a tendência de reversão ou persistência. Valores próximos de 0.5 são ideais.")

                    # Beta Rotation
                    col4.metric("Beta", f"{beta_rotations[pairs.index(pair_selected)]:.4f}")
                    col4.caption("📌 Sensibilidade do ativo em relação ao outro. Quanto menor, melhor para pares estáveis.")

                    # Half-Life
                    col5.metric("Half-Life", f"{half_lives[pairs.index(pair_selected)]:.2f}")
                    col5.caption("📌 Tempo esperado para metade da reversão ao valor médio. Quanto menor, melhor.")


    if selected == "Operações":
        supabase = get_supabase_client()
        usuario_id = st.session_state["usuario"]["id"]
        operacoes_b64 = get_base64("logos/operacoes.png")
        
        st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{operacoes_b64}"
                 alt="Operações"
                 style="height: 125px; margin-right: 25px;">
            <h1 style="margin: 0;">Operações</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
              
       
        try:
            response = (
            supabase.table("operacoes")
            .select("*")
            .eq("usuario_id", usuario_id)
            .eq("status", "aberta")  # 👈 aqui o filtro necessário
            .order("data_operacao", desc=True)
            .execute()
        )
            operacoes_supabase = response.data
        except Exception as e:
            st.error(f"Erro ao buscar operações do Supabase: {e}")
            operacoes_supabase = []

        if operacoes_supabase:
                st.markdown("### 🔘 Minhas Operações Salvas")
                cols_buttons = st.columns(4)
                for idx, operacao in enumerate(operacoes_supabase):
                    resumo = (
                        f"{operacao['ativo_venda']} / {operacao['ativo_compra']} | "
                        f"{pd.to_datetime(operacao['data_operacao']).strftime('%d/%m/%Y %H:%M')}"
                    )
                    col = cols_buttons[idx % 4]
                    with col:
                        if st.button(resumo, key=f"botao_op_{operacao['id']}"):
                            st.session_state["operacao_carregada"] = operacao

        else:
            st.info("Nenhuma operação salva encontrada.")

        if "operacao_carregada" in st.session_state:
            operacao = st.session_state["operacao_carregada"]

            cotacoes_df = st.session_state.get("global_cotacoes", pd.DataFrame())
            if cotacoes_df.empty:
                st.error("❗ Nenhuma cotação carregada. Vá para a aba 'Cotações'.")
                st.stop()

            if not pd.api.types.is_datetime64_any_dtype(cotacoes_df.index):
                cotacoes_df.index = pd.to_datetime(cotacoes_df.index, errors='coerce')
                cotacoes_df = cotacoes_df[~cotacoes_df.index.isnull()]
                st.session_state["global_cotacoes"] = cotacoes_df

            ativo_venda = operacao["ativo_venda"]
            ativo_compra = operacao["ativo_compra"]
            data_operacao = pd.to_datetime(operacao["data_operacao"])
            preco_venda = operacao["preco_venda"]
            preco_compra = operacao["preco_compra"]
            quantidade_venda = operacao["quantidade_venda"]
            quantidade_compra = operacao["quantidade_compra"]
            numero_periodos = 120
      

            if ativo_venda not in cotacoes_df.columns or ativo_compra not in cotacoes_df.columns:
                st.error(f"❌ {ativo_venda} ou {ativo_compra} não estão nas cotações.")
                st.stop()

            serie_venda = cotacoes_df[ativo_venda].tail(numero_periodos)
            serie_compra = cotacoes_df[ativo_compra].tail(numero_periodos)

            # === Limpeza das séries ===
            serie_venda = serie_venda.replace([np.inf, -np.inf], np.nan)
            serie_compra = serie_compra.replace([np.inf, -np.inf], np.nan)
            serie_venda, serie_compra = serie_venda.align(serie_compra, join="inner")
            mascara = (~serie_venda.isna()) & (~serie_compra.isna())
            serie_venda = serie_venda[mascara]
            serie_compra = serie_compra[mascara]

            # === Métricas Estatísticas ===
            if len(serie_venda) >= 30:
                ratio = serie_venda / serie_compra
                zscore_series = (ratio - ratio.mean()) / ratio.std()
                zscore_atual = zscore_series.iloc[-1]

                pvalue_atual = coint(serie_venda, serie_compra)[1]
                hurst_atual, _, _ = compute_Hc((serie_venda / serie_compra).dropna(), kind='price', simplified=True)

                X = serie_compra.values.reshape(-1, 1)
                y = serie_venda.values
                model_beta = LinearRegression().fit(X, y)
                beta_atual = model_beta.coef_[0]

                spread = serie_venda - beta_atual * serie_compra
                spread_lag = spread.shift(1).dropna()
                spread_ret = spread.diff().dropna()
                spread_lag = spread_lag.loc[spread_ret.index]

                model_hl = LinearRegression().fit(spread_lag.values.reshape(-1, 1), spread_ret.values)
                half_life_atual = -np.log(2) / model_hl.coef_[0]
            else:
                zscore_series = pd.Series()
                zscore_atual = pvalue_atual = hurst_atual = beta_atual = half_life_atual = np.nan

            # === Resultados da Operação ===
            total_venda = quantidade_venda * serie_venda.iloc[-1]
            total_compra = quantidade_compra * serie_compra.iloc[-1]
            saldo_op = total_venda - total_compra

            preco_atual_venda = serie_venda.iloc[-1]
            preco_atual_compra = serie_compra.iloc[-1]
            lucro_venda = (preco_venda - preco_atual_venda) * quantidade_venda
            lucro_compra = (preco_atual_compra - preco_compra) * quantidade_compra
            saldo_final = lucro_venda + lucro_compra
            
            # Calcular dias desde a abertura
            dias_operacao = (pd.Timestamp.now(tz=None).normalize() - data_operacao.tz_localize(None).normalize()).days

            # Definir sinal
            sinal = "+" if saldo_final > 0 else "-" if saldo_final < 0 else ""
            valor_formatado = f"{sinal} R$ {abs(saldo_final):,.2f}"
            delta_formatado = f"⏳ {dias_operacao} {'dias de operaçao' if dias_operacao != 1 else 'dia de operação'}"

            # Cor do delta
            cor_delta = "normal"
            if saldo_final > 0:
                cor_delta = "inverse"
            elif saldo_final < 0:
                cor_delta = "off"

            # Layout centralizado
            st.markdown("---")
            st.markdown("### 💰 Saldo Final Consolidado")
            col_esq, col_centro, col_dir = st.columns([1, 1, 1])

            with col_centro:
                st.metric(
                    label="Resultado Total",
                    value=valor_formatado,
                    delta=delta_formatado,
                    delta_color=cor_delta
                )
            st.markdown("---")
            plt.figure(figsize=(10, 3))
            plt.plot(zscore_series, label="Z-Score")
            plt.axhline(0, color='black', linestyle='--')
            plt.axhline(2, color='red', linestyle='--')
            plt.axhline(-2, color='green', linestyle='--')
            plt.axhline(3, color='orange', linestyle='--')
            plt.axhline(-3, color='orange', linestyle='--')
            plt.legend()
            plt.title(f"Z-Score: {ativo_venda} / {ativo_compra}")
            st.pyplot(plt)

            # === Tabela Consolidada ===
            st.markdown("---")
            st.markdown("### 📄 Tabela Consolidada de Operações")
            st.dataframe(pd.DataFrame([
                {
                    "Data Operação": data_operacao,
                    "Ativo": ativo_venda,
                    "Quantidade": quantidade_venda,
                    "Tipo": "Venda",
                    "Valor Inicial": f"R$ {preco_venda:.2f}",
                    "Valor Total": f"R$ {total_venda:.2f}",
                    "Saldo": f"R$ {saldo_op:.2f}"
                },
                {
                    "Data Operação": data_operacao,
                    "Ativo": ativo_compra,
                    "Quantidade": quantidade_compra,
                    "Tipo": "Compra",
                    "Valor Inicial": f"R$ {preco_compra:.2f}",
                    "Valor Total": f"R$ {total_compra:.2f}",
                    "Saldo": f"R$ {saldo_op:.2f}"
                }
            ]))


            st.markdown("### 📌 Posição Atual")
            st.dataframe(pd.DataFrame([
                {
                    "Ativo": ativo_venda,
                    "Tipo": "Venda",
                    "Quantidade": quantidade_venda,
                    "Preço Inicial": f"R$ {preco_venda:.2f}",
                    "Preço Atual": f"R$ {preco_atual_venda:.2f}",
                    "Lucro/Prejuízo": f"R$ {lucro_venda:.2f}"
                },
                {
                    "Ativo": ativo_compra,
                    "Tipo": "Compra",
                    "Quantidade": quantidade_compra,
                    "Preço Inicial": f"R$ {preco_compra:.2f}",
                    "Preço Atual": f"R$ {preco_atual_compra:.2f}",
                    "Lucro/Prejuízo": f"R$ {lucro_compra:.2f}"
                }
            ]))
            st.markdown("---")
            # --- Comparativo entre valores salvos e valores atuais ---
            st.markdown("### 🧮 Comparativo de Resultados")

            # Recalcular valores com as séries carregadas
            ratio_atual = serie_venda / serie_compra
            zscore_atual = ((ratio_atual - ratio_atual.mean()) / ratio_atual.std()).iloc[-1]

           
            pvalue_atual = coint(serie_venda, serie_compra)[1]

            def calcular_hurst(ts):
                import numpy as np
                lags = range(2, 100)
                tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0]

            hurst_atual = calcular_hurst(ratio_atual)

            import statsmodels.api as sm
            X = sm.add_constant(serie_compra)
            model = sm.OLS(serie_venda, X).fit()
            beta_atual = model.params[1]

            spread = serie_venda - beta_atual * serie_compra
            spread_lag = spread.shift(1).dropna()
            delta_spread = spread.diff().dropna()
            model_hl = sm.OLS(delta_spread, sm.add_constant(spread_lag)).fit()
            coef_name = model_hl.params.index[1]  # geralmente é o nome da variável lag
            half_life_atual = -np.log(2) / model_hl.params[coef_name]

            # Valores salvos
            zscore_salvo = operacao['zscore']
            pvalue_salvo = operacao['p_value']
            hurst_salvo = operacao['hurst']
            beta_salvo = operacao['beta']
            half_life_salvo = operacao['half_life']

            # Exibir métricas comparativas
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Z-Score", format_metric(zscore_atual), delta=format_metric(zscore_atual - zscore_salvo, "+N/A", "{:+.2f}"))
            col2.metric("P-Value", f"{pvalue_atual:.4f}", delta=f"{pvalue_atual - pvalue_salvo:+.4f}")
            col3.metric("Hurst", f"{hurst_atual:.4f}", delta=f"{hurst_atual - hurst_salvo:+.4f}")
            col4.metric("Beta", f"{beta_atual:.4f}", delta=f"{beta_atual - beta_salvo:+.4f}")
            col5.metric("Half-Life", f"{half_life_atual:.2f}", delta=f"{half_life_atual - half_life_salvo:+.2f}")




            st.markdown("---")
            st.markdown("### 📈 Análise Gráfica da Operação")
            st.markdown(f"#### {ativo_venda} vs {ativo_compra} ({data_operacao.date()})")


            plt.figure(figsize=(10, 3))
            plt.plot(serie_venda / serie_venda.iloc[0], label=ativo_venda)
            plt.plot(serie_compra / serie_compra.iloc[0], label=ativo_compra)
            plt.title("Cotação Normalizada")
            plt.legend()
            st.pyplot(plt)

            col1, col2 = st.columns(2)
            with col1:
                plotar_beta_movel(serie_venda, serie_compra, window=40)
            with col2:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(serie_venda, serie_compra, alpha=0.7)
                ax.set_xlabel(ativo_venda)
                ax.set_ylabel(ativo_compra)
                ax.set_title("Dispersão")
                ax.grid(True)
                fig.tight_layout()
                st.pyplot(fig)

            st.markdown("### 📉 Candlestick dos Ativos")

            col_a, col_b = st.columns(2)
            with col_a:
                     plot_candlestick_e_volume(cotacoes_df[ativo_venda], ativo_venda)
            with col_b:
                     plot_candlestick_e_volume(cotacoes_df[ativo_compra], ativo_compra)
            st.markdown("### 📈 Análise Gráfica da Operação")

        # --- Final da análise ---
            st.markdown("---")
            st.markdown("### 🛑 Encerrar Operação")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🛑 Encerrar Operação", type="primary"):
                        try:
                            # Obter preços atuais das ações
                            cotacoes_df = st.session_state.get("global_cotacoes", pd.DataFrame())
                            if cotacoes_df.empty:
                                st.error("❗ Nenhuma cotação carregada. Vá para a aba 'Cotações'.")
                                st.stop()

                            # Certifica que o índice seja datetime
                            if not pd.api.types.is_datetime64_any_dtype(cotacoes_df.index):
                                cotacoes_df.index = pd.to_datetime(cotacoes_df.index, errors='coerce')
                                cotacoes_df = cotacoes_df[~cotacoes_df.index.isnull()]
                                st.session_state["global_cotacoes"] = cotacoes_df

                            preco_fechamento_venda = cotacoes_df[operacao["ativo_venda"]].iloc[-1]
                            preco_fechamento_compra = cotacoes_df[operacao["ativo_compra"]].iloc[-1]
                            agora = datetime.now().isoformat()

                            # Atualiza no Supabase
                            supabase.table("operacoes") \
                                .update({
                                    "status": "encerrado",
                                    "data_encerramento": agora,
                                    "preco_encerramento_venda": float(preco_fechamento_venda),
                                    "preco_encerramento_compra": float(preco_fechamento_compra)
                                }) \
                                .eq("id", operacao["id"]) \
                                .execute()

                            st.success("✅ Operação encerrada com sucesso!")
                            del st.session_state["operacao_carregada"]  # limpa a operação ativa
                            st.rerun()

                        except Exception as e:
                            st.error(f"Erro ao encerrar operação: {e}")


                    
    if selected == "Encerradas":
        st.title("📁 Operações Encerradas")

        supabase = get_supabase_client()
        usuario_id = st.session_state["usuario"]["id"]

        # --- Buscar datas mínima e máxima disponíveis para esse usuário ---
        try:
            response = supabase.table("operacoes") \
                .select("data_operacao") \
                .eq("usuario_id", usuario_id) \
                .eq("status", "encerrado") \
                .execute()

            datas = [pd.to_datetime(item["data_operacao"]) for item in response.data]
            data_min = min(datas) if datas else datetime.today()
            data_max = max(datas) if datas else datetime.today()

        except Exception as e:
            st.error(f"Erro ao buscar datas: {e}")
            data_min = data_max = datetime.today()

        # --- Filtros de data ---
        col1, col2 = st.columns(2)
        with col1:
            data_inicio = st.date_input("De", value=data_min)
        with col2:
            data_fim = st.date_input("Até", value=data_max)

        # Botão centralizado
        st.markdown(" ")
        col_left, col_mid, col_right = st.columns([1, 2, 1])
        with col_mid:
            buscar = st.button("🔍 Buscar Operações", use_container_width=True)

        # Lógica de busca
        if buscar:
            with st.spinner("🔎 Buscando operações encerradas..."):
                try:
                    inicio_datetime = datetime.combine(data_inicio, dt_time.min)
                    fim_datetime = datetime.combine(data_fim, dt_time.max)


                    response = (
                        supabase.table("operacoes")
                        .select("*")
                        .eq("usuario_id", usuario_id)
                        .eq("status", "encerrado")  # status deve estar exatamente assim
                        .gte("data_operacao", inicio_datetime.isoformat())
                        .lte("data_operacao", fim_datetime.isoformat())
                        .order("data_operacao", desc=True)
                        .execute()
                    )
                    encerradas = response.data
                except Exception as e:
                    st.error(f"Erro ao buscar operações encerradas: {e}")
                    encerradas = []

         
                if encerradas:
                    st.markdown("### 📄 Resultados das Operações Encerradas")

                    df_resultados = pd.DataFrame(encerradas)
                    df_resultados["data_operacao"] = pd.to_datetime(df_resultados["data_operacao"]).dt.strftime("%d/%m/%Y")

                    # Totais financeiros com base nos preços de encerramento
                    df_resultados["total_venda"] = (
                    pd.to_numeric(df_resultados["preco_encerramento_venda"], errors="coerce") *
                    pd.to_numeric(df_resultados["quantidade_venda"], errors="coerce")
                   ).round(2)

                    df_resultados["total_compra"] = (
                    pd.to_numeric(df_resultados["preco_encerramento_compra"], errors="coerce") *
                    pd.to_numeric(df_resultados["quantidade_compra"], errors="coerce")
                     ).round(2)


                    # Retorno percentual sobre o valor vendido
                    df_resultados["retorno_pct"] = (
                        (df_resultados["resultado_total"] / df_resultados["total_venda"]) * 100
                         ).round(2)

                    # Renomear colunas
                    df_resultados.rename(columns={
                        "data_operacao": "Data",
                        "ativo_venda": "Ativo Vendido",
                        "quantidade_venda": "Qtd Venda",
                        "total_venda": "Total Venda (R$)",
                        "ativo_compra": "Ativo Comprado",
                        "quantidade_compra": "Qtd Compra",
                        "total_compra": "Total Compra (R$)",
                        "resultado_total": "Resultado (R$)",
                        "retorno_pct": "% Lucro/Prejuízo",
                    }, inplace=True)

                    # Exibir operação por operação
                    
                    for _, row in df_resultados.iterrows():
                        st.markdown("### 📌 Operação")

                        # Converter as datas de string para datetime
                        data_abertura = pd.to_datetime(row["Data"])
                        data_encerramento = pd.to_datetime(row.get("data_encerramento", None))

                        # Calcular a duração em dias, se houver data de encerramento
                        if pd.notnull(data_encerramento):
                            duracao_dias = (data_encerramento.tz_localize(None) - data_abertura.tz_localize(None)).days
                        else:
                            duracao_dias = "—"

                        # --- Cabeçalho informativo da operação ---
                        st.markdown("### 🕒 Período da Operação")
                        col_d1, col_d2, col_d3 = st.columns(3)
                        with col_d1:
                            st.write(f"📅 **Data de Abertura:** `{data_abertura.strftime('%d/%m/%Y')}`")
                        with col_d2:
                            st.write(f"📆 **Data de Encerramento:** `{data_encerramento.strftime('%d/%m/%Y') if pd.notnull(data_encerramento) else '—'}`")
                        with col_d3:
                            st.write(f"⏳ **Duração:** `{duracao_dias}` dias")
                                               

                        col1, col2, col3, col4 = st.columns(4)

                        # --- COLUNA 1: Venda ---
                        with col1:
                            st.markdown(f"**💰 Venda:** `{row['Ativo Vendido']}`")
                            st.write(f"- Quantidade: `{row['Qtd Venda']}`")
                            st.write(f"- Preço Inicial: `R$ {row['preco_venda']:.2f}`")
                            st.write(f"- Encerramento: `R$ {row['preco_encerramento_venda']:.2f}`")
                            st.write(f"- Total Venda Inicial: `R$ {(row['preco_venda'] * row['Qtd Venda']):,.2f}`")
                            st.write(f"- Total Venda Final: `R$ {row['Total Venda (R$)']:,.2f}`")

                        # --- COLUNA 2: Compra ---
                        with col2:
                            st.markdown(f"**🛒 Compra:** `{row['Ativo Comprado']}`")
                            st.write(f"- Quantidade: `{row['Qtd Compra']}`")
                            st.write(f"- Preço Inicial: `R$ {row['preco_compra']:.2f}`")
                            st.write(f"- Encerramento: `R$ {row['preco_encerramento_compra']:.2f}`")
                            st.write(f"- Total Compra Inicial: `R$ {(row['preco_compra'] * row['Qtd Compra']):,.2f}`")
                            st.write(f"- Total Compra Final: `R$ {row['Total Compra (R$)']:,.2f}`")

                        # --- COLUNA 3: Cálculos e Saldo Inicial ---
                        with col3:
                            total_venda_inicial = row["preco_venda"] * row["Qtd Venda"]
                            total_compra_inicial = row["preco_compra"] * row["Qtd Compra"]
                            saldo_inicial = total_venda_inicial - total_compra_inicial
                            resultado = row["Resultado (R$)"]
                            retorno = row["% Lucro/Prejuízo"]

                            st.markdown("**📊 Saldo Inicial:**")
                            st.metric(
                                label="Saldo Inicial",
                                value=f"R$ {saldo_inicial:,.2f}",
                                delta=f"{'+' if saldo_inicial > 0 else '-' if saldo_inicial < 0 else ''}R$ {abs(saldo_inicial):,.2f}",
                                delta_color="inverse" if saldo_inicial > 0 else "off"
                            )

                            st.markdown("**📈 Resultado Final:**")
                            st.metric(
                                "Resultado Final",
                                value=f"R$ {abs(resultado):,.2f}",
                                delta=f"{'+' if resultado > 0 else '-' if resultado < 0 else ''}R$ {abs(resultado):,.2f}",
                                delta_color="inverse" if resultado > 0 else "off"
                            )

                                               
                        with col4:
                            retorno = row["% Lucro/Prejuízo"]
                            cor = "🟢" if retorno > 0 else "🔴" if retorno < 0 else "⚪"
                            st.markdown("**📉 Lucro / Prejuízo (%):**")
                            st.metric(
                                label="% Lucro/Prejuízo",
                                value=f"{retorno:.2f}%",
                                delta=f"{'+' if retorno > 0 else '-' if retorno < 0 else ''}{abs(retorno):.2f}%",
                                delta_color="inverse" if retorno > 0 else "off"
                            )   


                        st.markdown("---")

                else:
                    st.warning("⚠️ Nenhuma operação encerrada encontrada nesse período.")