import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================

st.set_page_config(
    page_title="Scanner Estatístico AUVP11",
    layout="wide"
)

st.title("📊 Scanner Estatístico — AUVP11")
st.caption("Retorno à média | Mean Reversion | Gain fixo em 4%")

# =========================================================
# LISTA DE ATIVOS DO AUVP11
# =========================================================

AUVP11_HOLDINGS = [
  # Bancos / Financeiro
    "ITUB4.SA","BBDC4.SA","BBAS3.SA","BPAC11.SA","ITSA4.SA","B3SA3.SA",

    # Energia / Utilities
    "EGIE3.SA","CPLE6.SA","CPFE3.SA","TAEE11.SA","TRPL4.SA","CMIG4.SA","SAPR11.SA","SAPR4.SA",

    # Telecom
    "VIVT3.SA","TIMS3.SA",

    # Consumo / Serviços
    "ABEV3.SA","PSSA3.SA","MULT3.SA","ALOS3.SA","ODPV3.SA",

    # Construção / Industrial
    "CYRE3.SA","KEPL3.SA","POMO4.SA","TOTS3.SA",

    # Commodities / Outros
    "PETR4.SA","PRIO3.SA","VALE3.SA",

    # Crescimento
    "WEGE3.SA","RDOR3.SA","SBSP3.SA","BBSE3.SA"
]

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Configurações")

periodo = st.sidebar.slider(
    "Período Estatístico",
    min_value=10,
    max_value=100,
    value=20
)

zscore_minimo = st.sidebar.slider(
    "Z-Score mínimo",
    min_value=-5.0,
    max_value=-1.0,
    value=-2.0,
    step=0.1
)

gain_percentual = st.sidebar.slider(
    "Gain alvo (%)",
    min_value=1.0,
    max_value=10.0,
    value=4.0,
    step=0.5
)

periodo_dados = st.sidebar.selectbox(
    "Histórico utilizado",
    options=["2y", "5y", "10y"],
    index=1
)

# =========================================================
# DOWNLOAD DOS DADOS
# =========================================================

@st.cache_data(ttl=3600)
def baixar_dados(ticker, periodo_dados):

    try:

        df = yf.download(
            ticker,
            period=periodo_dados,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if df.empty:
            return None

        close = df["Close"].squeeze()

        if close.empty:
            return None

        return close.dropna()

    except:
        return None

# =========================================================
# FUNÇÃO DE ANÁLISE
# =========================================================

def analisar_ativo(ticker):

    close = baixar_dados(ticker, periodo_dados)

    if close is None:
        return None

    if len(close) < periodo + 50:
        return None

    # =====================================================
    # MÉDIA E DESVIO PADRÃO
    # =====================================================

    media = close.rolling(periodo).mean()

    desvio = close.rolling(periodo).std()

    # =====================================================
    # Z-SCORE
    # =====================================================

    zscore = (close - media) / desvio

    z_atual = zscore.iloc[-1]

    if np.isnan(z_atual):
        return None

    # =====================================================
    # FILTRO DE DISTORÇÃO
    # =====================================================

    if z_atual > zscore_minimo:
        return None

    preco_atual = close.iloc[-1]

    media_atual = media.iloc[-1]

    # =====================================================
    # BACKTEST HISTÓRICO
    # =====================================================

    ocorrencias = 0

    acertos = 0

    candles_medio = []

    for i in range(periodo, len(close) - 20):

        z_passado = zscore.iloc[i]

        if np.isnan(z_passado):
            continue

        if z_passado <= zscore_minimo:

            ocorrencias += 1

            preco_entrada = close.iloc[i]

            alvo = preco_entrada * (
                1 + gain_percentual / 100
            )

            for j in range(i + 1, min(i + 21, len(close))):

                if close.iloc[j] >= alvo:

                    acertos += 1

                    candles_medio.append(j - i)

                    break

    # =====================================================
    # PROBABILIDADE
    # =====================================================

    if ocorrencias == 0:

        probabilidade = 0

        tempo_medio = None

    else:

        probabilidade = (
            acertos / ocorrencias
        ) * 100

        tempo_medio = (
            np.mean(candles_medio)
            if candles_medio
            else None
        )

    # =====================================================
    # SCORE FINAL
    # =====================================================

    score = (
        (abs(z_atual) * 35)
        +
        (probabilidade * 0.65)
        -
        ((tempo_medio or 20) * 1.5)
    )

    distancia_media = (
        (preco_atual / media_atual) - 1
    ) * 100

    # =====================================================
    # CLASSIFICAÇÃO DAS OCORRÊNCIAS
    # =====================================================

    if ocorrencias < 10:
        confianca = "Fraca"

    elif ocorrencias < 20:
        confianca = "Moderada"

    elif ocorrencias < 40:
        confianca = "Boa"

    elif ocorrencias < 80:
        confianca = "Forte"

    else:
        confianca = "Excelente"

    # =====================================================
    # RETORNO FINAL
    # =====================================================

    return {

        "Ticker": ticker.replace(".SA", ""),

        "Preço": round(preco_atual, 2),

        "Z-Score": round(z_atual, 2),

        "Distância Média %": round(distancia_media, 2),

        "Probabilidade +4%": round(probabilidade, 1),

        "Tempo Médio": (
            round(tempo_medio, 1)
            if tempo_medio
            else "-"
        ),

        "Ocorrências": ocorrencias,

        "Confiabilidade": confianca,

        "Score": round(score, 1)
    }

# =========================================================
# EXECUÇÃO DO SCANNER
# =========================================================

if st.button("🚀 Executar Scanner Estatístico"):

    with st.spinner("Analisando ativos do AUVP11..."):

        resultados = Parallel(n_jobs=2)(
            delayed(analisar_ativo)(ticker)
            for ticker in AUVP11_HOLDINGS
        )

    resultados = [
        r for r in resultados
        if r is not None
    ]

    # =====================================================
    # SEM RESULTADOS
    # =====================================================

    if len(resultados) == 0:

        st.warning(
            "Nenhuma distorção estatística encontrada."
        )

    else:

        # =================================================
        # DATAFRAME
        # =================================================

        df = pd.DataFrame(resultados)

        df = df.sort_values(
            by=[
                "Score",
                "Probabilidade +4%"
            ],
            ascending=False
        )

        # =================================================
        # DASHBOARD SUPERIOR
        # =================================================

        melhor = df.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Ativos Encontrados",
            len(df)
        )

        c2.metric(
            "Melhor Oportunidade",
            melhor["Ticker"]
        )

        c3.metric(
            "Maior Probabilidade",
            f'{df["Probabilidade +4%"].max():.1f}%'
        )

        c4.metric(
            "Melhor Z-Score",
            df["Z-Score"].min()
        )

        st.divider()

        # =================================================
        # TABELA PRINCIPAL
        # =================================================

        st.subheader("📋 Ranking Estatístico")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

# =========================================================
# RODAPÉ
# =========================================================

st.divider()

st.caption(
    """
    Scanner quantitativo de retorno à média.
    Uso educacional e operacional discricionário.
    """
)
