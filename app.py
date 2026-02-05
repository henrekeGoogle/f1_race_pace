import streamlit as st
import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os

# Configuração da Página
st.set_page_config(page_title="F1 Race Pace Analyst", layout="wide")

# Título
st.title("🏎️ F1 Race Pace & Tyre Degradation & Fabrilson é bixa")

# Setup do FastF1
folder = 'cache'
if not os.path.exists(folder):
    os.makedirs(folder)
fastf1.Cache.enable_cache(folder)
fastf1.plotting.setup_mpl(misc_mpl_mods=False)


# --- FUNÇÕES COM CACHE (O segredo do Streamlit) ---
# O decorator @st.cache_data faz com que essa função pesada só rode
# se os parâmetros mudarem. Senão, ele pega da memória.
@st.cache_data
def carregar_dados(ano, gp, sessao):
    try:
        session = fastf1.get_session(ano, gp, sessao)
        session.load(telemetry=False, weather=False)
        return session
    except Exception as e:
        return None


def formato_f1(x, pos):
    minutes = int(x // 60)
    seconds = int(x % 60)
    millis = int((x - int(x)) * 1000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


# --- SIDEBAR (Menu Lateral) ---
st.sidebar.header("Configurações")
ano_selecionado = st.sidebar.selectbox("Ano", [2024, 2023, 2022])
gp_selecionado = st.sidebar.selectbox("Grand Prix", ["Bahrain", "Saudi Arabia", "Australia", "Japan", "Brazil",
                                                     "Las Vegas"])  # Adicione os outros
sessao_selecionada = st.sidebar.selectbox("Sessão", ["FP1", "FP2", "FP3", "Q", "R"])

# Botão de Carregar
if st.sidebar.button("Carregar Sessão"):
    with st.spinner('Baixando dados da telemetria...'):
        session = carregar_dados(ano_selecionado, gp_selecionado, sessao_selecionada)

        if session:
            st.session_state['session_data'] = session
            st.success(f"Dados de {gp_selecionado} carregados!")
        else:
            st.error("Erro ao carregar sessão. Verifique o nome do GP.")

# --- ÁREA PRINCIPAL ---
if 'session_data' in st.session_state:
    session = st.session_state['session_data']

    # Seleção de Pilotos (dinâmica baseada na sessão carregada)
    lista_pilotos = sorted(session.drivers)
    col1, col2 = st.columns(2)
    with col1:
        piloto1 = st.selectbox("Piloto 1", lista_pilotos, index=0)
    with col2:
        piloto2 = st.selectbox("Piloto 2", lista_pilotos, index=1)

    min_voltas = st.slider("Mínimo de Voltas (para considerar Stint)", 3, 20, 8)

    # Botão para Gerar Gráfico
    # Botão para Gerar Gráfico
    if st.button("Comparar Ritmo"):
        fig, ax = plt.subplots(figsize=(10, 6))

        drivers_numeros = [piloto1, piloto2]

        for driver_num in drivers_numeros:
            # --- CORREÇÃO AQUI ---
            # 1. Pegamos os dados completos do piloto usando o número
            info_piloto = session.get_driver(driver_num)

            # 2. Extraímos a sigla (Ex: '81' vira 'PIA')
            piloto_nome = info_piloto['Abbreviation']

            # 3. Usamos a sigla para pegar a cor (é muito mais seguro)
            driver_color = fastf1.plotting.get_driver_color(piloto_nome, session=session)

            # Agora filtramos as voltas usando o número ou a sigla
            laps = session.laps.pick_driver(driver_num)

            for stint_id, stint_data in laps.groupby('Stint'):
                clean_laps = stint_data.pick_quicklaps(threshold=1.07)

                if len(clean_laps) >= min_voltas:
                    composto = stint_data['Compound'].iloc[0]
                    x = clean_laps['TyreLife'].values
                    y = clean_laps['LapTime'].dt.total_seconds().values

                    if len(x) > 1:
                        slope, intercept = np.polyfit(x, y, 1)

                        sns.scatterplot(x=x, y=y, ax=ax, color=driver_color, s=80, alpha=0.6)
                        y_pred = slope * x + intercept

                        # Usamos 'piloto_nome' (PIA) no label em vez do número
                        ax.plot(x, y_pred, color=driver_color, linewidth=2, linestyle='--',
                                label=f"{piloto_nome} ({composto}) Deg: {slope:.3f}s")

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(formato_f1))
        ax.set_title(
            f"Race Sim: {session.get_driver(piloto1)['Abbreviation']} vs {session.get_driver(piloto2)['Abbreviation']}")
        ax.set_xlabel("Vida do Pneu")
        ax.set_ylabel("Tempo")
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

else:
    st.info("👈 Selecione um GP e clique em 'Carregar Sessão' para começar.")