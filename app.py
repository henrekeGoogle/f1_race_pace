import streamlit as st
import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Configuração da Página
st.set_page_config(page_title="F1 Race Pace Analyst", layout="wide")
st.title("🏎️ F1 Race Pace & Tyre Degradation")

# Setup do FastF1
folder = 'cache'
if not os.path.exists(folder):
    os.makedirs(folder)
fastf1.Cache.enable_cache(folder)
fastf1.plotting.setup_mpl(misc_mpl_mods=False)


@st.cache_data
def carregar_dados(ano, gp, sessao):
    try:
        session = fastf1.get_session(ano, gp, sessao)
        session.load(telemetry=True, weather=False)
        return session
    except Exception as e:
        return None


def formato_f1(x, pos):
    minutes = int(x // 60)
    seconds = int(x % 60)
    millis = int((x - int(x)) * 1000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


# --- SIDEBAR ---
st.sidebar.header("Configurações da Sessão")
ano_selecionado = st.sidebar.selectbox("Ano", [2026,2025, 2024, 2023, 2022])
gp_selecionado = st.sidebar.selectbox("Grand Prix",
                                      ["Bahrain", "Saudi Arabia", "Australia","CHINA", "Japan", "Great Britain", "Hungary",
                                       "Brazil", "Las Vegas"])
sessao_selecionada = st.sidebar.selectbox("Sessão", ["FP1", "FP2", "FP3", "Q", "R"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parâmetros de Engenharia")
delta_sm = st.sidebar.number_input("Delta Macio -> Médio (s)", value=0.80, step=0.05)
delta_sh = st.sidebar.number_input("Delta Macio -> Duro (s)", value=1.20, step=0.05)
delta_motor = st.sidebar.number_input("Vantagem Bateria/Motor Quali (s)", value=0.50, step=0.05)
peso_10kg = st.sidebar.number_input("Custo de 10kg de Combustível (s)", value=0.30, step=0.01)

if st.sidebar.button("Carregar Sessão"):
    with st.spinner('Baixando dados da telemetria...'):
        session = carregar_dados(ano_selecionado, gp_selecionado, sessao_selecionada)
        if session:
            st.session_state['session_data'] = session
            st.success("Dados carregados!")
        else:
            st.error("Erro ao carregar sessão.")

# --- ÁREA PRINCIPAL ---
if 'session_data' in st.session_state:
    session = st.session_state['session_data']

    lista_pilotos = sorted(session.drivers)
    col1, col2 = st.columns(2)
    with col1:
        piloto1 = st.selectbox("Piloto 1", lista_pilotos, index=0)
    with col2:
        piloto2 = st.selectbox("Piloto 2", lista_pilotos, index=1 if len(lista_pilotos) > 1 else 0)

    min_voltas = st.slider("Mínimo de Voltas (para considerar Stint)", 3, 20, 8)

    if st.button("Comparar Ritmo e Analisar Combustível"):

        # --- 4 ABAS ---
        aba_pace, aba_telemetry, aba_ranking, aba_peso_tempo = st.tabs([
            "⏱️ Ritmo de Corrida", "🏎️ Telemetria", "🏆 Ranking Bruto", "⚖️ Equalizador de Peso"
        ])

        fig_pace, ax_pace = plt.subplots(figsize=(10, 6))
        fig_telemetry, ax_telemetry = plt.subplots(figsize=(10, 6))

        drivers_numeros = [piloto1, piloto2]

        with st.spinner('Processando telemetria e calculando pesos...'):

            # ==========================================
            # 1. GRÁFICOS (Pilotos Selecionados)
            # ==========================================
            for driver_num in drivers_numeros:
                info_piloto = session.get_driver(driver_num)
                piloto_nome = info_piloto['Abbreviation']
                driver_color = fastf1.plotting.get_driver_color(piloto_nome, session=session)
                laps = session.laps.pick_driver(driver_num)

                for stint_id, stint_data in laps.groupby('Stint'):
                    clean_laps = stint_data.pick_quicklaps(threshold=1.07)
                    if len(clean_laps) >= min_voltas:
                        composto = stint_data['Compound'].iloc[0]
                        x = clean_laps['TyreLife'].values
                        y_tempo = clean_laps['LapTime'].dt.total_seconds().values

                        y_acelerador = []
                        for _, lap in clean_laps.iterlaps():
                            try:
                                tel = lap.get_telemetry()
                                y_acelerador.append(tel['Throttle'].mean())
                            except:
                                y_acelerador.append(np.nan)

                        if len(x) > 1:
                            slope, intercept = np.polyfit(x, y_tempo, 1)
                            sns.scatterplot(x=x, y=y_tempo, ax=ax_pace, color=driver_color, s=80, alpha=0.6)
                            y_pred = slope * x + intercept
                            ax_pace.plot(x, y_pred, color=driver_color, linewidth=2, linestyle='--',
                                         label=f"{piloto_nome} ({composto}) Deg: {slope:.3f}s")

                        ax_telemetry.plot(x, y_acelerador, marker='o', color=driver_color, linewidth=2, alpha=0.7,
                                          label=f"{piloto_nome} ({composto})")

            # ==========================================
            # 2. PROCESSAMENTO DO RANKING E COMBUSTÍVEL
            # ==========================================
            dados_ranking = []

            for driver in session.drivers:
                info_piloto = session.get_driver(driver)
                piloto_nome = info_piloto['Abbreviation']
                laps = session.laps.pick_driver(driver)

                voltas_soft = laps[laps['Compound'] == 'SOFT']
                voltas_soft_validas = voltas_soft.dropna(subset=['LapTime'])

                if not voltas_soft_validas.empty:
                    melhor_tempo = voltas_soft_validas['LapTime'].min()
                    tempo_best_soft = melhor_tempo.total_seconds() if pd.notnull(melhor_tempo) else None
                else:
                    tempo_best_soft = None

                for stint_id, stint_data in laps.groupby('Stint'):
                    clean_laps = stint_data.pick_quicklaps(threshold=1.07)

                    if len(clean_laps) >= min_voltas:
                        composto = stint_data['Compound'].iloc[0]
                        tempos_segundos = clean_laps['LapTime'].dt.total_seconds()

                        mediana = tempos_segundos.median()
                        voltas_validas = clean_laps[abs(tempos_segundos - mediana) <= 1.0]

                        if not voltas_validas.empty:
                            media_segundos = voltas_validas['LapTime'].dt.total_seconds().mean()
                            vida_final_pneu = int(clean_laps['TyreLife'].max())

                            if tempo_best_soft:
                                diff_total = media_segundos - tempo_best_soft
                                delta_soft_str = f"+{diff_total:.3f}s"

                                if composto == 'MEDIUM':
                                    tyre_delta = delta_sm
                                elif composto == 'HARD':
                                    tyre_delta = delta_sh
                                else:
                                    tyre_delta = 0.0

                                tempo_perdido_peso = diff_total - tyre_delta - delta_motor

                                if tempo_perdido_peso > 0:
                                    peso_estimado = (tempo_perdido_peso / peso_10kg) * 10
                                else:
                                    peso_estimado = np.nan
                            else:
                                delta_soft_str = "N/A"
                                peso_estimado = np.nan

                            dados_ranking.append({
                                'Piloto': piloto_nome,
                                'Pneu': composto,
                                'Voltas': len(voltas_validas),
                                'Vida Max': vida_final_pneu,
                                'media_raw': media_segundos,
                                'Ritmo Real (Médio)': formato_f1(media_segundos, None),
                                'Delta Soft': delta_soft_str,
                                'Combustível (kg)': peso_estimado
                            })

            # ==========================================
            # 3. RENDERIZAÇÃO ABAS 1 E 2
            # ==========================================
            ax_pace.yaxis.set_major_formatter(ticker.FuncFormatter(formato_f1))
            ax_pace.set_title(
                f"Race Sim: {session.get_driver(piloto1)['Abbreviation']} vs {session.get_driver(piloto2)['Abbreviation']}")
            ax_pace.set_xlabel("Vida do Pneu")
            ax_pace.set_ylabel("Tempo de Volta")
            ax_pace.legend()
            ax_pace.grid(True, alpha=0.3)

            ax_telemetry.set_title("Média de Acelerador por Volta")
            ax_telemetry.set_ylabel("Média do Acelerador (%)")
            ax_telemetry.set_xlabel("Vida do Pneu")
            ax_telemetry.legend()
            ax_telemetry.grid(True, alpha=0.3)

            with aba_pace:
                st.pyplot(fig_pace)
            with aba_telemetry:
                st.pyplot(fig_telemetry)

            # ==========================================
            # 4. RENDERIZAÇÃO ABAS 3 E 4
            # ==========================================
            if dados_ranking:
                df_ranking = pd.DataFrame(dados_ranking)
                df_ranking = df_ranking.sort_values(by='media_raw', ascending=True)

                peso_min_global = df_ranking['Combustível (kg)'].min()
                peso_max_global = df_ranking['Combustível (kg)'].max()

                # Renderiza Aba 3: Ranking Bruto
                with aba_ranking:
                    st.markdown("### 🏆 Ranking Bruto de Ritmo por Composto")

                    compostos_encontrados = df_ranking['Pneu'].unique()

                    for comp in ['SOFT', 'MEDIUM', 'HARD']:
                        if comp in compostos_encontrados:
                            df_comp = df_ranking[df_ranking['Pneu'] == comp].copy()

                            tempo_lider = df_comp['media_raw'].iloc[0]
                            gaps = ["Líder" if t == tempo_lider else f"+{(t - tempo_lider):.3f}s" for t in
                                    df_comp['media_raw']]

                            idx_ritmo = df_comp.columns.get_loc('Ritmo Real (Médio)')
                            df_comp.insert(idx_ritmo + 1, 'Gap Bruto', gaps)

                            df_comp = df_comp.drop(columns=['media_raw', 'Pneu']).reset_index(drop=True)
                            df_comp.index = df_comp.index + 1

                            st.markdown(
                                f"#### {'🟡' if comp == 'MEDIUM' else '⚪' if comp == 'HARD' else '🔴'} Pneu {comp}")

                            if df_comp['Combustível (kg)'].notna().any():
                                df_styled = df_comp.style.background_gradient(
                                    subset=['Combustível (kg)'], cmap='coolwarm', vmin=peso_min_global,
                                    vmax=peso_max_global
                                ).format({'Combustível (kg)': "{:.1f} kg"}, na_rep="Inconclusivo")
                            else:
                                df_styled = df_comp.style.format({'Combustível (kg)': "{:.1f} kg"},
                                                                 na_rep="Inconclusivo")

                            st.dataframe(df_styled, use_container_width=True)

                # Renderiza Aba 4: Gráfico Peso x Tempo + TABELA EQUALIZADA
                with aba_peso_tempo:
                    st.markdown("### ⚖️ Gráfico: Carga de Combustível vs Ritmo Bruto")

                    df_grafico = df_ranking.dropna(subset=['Combustível (kg)']).copy()

                    if not df_grafico.empty:
                        # 1. Gráfico
                        if peso_max_global > peso_min_global:
                            df_grafico['Carga Normalizada'] = (df_grafico['Combustível (kg)'] - peso_min_global) / (
                                        peso_max_global - peso_min_global)
                        else:
                            df_grafico['Carga Normalizada'] = 0.5

                        fig_pt, ax_pt = plt.subplots(figsize=(10, 6))
                        cores_pneus = {'SOFT': '#FF3333', 'MEDIUM': '#FFD100', 'HARD': '#DDDDDD'}

                        sns.scatterplot(
                            data=df_grafico, x='Carga Normalizada', y='media_raw', hue='Pneu',
                            palette=cores_pneus, s=150, edgecolor='black', alpha=0.8, ax=ax_pt
                        )

                        for i, row in df_grafico.iterrows():
                            ax_pt.text(row['Carga Normalizada'] + 0.015, row['media_raw'], row['Piloto'], fontsize=9,
                                       fontweight='bold', alpha=0.8)

                        ax_pt.yaxis.set_major_formatter(ticker.FuncFormatter(formato_f1))
                        ax_pt.set_xlabel("Índice de Carga Estimada (0.0 = Mais Leve, 1.0 = Mais Pesado)")
                        ax_pt.set_ylabel("Ritmo Médio Bruto")
                        ax_pt.grid(True, linestyle='--', alpha=0.4)
                        ax_pt.set_xlim(-0.1, 1.1)
                        ax_pt.invert_yaxis()

                        st.pyplot(fig_pt)

                        # ==========================================
                        # A NOVA TABELA EQUALIZADA (CORREÇÃO PELO MAIS PESADO)
                        # ==========================================
                        st.markdown("---")
                        st.markdown(f"### 🏁 Ordem de Forças Real (Ritmo Equalizado)")
                        st.markdown(
                            f"*(Simulando que todos os carros começaram com **{peso_max_global:.1f} kg**, que foi a carga máxima detectada na sessão).*")

                        # Prepara a tabela corrigida
                        df_corrigida = df_grafico.copy()

                        # Matemática: Quantos Kg de diferença para o líder de peso?
                        df_corrigida['Diferença Peso (kg)'] = peso_max_global - df_corrigida['Combustível (kg)']

                        # Penalidade de Tempo = (Diferença em Kg / 10) * Valor do Slidder
                        df_corrigida['Penalidade Tempo'] = (df_corrigida['Diferença Peso (kg)'] / 10) * peso_10kg

                        # O Ritmo Corrigido em segundos brutos
                        df_corrigida['Ritmo Corrigido (Raw)'] = df_corrigida['media_raw'] + df_corrigida[
                            'Penalidade Tempo']

                        # Ordena a tabela do mais rápido pro mais lento (já com os tempos corrigidos)
                        df_corrigida = df_corrigida.sort_values(by='Ritmo Corrigido (Raw)', ascending=True).reset_index(
                            drop=True)

                        # Calcula os Gaps Equalizados
                        tempo_lider_corrigido = df_corrigida['Ritmo Corrigido (Raw)'].iloc[0]
                        gaps_corrigidos = [
                            "Líder" if t == tempo_lider_corrigido else f"+{(t - tempo_lider_corrigido):.3f}s" for t in
                            df_corrigida['Ritmo Corrigido (Raw)']]

                        # Formatação para o usuário ver
                        df_corrigida['Gap Equalizado'] = gaps_corrigidos
                        df_corrigida['Ritmo Equalizado'] = df_corrigida['Ritmo Corrigido (Raw)'].apply(
                            lambda x: formato_f1(x, None))
                        df_corrigida['Penalidade Aplicada'] = df_corrigida['Penalidade Tempo'].apply(
                            lambda x: f"+{x:.3f}s" if x > 0 else "Base (Mais Pesado)")

                        # Seleciona só as colunas que importam para a tela
                        colunas_exibir = ['Piloto', 'Pneu', 'Combustível (kg)', 'Ritmo Real (Médio)',
                                          'Penalidade Aplicada', 'Ritmo Equalizado', 'Gap Equalizado']
                        df_tela = df_corrigida[colunas_exibir]
                        df_tela.index = df_tela.index + 1

                        # Renderiza com o mapa de cores no combustível para manter a identidade visual
                        df_tela_styled = df_tela.style.background_gradient(
                            subset=['Combustível (kg)'], cmap='coolwarm', vmin=peso_min_global, vmax=peso_max_global
                        ).format({'Combustível (kg)': "{:.1f} kg"})

                        st.dataframe(df_tela_styled, use_container_width=True)

                    else:
                        st.warning(
                            "Não há dados de combustível suficientes para gerar o gráfico e a tabela equalizada.")

            else:
                st.warning("Nenhum piloto atendeu aos critérios mínimos de análise.")

else:
    st.info("👈 Selecione um GP e clique em 'Carregar Sessão' para começar.")

