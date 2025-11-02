import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score
)

# =========================
# CONFIGURAÇÃO DO APP
# =========================

st.set_page_config(
    page_title="CardioFit Check",
    layout="wide"
)

st.title("CardioFit Check")
st.markdown("""
**Avaliação de Tolerância ao Esforço Cardíaco usando Machine Learning**

Este painel tem três objetivos principais:

1. **Entender a resposta do coração ao esforço físico**  
   (frequência cardíaca alcançada, dor no peito induzida por exercício, sinais de isquemia).

2. **Estimar o risco de baixa tolerância ao esforço ANTES do esforço**  

3. **Identificar perfis de resposta ao esforço**  

>
""")


# =========================
# CARREGAMENTO E FEATURE ENGINEERING
# =========================

@st.cache_data
def carregar_dados(caminho_csv: str):
    df_raw = pd.read_csv(caminho_csv)

    # Criar rótulo "low_tolerance": baixa tolerância ao esforço
    # Regras heurísticas:
    # - Frequência cardíaca máxima muito baixa (abaixo do percentil 25 da amostra)
    # - Dor no peito induzida por exercício (exang == 1)
    # - Queda significativa no segmento ST durante esforço (oldpeak >= 2.0)
    thalach_threshold = df_raw["thalach"].quantile(0.25)

    df_raw["low_tolerance"] = (
        (df_raw["thalach"] <= thalach_threshold) |
        (df_raw["exang"] == 1) |
        (df_raw["oldpeak"] >= 2.0)
    ).astype(int)

    return df_raw.copy(), thalach_threshold

df, thalach_threshold = carregar_dados("heart.csv")

# Mapas para rótulos
map_sexo = {0: "Feminino", 1: "Masculino"}
map_cp = {
    0: "Dor típica",
    1: "Dor atípica",
    2: "Dor não anginosa",
    3: "Assintomática"
}
map_exang = {0: "Não", 1: "Sim"}
map_slope = {
    0: "Inclinação Ascendente",
    1: "Inclinação Plana",
    2: "Inclinação Descendente"
}


# =========================
# FUNÇÕES AUXILIARES (EDA)
# =========================

def filtros_sidebar(df_original: pd.DataFrame):
    st.sidebar.header("🔎 Filtros da População Avaliada")

    # Filtro de sexo
    sexo_opcao = st.sidebar.selectbox(
        "Sexo biológico",
        ["Todos", "Feminino", "Masculino"]
    )

    # Filtro de idade
    idade_min = int(df_original["age"].min())
    idade_max = int(df_original["age"].max())
    faixa_idade = st.sidebar.slider(
        "Faixa etária (anos)",
        min_value=idade_min,
        max_value=idade_max,
        value=(idade_min, idade_max)
    )

    # Filtro por presença de baixa tolerância ao esforço
    tol_opcao = st.sidebar.selectbox(
        "Possível baixa tolerância ao esforço",
        ["Todos", "Baixa tolerância", "Sem baixa tolerância"]
    )

    return {
        "sexo": sexo_opcao,
        "faixa_idade": faixa_idade,
        "tolerancia": tol_opcao
    }


def aplicar_filtros(df_inicial: pd.DataFrame, filtros: dict):
    df_filtrado = df_inicial.copy()

    # Sexo
    if filtros["sexo"] == "Feminino":
        df_filtrado = df_filtrado[df_filtrado["sex"] == 0]
    elif filtros["sexo"] == "Masculino":
        df_filtrado = df_filtrado[df_filtrado["sex"] == 1]

    # Idade
    idade_min, idade_max = filtros["faixa_idade"]
    df_filtrado = df_filtrado[
        (df_filtrado["age"] >= idade_min) &
        (df_filtrado["age"] <= idade_max)
    ]

    # Baixa tolerância ao esforço
    if filtros["tolerancia"] == "Baixa tolerância":
        df_filtrado = df_filtrado[df_filtrado["low_tolerance"] == 1]
    elif filtros["tolerancia"] == "Sem baixa tolerância":
        df_filtrado = df_filtrado[df_filtrado["low_tolerance"] == 0]

    return df_filtrado


def kpi_populacao(df_view: pd.DataFrame, df_total: pd.DataFrame):
    st.subheader("Visão Geral do Grupo Selecionado")

    total_sel = len(df_view)
    if total_sel > 0:
        pct_low_tol = (df_view["low_tolerance"].mean()) * 100
        idade_media = df_view["age"].mean()
    else:
        pct_low_tol = 0.0
        idade_media = 0.0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Pacientes nesse recorte", total_sel)

    with col2:
        st.metric("Baixa tolerância ao esforço (%)", f"{pct_low_tol:.1f}%")

    with col3:
        st.metric("Idade média (anos)", f"{idade_media:.1f}")


def grafico_capacidade_cardiaca(df_view: pd.DataFrame, thalach_threshold: float):
    st.subheader("Capacidade Cardíaca sob Esforço")

    if len(df_view) == 0:
        st.info("Sem dados suficientes para mostrar este gráfico com os filtros atuais.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    hist = sns.histplot(
        df_view["thalach"],
        bins=20,
        kde=True,
        ax=ax,
        color="#4C78A8"
    )
    ax.axvline(
        thalach_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Limite de alerta (p25): {thalach_threshold:.0f} bpm"
    )
    ax.set_title("Distribuição da Frequência Cardíaca Máxima Alcançada no Esforço")
    ax.set_xlabel("Frequência cardíaca máxima atingida (bpm)")
    ax.set_ylabel("Número de pacientes")
    ax.legend()
    for container in hist.containers:
        ax.bar_label(container, fontsize=8)
    st.pyplot(fig)

    st.caption("""
    Quanto maior a frequência cardíaca máxima atingida (bpm), maior a capacidade cardiovascular de responder ao esforço.
    Valores muito baixos podem sugerir limitação da resposta cardíaca ao estresse físico.
    """)


def grafico_angina_por_faixa_etaria(df_view: pd.DataFrame):
    st.subheader("Dor no Peito Durante Exercício por Faixa Etária")

    if len(df_view) == 0:
        st.info("Sem dados suficientes para mostrar este gráfico com os filtros atuais.")
        return

    # Criar faixas etárias de 10 em 10 anos
    df_temp = df_view.copy()
    df_temp["faixa_idade"] = pd.cut(
        df_temp["age"],
        bins=[20, 30, 40, 50, 60, 70, 80, 90],
        right=False,
        include_lowest=True,
        labels=["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    )

    # Calcular % de pacientes que relatam angina induzida por exercício (exang == 1)
    angina_por_faixa = (
        df_temp.groupby("faixa_idade")["exang"]
        .mean()
        .fillna(0) * 100
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        angina_por_faixa.index.astype(str),
        angina_por_faixa.values,
        color="#E45756"
    )
    ax.set_title("Dor no Peito Durante Exercício (%) por Faixa Etária")
    ax.set_xlabel("Faixa etária (anos)")
    ax.set_ylabel("Pacientes com dor no peito sob esforço (%)")
    ax.set_ylim(0, max(angina_por_faixa.values.tolist() + [5]) * 1.2)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=8
        )
    st.pyplot(fig)

    st.caption("""
    Aqui avaliamos quantos pacientes relataram dor torácica induzida por exercício físico.
    Isso é relevante para triagem de segurança antes de atividade física intensa.
    """)


def grafico_isquemia_vs_idade(df_view: pd.DataFrame):
    st.subheader("Sinais de Isquemia Durante Esforço vs Idade")

    if len(df_view) == 0:
        st.info("Sem dados suficientes para mostrar este gráfico com os filtros atuais.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(
        x=df_view["age"],
        y=df_view["oldpeak"],
        hue=df_view["low_tolerance"].map({0: "Tolerância adequada", 1: "Baixa tolerância"}),
        palette=["#4C78A8", "#E45756"],
        alpha=0.7,
        ax=ax
    )
    ax.set_title("Queda do Segmento ST Durante Esforço vs Idade")
    ax.set_xlabel("Idade (anos)")
    ax.set_ylabel("Queda do ST durante esforço (oldpeak)")
    ax.legend(title="Classificação de tolerância ao esforço", loc="best")
    st.pyplot(fig)

    st.caption("""
    Quanto maior o valor de 'queda do ST', maior indicação de isquemia induzida pelo esforço.
    Pontos vermelhos indicam pacientes classificados como 'baixa tolerância ao esforço'.
    """)


def grafico_inclinacao_st(df_view: pd.DataFrame):
    st.subheader("Padrão da Resposta do Segmento ST Pós-Esforço")

    if len(df_view) == 0:
        st.info("Sem dados suficientes para mostrar este gráfico com os filtros atuais.")
        return

    # Contagem de cada padrão de inclinação
    slope_counts = (
        df_view["slope"]
        .map(map_slope)
        .value_counts()
        .reindex(map_slope.values(), fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        slope_counts.index,
        slope_counts.values,
        color="#72B7B2",
        alpha=0.8
    )

    ax.set_title("Padrão da Curva ST Após Esforço")
    ax.set_xlabel("Padrão observado")
    ax.set_ylabel("Número de pacientes")

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(slope_counts.values) * 0.02),
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout(pad=2.0)
    st.pyplot(fig)

    st.caption(
        "O formato da curva ST após esforço (ascendente/plana/descendente) "
        "é usado por cardiologistas para avaliar possíveis alterações isquêmicas."
    )


def heatmap_correlacao_esforco(df_view: pd.DataFrame):
    st.subheader("Correlação entre Indicadores de Esforço e Baixa Tolerância")

    if len(df_view) == 0:
        st.info("Sem dados suficientes para mostrar este gráfico com os filtros atuais.")
        return

    vars_esforco = df_view[[
        "thalach",       # frequência cardíaca máx atingida
        "exang",         # dor no peito induzida no exercício
        "oldpeak",       # queda ST
        "slope",         # padrão ST pós-esforço
        "low_tolerance"  # nosso rótulo criado
    ]].copy()

    traducao_labels = {
        "thalach": "Frequência Cardíaca Máx. (Esforço)",
        "exang": "Dor no Peito Durante Esforço",
        "oldpeak": "Queda do ST no Esforço (Isquemia)",
        "slope": "Padrão da Curva ST Pós-Esforço",
        "low_tolerance": "Baixa Tolerância ao Esforço"
    }

    corr = vars_esforco.corr(numeric_only=True)

    ordered_cols = list(traducao_labels.keys())
    corr = corr.loc[ordered_cols, ordered_cols]

    tick_labels_traduzidos = [traducao_labels[col] for col in ordered_cols]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        center=0,
        ax=ax,
        xticklabels=tick_labels_traduzidos,
        yticklabels=tick_labels_traduzidos
    )

    ax.set_title("Correlação entre variáveis de esforço e baixa tolerância", pad=16)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=20,
        ha="right",
        fontsize=9,
        wrap=True
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=9,
        wrap=True
    )

    plt.tight_layout()
    st.pyplot(fig)

    st.caption("""
    • Valores positivos fortes (mais vermelhos) indicam que as variáveis tendem a andar juntas.
    
    • "Baixa Tolerância ao Esforço" - alerta para possível limitação ao esforço.
    """)


# =========================
# PREPARAÇÃO PARA OS MODELOS
# =========================

# O atributo alvo é "low_tolerance"
# Ideia: prever quem tem baixa tolerância ao esforço antes de colocar a pessoa sob esforço

# Features "pré-esforço" (avaliáveis em repouso/consulta)
features_pre_esforco = [
    "age",        # Idade
    "sex",        # Sexo biológico
    "trestbps",   # Pressão arterial em repouso
    "chol",       # Colesterol
    "fbs",        # Açúcar no sangue em jejum
    "restecg",    # Eletrocardiograma em repouso
    "ca",         # Número de vasos principais visíveis
    "thal"        # Resultado do exame thal
]

X_screen = df[features_pre_esforco].copy()
y_screen = df["low_tolerance"].copy()

# Padronização
scaler_screen = StandardScaler()
X_screen_scaled = scaler_screen.fit_transform(X_screen)

# Divisão treino/teste
X_tr, X_te, y_tr, y_te = train_test_split(
    X_screen_scaled,
    y_screen,
    test_size=0.2,
    random_state=42
)

# Modelo supervisionado (classificação)
modelo_screen = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
modelo_screen.fit(X_tr, y_tr)

# Métricas de validação do modelo supervisionado
y_te_pred = modelo_screen.predict(X_te)
acc = accuracy_score(y_te, y_te_pred)
prec = precision_score(y_te, y_te_pred)
rec = recall_score(y_te, y_te_pred)
f1 = f1_score(y_te, y_te_pred)
mat_conf = confusion_matrix(y_te, y_te_pred)

# Agora, modelo não supervisionado de perfis de resposta ao esforço.
# Só sinais coletados DURANTE o esforço:
vars_para_cluster = ["thalach", "oldpeak", "exang", "slope"]

X_cluster_raw = df[vars_para_cluster].copy()
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster_raw)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_cluster_scaled)

# Métricas de clusterização
inercia = kmeans.inertia_
silhouette = silhouette_score(X_cluster_scaled, clusters)

df_cluster_view = df.copy()
df_cluster_view["perfil_esforco"] = clusters


def resumo_perfis(df_perfis: pd.DataFrame):
    """
    Gera um resumo por cluster (perfil de resposta ao esforço):
    - idade média
    - % baixa tolerância ao esforço
    - média da frequência máxima atingida
    - média da queda ST (oldpeak)
    - % que relatou dor no esforço (exang)
    """
    resumo = df_perfis.groupby("perfil_esforco").agg({
        "age": "mean",
        "low_tolerance": "mean",
        "thalach": "mean",
        "oldpeak": "mean",
        "exang": "mean"
    }).round(2)

    # Ordena por % de baixa tolerância (do menor para o maior)
    resumo = resumo.sort_values("low_tolerance")

    # Criar nomes pros clusters
    nomes = [
        "Alta tolerância ao esforço",
        "Tolerância moderada",
        "Sinais de intolerância ao esforço"
    ]

    mapping_cluster_to_name = {}
    for i, cluster_idx in enumerate(resumo.index):
        mapping_cluster_to_name[cluster_idx] = nomes[i]

    return resumo, mapping_cluster_to_name

cluster_summary, cluster_name_map = resumo_perfis(df_cluster_view)
name_to_cluster = {v: k for k, v in cluster_name_map.items()}


# =========================
# LAYOUT EM ABAS
# =========================

aba_eda, aba_sup, aba_unsup = st.tabs([
    "Resposta ao Esforço (EDA)",
    "Triagem Pré-Esforço",
    "Perfis de Esforço"
])


# =========================
# ABA 1: EDA (Resposta ao Esforço)
# =========================
with aba_eda:
    st.header("Resposta ao Esforço Físico")

    st.markdown("""
    Aqui analisamos **como o coração reage ao esforço físico**:
    - Frequência cardíaca máxima atingida
    - Dor no peito induzida por exercício
    - Sinais de isquemia
    - Padrão do segmento ST após esforço

    A ideia é entender **capacidade funcional cardiovascular**.
    """)

    filtros = filtros_sidebar(df)
    df_filtrado = aplicar_filtros(df, filtros)

    if len(df_filtrado) == 0:
        st.warning("Nenhum paciente corresponde a esses filtros. Mostrando todos os pacientes.")
        df_filtrado = df.copy()

    kpi_populacao(df_filtrado, df)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        grafico_capacidade_cardiaca(df_filtrado, thalach_threshold)
    with col2:
        grafico_angina_por_faixa_etaria(df_filtrado)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        grafico_isquemia_vs_idade(df_filtrado)
    with col4:
        grafico_inclinacao_st(df_filtrado)

    st.markdown("---")
    heatmap_correlacao_esforco(df_filtrado)


# =========================
# ABA 2: Modelo Supervisionado
# =========================
with aba_sup:
    st.header("Triagem Pré-Esforço")

    st.markdown("""
    Objetivo desta aba:
    - Estimar se o paciente tem sinal de **baixa tolerância ao esforço físico intenso**
      *antes mesmo* de fazer o teste de esforço.
    """)

    col_left, col_right = st.columns(2)
    entrada_user = {}

    with col_left:
        entrada_user["age"] = st.number_input(
            "Idade (anos)",
            min_value=20,
            max_value=100,
            value=55,
            help="Idade do paciente"
        )
        entrada_user["sex"] = st.selectbox(
            "Sexo biológico",
            options=[0, 1],
            format_func=lambda x: "Feminino" if x == 0 else "Masculino",
            help="0 = Feminino | 1 = Masculino"
        )
        entrada_user["trestbps"] = st.number_input(
            "Pressão arterial em repouso (mmHg)",
            min_value=80,
            max_value=220,
            value=130,
            help="Pressão sistólica medida em repouso"
        )
        entrada_user["chol"] = st.number_input(
            "Colesterol (mg/dL)",
            min_value=100,
            max_value=600,
            value=240,
            help="Colesterol sérico"
        )
        entrada_user["fbs"] = st.selectbox(
            "Açúcar no sangue em jejum > 120 mg/dL?",
            options=[0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="1 = Glicemia elevada em jejum"
        )

    with col_right:
        entrada_user["restecg"] = st.selectbox(
            "Eletrocardiograma em repouso",
            options=[0, 1, 2],
            format_func=lambda x: [
                "Normal",
                "Alteração ST-T",
                "Hipertrofia ventricular esquerda"
            ][x],
            help="Classificação do ECG em repouso"
        )
        entrada_user["ca"] = st.selectbox(
            "Vasos principais visíveis (0-3)",
            options=[0, 1, 2, 3],
            help="Número de vasos principais vistos na fluoroscopia"
        )
        entrada_user["thal"] = st.selectbox(
            "Resultado do exame Thal",
            options=[0, 1, 2, 3],
            format_func=lambda x: [
                "Normal",
                "Defeito fixo",
                "Defeito reversível",
                "Não disponível"
            ][x],
            help="Resultado do teste de perfusão miocárdica"
        )
        entrada_user["cp"] = st.selectbox(
            "Tipo de dor no peito relatada",
            options=[0, 1, 2, 3],
            format_func=lambda x: map_cp[x],
            help="Descrição do desconforto torácico"
        )

    # Monta dataframe com as features usadas pelo modelo
    entrada_df = pd.DataFrame([{
        "age": entrada_user["age"],
        "sex": entrada_user["sex"],
        "trestbps": entrada_user["trestbps"],
        "chol": entrada_user["chol"],
        "fbs": entrada_user["fbs"],
        "restecg": entrada_user["restecg"],
        "ca": entrada_user["ca"],
        "thal": entrada_user["thal"]
    }])

    entrada_scaled = scaler_screen.transform(entrada_df)

    st.markdown("---")
    if st.button("Estimar risco de baixa tolerância ao esforço", type="primary"):
        prob_low_tol = modelo_screen.predict_proba(entrada_scaled)[0][1]
        classe_prevista = modelo_screen.predict(entrada_scaled)[0]

        st.subheader("Resultado da estimativa")

        if classe_prevista == 1:
            st.error(
                f"Possível **baixa tolerância ao esforço físico intenso**.\n"
                f"Probabilidade estimada: {prob_low_tol*100:.1f}%"
            )
            st.caption("Sugestão: avaliar clinicamente antes de liberar para esforço intenso.")
        else:
            st.success(
                f"Perfil compatível com **tolerância adequada ao esforço**.\n"
                f"Probabilidade de baixa tolerância: {prob_low_tol*100:.1f}%"
            )
            st.caption("Sugestão: manter acompanhamento preventivo padrão.")

    # =========================
    # MÉTRICAS DO MODELO SUPERVISIONADO
    # =========================
    st.markdown("### Métricas do Modelo Supervisionado")

    st.write(f"**Acurácia:** {acc*100:.2f}%")
    st.write(f"**Precisão (Precision):** {prec*100:.2f}%")
    st.write(f"**Recall (Sensibilidade):** {rec*100:.2f}%")
    st.write(f"**F1-Score:** {f1*100:.2f}%")

    # Matriz de confusão
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(
        mat_conf,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax_cm
    )
    ax_cm.set_xlabel("Previsto (0 = tolerância ok | 1 = baixa tolerância)")
    ax_cm.set_ylabel("Real (0 = tolerância ok | 1 = baixa tolerância)")
    ax_cm.set_title("Matriz de Confusão – Modelo Supervisionado")
    st.pyplot(fig_cm)

    st.caption("""
    • Acurácia: % total de acertos.
    
    • Precisão: entre os que o modelo marcou como "baixa tolerância", quantos realmente eram.
    
    • Recall: quantos dos pacientes de "baixa tolerância" o modelo conseguiu identificar.
    
    • F1-Score: equilíbrio entre Precisão e Recall.
    
    • Matriz de confusão: mostra os acertos e erros em números absolutos.
    """)


# =========================
# ABA 3: Clusterização de Perfis de Esforço
# =========================
with aba_unsup:
    st.header("Perfis de Resposta ao Esforço")

    st.markdown("""
    Aqui usamos **agrupamento não supervisionado (K-Means)** nas variáveis coletadas DURANTE o esforço:
    - Frequência cardíaca máxima atingida
    - Dor no peito induzida por exercício
    - Queda do segmento ST
    - Padrão do segmento ST pós-esforço

    A ideia é: pacientes podem cair em **perfis funcionais** diferentes.  
    Isso ajuda a decidir quem precisa de atenção antes de atividades físicas mais intensas.
    """)

    # Visão geral de todos os clusters
    st.subheader("Visão Geral dos Perfis Encontrados")
    resumo = cluster_summary.copy()

    # Deixar os nomes dos clusters legíveis no índice
    resumo.index = [cluster_name_map[i] for i in resumo.index]

    # Renomeando colunas
    resumo = resumo.rename(columns={
        "age": "Idade média (anos)",
        "low_tolerance": "% Baixa Tolerância ao Esforço",
        "thalach": "Frequência Cardíaca Máx Média (bpm)",
        "oldpeak": "Queda ST Média (isquemia no esforço)",
        "exang": "% Dor no Peito no Esforço"
    })

    # Transformar colunas de proporção em %
    if "% Baixa Tolerância ao Esforço" in resumo.columns:
        resumo["% Baixa Tolerância ao Esforço"] = (resumo["% Baixa Tolerância ao Esforço"] * 100).round(1)
    if "% Dor no Peito no Esforço" in resumo.columns:
        resumo["% Dor no Peito no Esforço"] = (resumo["% Dor no Peito no Esforço"] * 100).round(1)

    st.dataframe(resumo)

    # Escolher um perfil pra detalhar
    perfil_escolhido = st.selectbox(
        "Escolha um perfil para entender melhor:",
        options=list(name_to_cluster.keys())
    )

    cluster_id = name_to_cluster[perfil_escolhido]
    subset_cluster = df_cluster_view[df_cluster_view["perfil_esforco"] == cluster_id]

    st.markdown("---")
    if "Alta tolerância" in perfil_escolhido:
        st.success(f"🟢 {perfil_escolhido}")
    elif "Moderada" in perfil_escolhido:
        st.warning(f"🟡 {perfil_escolhido}")
    else:
        st.error(f"🔴 {perfil_escolhido}")

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Pacientes neste perfil", len(subset_cluster))
    with colB:
        st.metric("Idade média (anos)", f"{subset_cluster['age'].mean():.1f}")
    with colC:
        pct_baixa_tol = subset_cluster["low_tolerance"].mean() * 100
        st.metric("Baixa tolerância ao esforço (%)", f"{pct_baixa_tol:.1f}%")

    st.subheader("Distribuição da Frequência Cardíaca Máxima (bpm)")
    fig_fc, ax_fc = plt.subplots(figsize=(6, 4))
    sns.histplot(subset_cluster["thalach"], bins=15, kde=True, ax=ax_fc, color="#4C78A8")
    ax_fc.set_title("Frequência cardíaca máxima atingida no esforço")
    ax_fc.set_xlabel("Frequência cardíaca máxima (bpm)")
    ax_fc.set_ylabel("Número de pacientes")
    st.pyplot(fig_fc)

    st.subheader("Distribuição da Queda ST (Isquemia induzida)")
    fig_st, ax_st = plt.subplots(figsize=(6, 4))
    sns.histplot(subset_cluster["oldpeak"], bins=15, kde=True, ax=ax_st, color="#E45756")
    ax_st.set_title("Queda do segmento ST durante esforço (oldpeak)")
    ax_st.set_xlabel("Queda ST (maior = mais isquemia)")
    ax_st.set_ylabel("Número de pacientes")
    st.pyplot(fig_st)

    st.markdown("---")
    st.markdown("### Métricas do Agrupamento (Modelo Não Supervisionado)")

    # Formatação
    inercia_fmt = f"{inercia:,.0f}".replace(",", ".")
    silhouette_fmt = f"{silhouette:.3f}"

    st.write(f"**Inércia (coerência interna dos clusters):** {inercia_fmt}")
    st.write(f"**Coeficiente de Silhouette (separação entre clusters):** {silhouette_fmt}")

    st.caption("""
    • Inércia: mede o quão compactos são os grupos. Valores menores indicam pacientes mais parecidos dentro de cada grupo.

    • Silhouette: varia de -1 a 1. Quanto mais próximo de 1, melhor separados e mais "distintos" estão os perfis.
    """)

