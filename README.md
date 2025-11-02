# CardioFit Check  
## Avaliação de Tolerância ao Esforço Cardíaco com Machine Learning  

---

### Resumo  

O **CardioFit Check** é uma aplicação interativa desenvolvida em **Streamlit** que utiliza técnicas de **Aprendizado de Máquina supervisionadas e não supervisionadas** para analisar a **resposta cardíaca ao esforço físico**.  

O projeto tem como propósito **apoiar a avaliação de tolerância ao esforço** e **identificar possíveis limitações cardiovasculares** antes e durante o exercício — sempre com caráter **educacional e preditivo**, não diagnóstico médico.  

---

### Objetivo  

- Analisar como diferentes variáveis fisiológicas se comportam durante o esforço físico;  
- Estimar o risco de baixa tolerância ao esforço a partir de dados clínicos básicos (modelo supervisionado);  
- Agrupar pacientes em perfis funcionais de resposta ao esforço (modelo não supervisionado);  
- Comunicar resultados de forma visual, intuitiva e acessível.  

---

### Principais Etapas do Projeto  

#### 1. Exploração dos Dados (EDA)  
- Limpeza e mapeamento de variáveis clínicas;  
- Criação de um novo rótulo: **low_tolerance**, baseado em indicadores como frequência cardíaca máxima, dor no peito ao esforço e isquemia (queda ST);  
- Visualizações com histograma, dispersão e heatmaps para análise de padrões.  

#### 2. Aprendizagem Supervisionada (Classificação)  
- **Modelo:** Random Forest Classifier  
- **Objetivo:** prever se o paciente pode apresentar baixa tolerância ao esforço físico.  
- **Métricas apresentadas:**  
  - Acurácia nos dados de teste;  
  - Probabilidade estimada de baixa tolerância para cada novo paciente.  

#### 3. Aprendizagem Não Supervisionada (Clusterização)  
- **Modelo:** K-Means (3 clusters)  
- **Objetivo:** identificar perfis de resposta ao esforço com base em variáveis observadas durante o teste (frequência máxima, dor no peito, ST, etc).  
- **Métricas avaliadas:**  
  - Inércia (compactação dos grupos);  
  - Coeficiente de Silhouette (separação entre clusters).  

#### 4. Interface Interativa (Streamlit)  
- Filtros dinâmicos de idade, sexo e tolerância;  
- Visualização de KPIs, gráficos comparativos e distribuições;  
- Estimativa de risco personalizada e dashboards intuitivos.  

---

### Tecnologias Utilizadas  

- **Python**  
- **Streamlit** – Interface interativa e visualização  
- **Pandas**, **NumPy** – Manipulação e análise de dados  
- **Matplotlib**, **Seaborn** – Visualização gráfica e EDA  
- **Scikit-learn** – Modelagem supervisionada e clusterização (K-Means, Random Forest)  

---

### Dataset  

O projeto utiliza o **Heart Disease Dataset (Kaggle)**, baseado nos estudos de **Cleveland, Hungria, Suíça e Long Beach**.  
Ele contém variáveis clínicas relacionadas à função cardíaca, como:  

- Idade e sexo;  
- Pressão arterial, colesterol e glicemia;  
- Resultados de eletrocardiograma;  
- Frequência cardíaca máxima, dor no peito e alterações do segmento ST.  