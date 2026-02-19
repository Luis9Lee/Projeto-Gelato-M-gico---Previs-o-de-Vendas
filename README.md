# 🍦 Projeto Gelato Mágico: Previsão de Vendas com ML

Este projeto utiliza Machine Learning para prever a demanda diária de sorvetes com base na temperatura ambiente, otimizando a produção e evitando desperdícios.

## 🚀 Tecnologias e Conceitos Aplicados
* [cite_start]**Azure Machine Learning**: Gerenciamento de Workspaces e recursos de nuvem.
* **MLflow**: Utilizado para **Tracking** (rastreio de métricas) e **Registry** (versionamento do modelo).
* **Scikit-Learn**: Implementação do modelo de Regressão Linear.
* [cite_start]**Git/GitHub**: Versionamento de código e colaboração.

## [cite_start]📊 Ciclo de Vida do Projeto [cite: 24, 25]
1. **Inputs**: Dados históricos de temperatura e volume de vendas.
2. [cite_start]**Treinamento**: Executado em Notebooks Jupyter integrados com MLflow[cite: 106].
3. [cite_start]**Monitoramento**: Acompanhamento de métricas como MAE e RMSE para garantir a precisão das previsões[cite: 119].

## 💡 Insights
Durante o desenvolvimento, percebi que a escolha da métrica correta é fundamental. Diferente da **Acurácia** e **Recall** (usados em classificação), na regressão focamos no erro médio para garantir que o estoque da sorveteria seja o mais fiel possível à demanda real.
