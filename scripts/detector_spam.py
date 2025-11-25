# ============================================================================
# DETECTOR DE SPAM EM E-MAILS - CLASSIFICAÇÃO SUPERVISIONADA
# Disciplina: Engenharia Transformacional
# Tecnologias: Python + Scikit-learn + CSV
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import os

# ============================================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================================

print("=" * 70)
print("DETECTOR DE SPAM EM E-MAILS")
print("Classificação Supervisionada com Scikit-learn")
print("=" * 70)
print()

# Carregar o arquivo CSV com os dados de treinamento
caminho_dados = "dados_emails.csv"

try:
    print(f"📂 Carregando dados de: {caminho_dados}")
    df = pd.read_csv(caminho_dados)

    print(f"✅ Total de e-mails carregados: {len(df)}")
    print(f"   - Não-spam (0): {(df['classificacao'] == 0).sum()}")
    print(f"   - Spam (1): {(df['classificacao'] == 1).sum()}")
    print()
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível ler o arquivo. Verifique se 'dados_emails.csv' está na pasta.")
    print(f"Detalhe do erro: {e}")
    exit()

# ============================================================================
# 2. SEPARAÇÃO DOS DADOS EM TREINO E TESTE
# ============================================================================

print("-" * 70)
print("ETAPA 2: Dividindo dados em conjunto de treino e teste")
print("-" * 70)

X = df['texto']  # Textos dos e-mails
y = df['classificacao']  # Classificação (0 ou 1)

# Dividir em 80% treino e 20% teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y 
)

print(f"✅ Conjunto de TREINO: {len(X_treino)} e-mails")
print(f"✅ Conjunto de TESTE: {len(X_teste)} e-mails")
print()

# ============================================================================
# 3. CRIAÇÃO E TREINAMENTO DO MODELO
# ============================================================================

print("-" * 70)
print("ETAPA 3: Criando e treinando o modelo de classificação")
print("-" * 70)
print()

# Pipeline: Vetorização + Classificador Naive Bayes
modelo = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,          
        max_features=1000,       
        ngram_range=(1, 2)       
    )),
    ('classificador', MultinomialNB())
])

print("🤖 Modelo: Pipeline com TF-IDF + Naive Bayes")
print("   - TfidfVectorizer: Converte texto em números")
print("   - MultinomialNB: Classificador probabilístico")
print()

# Treinar o modelo
print("🎓 Iniciando treinamento supervisionado...")
modelo.fit(X_treino, y_treino)
print("✅ Modelo treinado com sucesso! (O computador aprendeu os padrões)")
print()

# ============================================================================
# 4. AVALIAÇÃO DO MODELO
# ============================================================================

print("-" * 70)
print("ETAPA 4: Avaliando desempenho (A Prova)")
print("-" * 70)
print()

y_predito = modelo.predict(X_teste)

acuracia = accuracy_score(y_teste, y_predito)
print("📊 RESULTADO DA AVALIAÇÃO:")
print(f"   - Acurácia: {acuracia:.2%} (Porcentagem de acertos no teste)")
print()
print("📈 RELATÓRIO DETALHADO:")
print(classification_report(y_teste, y_predito, target_names=['Não-Spam', 'Spam']))

# ============================================================================
# 5. TESTE COM NOVOS E-MAILS (DEMONSTRAÇÃO)
# ============================================================================

print("-" * 70)
print("ETAPA 5: Teste ao Vivo (Simulação)")
print("-" * 70)
print()

novos_emails = [
    "Olá, como você está? Temos que conversar.",
    "Clique aqui e ganhe R$ 5000 hoje mesmo!",
    "Reunião importante amanhã às 14 horas.",
    "Voce foi selecionado para receber um prêmio",
    "Segue em anexo o relatório de vendas",
]

print("🧪 CLASSIFICANDO NOVOS E-MAILS:")
print()

for i, email in enumerate(novos_emails, 1):
    predicao = modelo.predict([email])[0]
    label = "SPAM ⚠️" if predicao == 1 else "NÃO-SPAM ✅"
    print(f"{i}. Email: \"{email}\" -> {label}")
    
print()
print("=" * 70)