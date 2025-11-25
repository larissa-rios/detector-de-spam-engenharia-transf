"""
ANÁLISE DE PALAVRAS MAIS FREQUENTES EM SPAM
Complemento ao Detector de Spam - Análise Exploratória

Este script identifica as palavras mais comuns em e-mails spam
versus e-mails legítimos, ajudando a entender os padrões.
"""

import pandas as pd
import re
from collections import Counter

print("="*70)
print("ANÁLISE DE PALAVRAS - SPAM vs LEGÍTIMO")
print("="*70)
print()

# Carregar dados
dados = pd.read_csv('dados_emails.csv')

# Separar spam e legítimos
spam = dados[dados['classificacao'] == 1]['texto']
legitimo = dados[dados['classificacao'] == 0]['texto']

print(f"Total de e-mails spam: {len(spam)}")
print(f"Total de e-mails legítimos: {len(legitimo)}")
print()

# Função para extrair palavras
def extrair_palavras(textos):
    """
    Extrai e processa palavras de uma coleção de textos
    """
    todas_palavras = []
    for texto in textos:
        # Converter para minúsculas e remover pontuação
        palavras = re.findall(r'\b\w+\b', texto.lower())
        todas_palavras.extend(palavras)
    return todas_palavras

# Extrair palavras
palavras_spam = extrair_palavras(spam)
palavras_legitimo = extrair_palavras(legitimo)

# Contar frequências
frequencia_spam = Counter(palavras_spam)
frequencia_legitimo = Counter(palavras_legitimo)

print("🚨 TOP 15 PALAVRAS EM SPAM:")
print("-" * 70)
for palavra, freq in frequencia_spam.most_common(15):
    print(f"  {palavra:20s} - {freq} ocorrências")

print()
print("📧 TOP 15 PALAVRAS EM LEGÍTIMOS:")
print("-" * 70)
for palavra, freq in frequencia_legitimo.most_common(15):
    print(f"  {palavra:20s} - {freq} ocorrências")

print()
print("✅ Análise concluída!")