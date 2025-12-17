# 📧 Detector de Spam - Engenharia Transformacional

Projeto de Classificação Supervisionada para identificar e-mails como **Spam** ou **Legítimo** automaticamente.

---

## 📂 Estrutura do Projeto

- `scripts/`: Pasta contendo os códigos Python e a base de dados.
- `requirements.txt`: Lista de bibliotecas necessárias.
- `README.md`: Guia de utilização.

---

## 🚀 Como Rodar (Passo a Passo)

Siga estes passos na ordem para garantir que funcione.

### Passo 0: Preparar o Ambiente

1. Abra a pasta do projeto no **VS Code**.
2. Abra o terminal (`Ctrl + '`).

### Passo 1: Instalar dependências (Apenas na 1ª vez)

Copie e cole este comando no terminal:

```bash
pip install -r requirements.txt
```

### Passo 2: Entrar na pasta correta (⚠️ IMPORTANTE)

O código e o arquivo CSV estão dentro da pasta `scripts`. Você **precisa** entrar nela pelo terminal antes de rodar qualquer coisa.

Copie e cole este comando:

```bash
cd scripts
```

_(Se der certo, o texto do seu terminal vai terminar com `.../scripts>`)_

### Passo 3: Rodar a Análise de Palavras

Este script mostra gráficos e as palavras mais comuns em Spams.

```bash
python analise_resultados.py
```

### Passo 4: Rodar o Detector (Principal)

Este script treina o modelo inteligente e classifica os e-mails.

```bash
python detector_spam.py
```

---

## 🧠 Resumo

1.  **O Problema:** Identificar mensagens indesejadas (Spam).
2.  **A Solução:** Um classificador usando **Python** e **Scikit-learn**.
3.  **Pipeline (O Processo):**
    - **Entrada:** Lemos o arquivo `dados_emails.csv`.
    - **Processamento:** Transformamos texto em números (TF-IDF).
    - **Inteligência:** O algoritmo **Naive Bayes** aprende as probabilidades.
    - **Teste:** O sistema classifica novos e-mails ao vivo.

---
