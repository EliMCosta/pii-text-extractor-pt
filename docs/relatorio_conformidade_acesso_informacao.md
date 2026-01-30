# Relatório de conformidade — Categoria “Acesso à Informação” (Edital DODF nº 10/2025 — Desafio Participa DF)

- **Projeto avaliado**: `pii-text-identifier`
- **Base analisada (commit)**: `8c6546babefc61906fffa907e06fe4e7ced5d68d` (working tree com modificações locais)
- **Fonte normativa**: Edital “EDITAL N° 10, DE 24 DE NOVEMBRO DE 2025” (DODF) em: `https://www.dodf.df.gov.br/dodf/materia/visualizar?co_data=550905&p=edital-n-10de-24-de-novembro-de-2025`

## 1) Escopo e método

Este relatório verifica, **ponto a ponto**, a aderência do repositório aos requisitos aplicáveis à **categoria “Acesso à Informação”**, conforme itens do Edital (principalmente **2.2**, **6.6**, **7.2** e **8.1**).

- **O que foi verificado no repositório**: `README.md`, `docs/requirements.md`, scripts de inferência/avaliação (`infer_pii.py`, `inference/*`), taxonomia (`ner_labels.py`), dependências (`requirements.txt`) e presença/conteúdo de datasets em `data/`.
- **Como os itens foram marcados**:
  - **Conforme**: atende integralmente ao requisito.
  - **Parcial**: atende em grande parte, mas falta um aspecto relevante.
  - **Não conforme / risco**: não atende e/ou cria risco direto frente ao que o Edital exige/afirma.
  - **Não verificável aqui**: depende do host (ex.: GitHub/GitLab público), histórico de commits, ou ato de submissão.

## 2) Resumo executivo

- **Aderência funcional ao objetivo da categoria**: **Conforme** (o projeto implementa detecção de PII em pedidos LAI/e-SIC e deriva sinal de “público vs não público”).
- **Documentação (P2)**: **Conforme em alto grau** (há pré-requisitos, dependências, comandos e organização); há **pontos de melhoria** para tornar o formato de I/O ainda mais explícito no `README.md`.
- **Risco crítico**: o Edital explicita que **“Nenhum dado pessoal real será disponibilizado aos participantes”** (item **8.1.1**). Este repositório contém textos com **CPFs, telefones, e-mails, nomes e endereços** em `data/` (ex.: `data/esic_sample_eval.jsonl`, `data/csv/AMOSTRA_e-SIC.csv`), o que pode contrariar o espírito do item e/ou gerar risco de conformidade (LGPD, governança de dados e eventual desclassificação dependendo de regras do desafio e do ambiente de publicação).

## 2.1) Trechos do Edital (citações-chave)

> **2.2 (I)** — “Acesso à Informação: desenvolvimento de solução para identificar automaticamente, entre os pedidos de acesso à informação marcados como públicos, aqueles que contenham dados pessoais e que, portanto, deveriam ser classificados como não públicos. (…)”

> **6.6** — “O repositório deverá estar configurado como público, permitindo acesso para visualização e clonagem sem necessidade de autenticação. (…)”

> **8.1.1** — “Parte dessas manifestações conterá dados sintéticos (…) (…) **Nenhum dado pessoal real será disponibilizado aos participantes.**”

> **8.1.5.3** — P2 avalia clareza de instalação/configuração/execução e documentação (tabela com itens 1(a–c), 2(a–b), 3(a–c)).

## 3) Matriz de conformidade (ponto a ponto)

### 3.1 Requisitos centrais da categoria “Acesso à Informação”

| Item do Edital | Requisito | Evidência no projeto | Status | Observações / lacunas |
|---|---|---|---|---|
| **2.2 (I)** | Solução deve identificar, dentre pedidos marcados como públicos, os que contêm dados pessoais e deveriam ser “não públicos”; define dados pessoais (nome, CPF, RG, telefone, e-mail) | `README.md` posiciona o projeto como identificador de PII em pedidos LAI/e-SIC; `infer_pii.py` produz `should_be_public` com base em `PII_TYPES` | **Conforme** | A lógica de “público/não público” está implementada em nível de documento, derivada de spans. |
| **8.1.1** | Conjunto amostral será disponibilizado; parte com dados sintéticos; **“Nenhum dado pessoal real será disponibilizado aos participantes.”** | Existem datasets versionados em `data/` contendo PII explícita (ex.: CPFs, telefones, e-mails) | **Não conforme / risco** | Mesmo que parte seja sintética, o repositório **não comprova** ausência de dado pessoal real; além disso, `AMOSTRA_e-SIC.csv` se declara “Texto Mascarado”, mas contém identificadores. Recomenda-se saneamento/remoção (ver seção 4). |
| **8.1.2** | Participantes devem submeter modelos desenvolvidos e documentação técnica | Repositório contém scripts de treino, inferência e avaliação; `outputs/` é ignorado por `.gitignore` | **Parcial** | Há documentação e pipeline, mas **não há artefato de modelo** versionado (checkpoint) nem instrução clara de como disponibilizá-lo para avaliação sem re-treinar (ex.: link para release/model hub). |
| **8.1.5.2 – 8.1.5.2.3** | P1 medido por Precisão, Sensibilidade/Recall e \(F1\) | `infer_pii.py` e `inference/eval_report.py` computam métricas (token/span/binary), e `README.md` descreve avaliação | **Conforme** | O projeto vai além do mínimo ao produzir relatório detalhado (`outputs/eval_report_*.md`). |
| **8.1.5.4** | Desempate: menor FN, menor FP, maior P1 | `infer_pii.py` calcula FN/FP (nível doc e token), e reporta buckets `tp/tn/fp/fn` | **Conforme** | Alinhado ao racional do edital; útil para tuning de threshold. |

### 3.2 Requisitos de submissão/entrega que impactam diretamente a categoria

| Item do Edital | Requisito | Evidência no projeto | Status | Observações / lacunas |
|---|---|---|---|---|
| **6.6** | Repositório deve estar **público** (visualização e clonagem sem autenticação) | Não verificável localmente | **Não verificável aqui** | Verificar configuração no GitHub/GitLab antes da submissão. |
| **6.6.1** | Commits/uploads após submissão “não serão considerados” | `git status` indica working tree modificada; histórico de submissão não está no repo | **Não verificável aqui** | Requisito processual: depende da data/hora do formulário e do estado do repositório remoto no momento. |
| **7.2** | Infraestrutura do participante (CGDF não fornece) | `README.md` exige execução local (Python/venv; GPU opcional para treino) | **Conforme** | O projeto assume execução local e trata GPU/CPU. |

### 3.3 Pontuação P2 — Documentação (tabela do Edital 8.1.5.3)

| Critério (Edital 8.1.5.3) | Requisito | Evidência no projeto | Status | Observações / lacunas |
|---|---|---|---|---|
| **1(a)** | Lista pré-requisitos | `README.md` lista Linux, Python 3.12, `uv`, GPU CUDA (para treino) | **Conforme** | — |
| **1(b)** | Arquivo de dependências automatizado | `requirements.txt` | **Conforme** | — |
| **1(c)** | Comandos exatos para criar/configurar ambiente | `README.md` com `python -m venv` e `uv pip install -r requirements.txt` | **Conforme** | — |
| **2(a)** | Comandos exatos para executar (com args) | `README.md` com exemplos de geração/treino/inferência/avaliação | **Conforme** | — |
| **2(b)** | Formato de entrada e saída | `README.md` descreve JSONL (`text`, `entities`) e flags; `infer_pii.py` define saída com `spans` e `should_be_public` | **Parcial** | Sugere-se explicitar no `README.md` o schema da saída JSON (campos e tipos), inclusive modo `--jsonl_in/--jsonl_out`. |
| **3(a)** | README descreve objetivo e função dos arquivos importantes | `README.md` seção “Estrutura do Projeto” | **Conforme** | — |
| **3(b)** | Código com comentários em trechos complexos | Há docstrings e comentários (ex.: `infer_pii.py`) | **Conforme** | — |
| **3(c)** | Estrutura lógica e organizada | Pastas `training/`, `inference/`, `data_preprocessing/`, `synth_dataset/` | **Conforme** | — |

## 4) Achados críticos e recomendações objetivas

### 4.1 Risco: presença de dados pessoais em `data/`

O Edital (item **8.1.1**) afirma que **nenhum dado pessoal real** seria disponibilizado. No repositório, há material com PII explícita (ex.: CPFs/telefones/e-mails/nome/endereço) em:

- `data/esic_sample_eval.jsonl`
- `data/csv/AMOSTRA_e-SIC.csv` (a coluna se chama “Texto Mascarado”, mas há CPFs e outros identificadores no texto)

**Recomendação (alta prioridade)**:
- Remover do repositório qualquer dado que possa ser **real** (ou cuja origem não esteja provada como sintética) e substituir por:
  - **dataset sintético** gerado pelo próprio pipeline (`synth_dataset/*`) ou
  - **instrução de obtenção** do dataset oficial (quando aplicável), sem redistribuí-lo no repositório público.
- Se a intenção for manter exemplos, publicar apenas **excertos anonimizados** e/ou amostras **com PII irrecuperável**.

### 4.2 Submissão “modelo + documentação” (8.1.2)

**Recomendação (média/alta prioridade)**:
- Definir claramente o **artefato de modelo** a ser avaliado (ex.: pasta `outputs/.../best`, link para Hugging Face Hub, ou release com checkpoint) e como a comissão deve carregá-lo via `--model_name_or_path`.
- Manter `.gitignore` para artefatos pesados é correto, mas precisa de um caminho de distribuição **reprodutível**.

### 4.3 Melhorar explicitação de I/O no `README.md` (P2 2(b))

**Recomendação (média prioridade)**:
- Adicionar uma seção curta “Formato de entrada e saída” com exemplos mínimos:
  - **Entrada**: JSONL com `text: string` (e opcionalmente `entities` para eval)
  - **Saída**: JSONL/JSON com `spans: [{type,start,end,value}]` e `should_be_public: bool`

## 5) Conclusão

O projeto está **alinhado tecnicamente** ao objetivo e à metodologia da categoria “Acesso à Informação” e atende **muito bem** aos itens de documentação (P2), com pequenas melhorias recomendadas para deixar o I/O explícito no `README.md`.

O principal ponto de atenção é a **governança do dataset versionado em `data/`**: do jeito que está, há risco de conflito com a diretriz do Edital sobre **não disponibilizar dados pessoais reais**, além de potenciais implicações de LGPD ao publicar repositório como **público** (item 6.6).

