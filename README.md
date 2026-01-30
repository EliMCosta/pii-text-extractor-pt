# PII Text Extractor (Extrator de Dados Pessoais)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EliMCosta/pii-text-extractor-pt/blob/main/notebooks/pii_extractor_colab.ipynb)

Este projeto utiliza **modelos de linguagem (NLP)** para identificar automaticamente pedidos de acesso à informação que contenham **dados pessoais** (PII - Personally Identifiable Information) em requerimentos de e-SIC/LAI. A ferramenta extrai entidades sensíveis e classifica se o texto pode ou não ser tornado público, garantindo que a informação circule sob o princípio do *need-to-know* (necessidade de saber).

### Interface e Expectativas de Dados

- **Entrada**: Texto bruto (strings) ou arquivos JSONL (campo `text`) contendo requerimentos administrativos, manifestações de ouvidoria ou qualquer texto em linguagem natural. Graças ao **Smart Chunking**, o sistema lida nativamente com textos de qualquer extensão (desde frases curtas até documentos de várias páginas).
- **Saída**: Estrutura JSON padronizada contendo:
  - `spans`: Lista de entidades detectadas, incluindo o tipo (ex: `DOC_PESSOAL`, `NOME_PESSOA`), índices de início/fim, valor literal e confiança da predição.
  - `should_be_public`: Booleano calculado automaticamente. É `false` se qualquer PII for detectada e `true` caso o texto contenha apenas informações impessoais (ou entidades não-sensíveis como nomes de órgãos públicos).

---

Tecnicamente, a solução consiste em um modelo **BERT** (`neuralmind/bert-base-portuguese-cased`) com treinamento de ajuste para a tarefa de **Named Entity Recognition (NER)** via **Token Classification**. O pipeline inclui uma etapa de **Smart Chunking** que fragmenta textos longos respeitando fronteiras de palavras e sentenças, e técnicas de pós-processamento para garantir sequências válidas. Todo o pós-processamento é baseado em NumPy, mantendo compatibilidade com exportação **ONNX/TensorRT** para implantação em produção.


## Motivação/Problema

A Lei de Acesso à Informação (LAI) e os sistemas de e-SIC são pilares da transparência pública, mas sua operação enfrenta um desafio crítico de conformidade com a LGPD:

1. **Conflito Transparência vs. Privacidade**: Requerimentos frequentemente contêm dados pessoais (CPFs, endereços, contatos) que, se publicados sem tratamento, violam a privacidade do cidadão.
2. **Princípio da Necessidade e Acesso Restrito**: Segundo a LGPD, o dado pessoal deve ser acessível somente a quem realmente precisa tratá-lo para responder ao pedido. Isso exige que o dado seja identificado e classificado não apenas para exclusão total posterior, mas para **redação seletiva**: protegendo a identidade do solicitante em fluxos de trabalho internos e em consultas públicas, enquanto preserva a informação para o servidor responsável pelo atendimento.
3. **Escalabilidade e Erro Humano**: Com o [aumento expressivo de pedidos](https://www.ouvidoria.df.gov.br/distrito-federal-registra-crescimento-de-107-nos-pedidos-de-acesso-a-informacao/), a triagem manual é lenta e falha. Um único dado não mascarado (vazamento) pode gerar sanções e danos à reputação institucional.
4. **Especificidade do Domínio**: Identificadores brasileiros (RG, SIAPE, números de processos SEI/PJE) possuem formatos que ferramentas genéricas de PII falham em detectar.


---

## 1. Instalação e Configuração

### Pré-requisitos
- Linux
- Python 3.12 instalado no host
- Pacote uv  (https://docs.astral.sh/uv/getting-started/installation/)
- GPU NVIDIA com suporte a CUDA (para treinamento)

### Configuração do Ambiente
1. Crie um ambiente virtual:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

2. Instale as dependências:

   ```bash
   uv pip install -r requirements.txt
   ```
---

## 2. Dataset

O projeto utiliza o dataset [EliMC/esic-ner](https://huggingface.co/datasets/EliMC/esic-ner) do Hugging Face, que já está no formato compatível com o pipeline de fine-tuning.

### Formato do Dataset

Cada linha do dataset contém:
```json
{"text": "<str>", "entities": [{"type": "<str>", "value": "<str>"}, ...]}
```

### Download e Preparação

O dataset pode ser baixado diretamente via Hugging Face Datasets:

```bash
mkdir data && cd data
git clone https://huggingface.co/datasets/EliMC/esic-ner && cd ..
```

---

## 3. Processamento e Curadoria do Dataset (`data_preprocessing`)

### 3.1 Revisão de Valores PII (`pii_value_review.py`) [OPCIONAL]

Este script implementa um workflow de revisão para garantir a qualidade das anotações no dataset. **Esta etapa é opcional** e recomendada caso você queira fazer uma curadoria manual fina dos valores identificados (geralmente por um LLM) antes de prosseguir para o treinamento. Ele permite extrair todos os valores únicos anotados (agrupados por label) para revisão humana e, após a revisão, aplicar as correções ao dataset original.

#### Como executar:

**1. Extração para revisão:**
Gera um arquivo JSON com o mapeamento `label -> [valores_únicos]`. Use a flag `--pii-only` para focar apenas em dados pessoais (excluindo labels como `ORG_JURIDICA`).

```bash
python data_preprocessing/pii_value_review.py extract \
    --input data/esic-ner/train.jsonl \
    --output data/review/values_to_review.json \
    --normalize \
    --pii-only
```

**2. Aplicação da revisão:**
Lê o JSON revisado e reescreve o dataset filtrando entidades cujos valores não foram aprovados. A flag `--add-missing` permite que o script identifique e anote valores da lista de revisão que aparecem no texto mas não foram originalmente marcados pelo LLM.

```bash
python data_preprocessing/pii_value_review.py apply \
    --input data/esic-ner/train.jsonl \
    --review data/review/values_reviewed.json \
    --output data/esic-ner/train_cleaned.jsonl \
    --normalize \
    --add-missing
```

#### Argumentos principais:
- `extract / apply`: Subcomandos para extração ou aplicação.
- `--normalize`: Sanitiza valores (remove prefixos como "CPF:", "RG:", etc.) antes de processar.
- `--migrate-company-ids`: Migra heuristicamente identificadores de empresas (CNPJ/IE) que o LLM possa ter classificado erroneamente como `ORG_JURIDICA` para `DOC_EMPRESA`.
- `--add-missing` (apenas `apply`): Recupera entidades presentes no texto que constam na lista de revisão mas não estavam no dataset original.
- `--require-all-labels` (apenas `apply`): Força que todos os labels presentes no dataset estejam no arquivo de revisão.
- `--allow-value-in-multiple-labels` (apenas `apply`): Permite que um mesmo valor literal seja mapeado para diferentes tipos de entidade.
- `--pii-only` (apenas `extract`): Filtra apenas labels de dados pessoais.

---

### 3.2 Preparação para Fine-tuning (`build_finetune_jsonl.py`)

O script `data_preprocessing/build_finetune_jsonl.py` converte o dataset em um formato adequado para o fine-tuning. Ele utiliza o módulo compartilhado `data_preprocessing/chunking.py` para realizar o **Smart Chunking** dos textos, mapeando os candidatos a PII para cada fragmento e garantindo consistência total entre treino e inferência.

### Vantagens do Smart Chunking:
- **Respeito a Fronteiras de Palavras**: Evita que chunks comecem ou terminem no meio de uma subword ou palavra, reduzindo ruído no aprendizado.
- **Preferência por Fins de Sentença**: O algoritmo busca encerrar o chunk em sinais de pontuação (`.`, `!`, `?`) ou quebras de linha quando está próximo ao limite máximo de tokens (`max_length`).
- **Sem Lacunas de Caracteres**: Inclui separadores (espaços/pontuação) residuais entre tokens, garantindo que a união dos chunks recupere o texto original sem perda de informação.
- **Alinhamento de Inferência**: Garante que o modelo veja o texto exatamente da mesma forma durante o treinamento e o uso real.

### Como executar:
```bash
python data_preprocessing/build_finetune_jsonl.py \
    --input data/esic-ner/train.jsonl \
    --output data/esic-ner/train_chunks.jsonl \
    --max_length 512 \
    --stride 64
```

### Argumentos principais:
- `--input`: Caminho para o JSONL de entrada (default: `data/esic-ner/train.jsonl`).
- `--output`: Caminho para o arquivo processado (default: `data/esic-ner/train_chunks.jsonl`).
- `--model_name_or_path`: Tokenizer a ser utilizado (default: `neuralmind/bert-base-portuguese-cased`).
- `--max_length`: Comprimento máximo da sequência (default: 512).
- `--stride`: Sobreposição (overlap) de tokens entre chunks consecutivos (default: 64).
- `--boundary_backoff`: Janela de busca para evitar cortes de chunks no meio de palavras ou sentenças (default: 32).

---

## 4. Fine-tuning - Token Classification (`training`)

O modelo é especializado na tarefa de extração de **entidades** (NER). O script `training/finetune_pii_token_classification.py` mapeia `entities` para labels no formato IOB (ex: `B-DOC_PESSOAL`, `I-DOC_PESSOAL`) e treina uma cabeça de classificação sobre o encoder. A taxonomia de rótulos está definida em `ner_labels.py`.

### Como executar:

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
    training/finetune_pii_token_classification.py \
    --model_name_or_path neuralmind/bert-base-portuguese-cased \
    --dataset_path data/esic-ner/train_chunks.jsonl \
    --output_dir outputs/pii-textx-pt \
    --num_train_epochs 3.0 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5

# Multi-GPU (2 GPUs example)
accelerate launch --num_processes 2 \
    training/finetune_pii_token_classification.py \
    --model_name_or_path neuralmind/bert-base-portuguese-cased \
    --dataset_path data/esic-ner/train_chunks.jsonl \
    --output_dir outputs/pii-textx-pt \
    --num_train_epochs 3.0 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5
```

### Argumentos principais:
- `--model_name_or_path`: Caminho do modelo base (default: `neuralmind/bert-base-portuguese-cased`).
- `--dataset_path`: Caminho do JSONL processado (default: `data/esic-ner/train_chunks.jsonl`).
- `--output_dir`: Diretório de saída (default: `outputs/pii-textx-pt`).
- `--best_checkpoint_dir`: Diretório para o melhor checkpoint final (default: `outputs/pii-textx-pt/best`).
- `--text_column` / `--entities_column`: Nomes das chaves no JSONL (default: `text` e `entities`).
- `--validation_split`: Proporção do dataset para validação (default: 0.10).
- `--max_length`: Comprimento máximo da sequência (default: 512).
- `--num_train_epochs`: Número de épocas de fine-tuning (default: 3.0).
- `--learning_rate`: Taxa de aprendizado (default: 5e-5).
- `--bf16` ou `--fp16`: Habilita precisão mista.
- `--resume_from_checkpoint`: Caminho para retomar o treino de um checkpoint anterior.

---

## 5. Inferência e Avaliação

O script `infer_pii.py` permite utilizar o modelo treinado para extrair entidades de novos textos ou avaliar o desempenho em datasets de teste. Ele implementa a mesma lógica de chunking utilizada no treino, garantindo resultados consistentes mesmo em textos longos.

Nota: `ORG_JURIDICA` e `DOC_EMPRESA` são **entidades úteis** para aprendizagem, mas são tratadas como **não-PII**. Assim, em inferência/avaliação, elas podem aparecer em `spans`, porém **não contam** para `should_be_public` nem para as métricas **PII-only**.

### Exemplos de Inferência

**1. Inferência simples em texto (stdout):**
```bash
python infer_pii.py \
    --model_name_or_path outputs/pii-textx-pt/best \
    infer \
    --text "O CPF do solicitante João Silva é 123.456.789-00."
```

Output esperado:
```json
{
  "text": "O CPF do solicitante João Silva é 123.456.789-00.",
  "spans": [
    {"type": "NOME_PESSOA", "start": 21, "end": 31, "value": "João Silva", "conf": 0.99},
    {"type": "DOC_PESSOAL", "start": 34, "end": 48, "value": "123.456.789-00", "conf": 0.99}
  ],
  "should_be_public": false
}
```

**2. Inferência com alta precisão (threshold de confiança):**
Útil para evitar falsos positivos em produção.
```bash
python infer_pii.py \
    --model_name_or_path outputs/pii-textx-pt/best \
    infer \
    --text "Encaminho o processo SEI 001.002.003/2024." \
    --span_conf_threshold 0.85
```

Output esperado:
```json
{
  "text": "Encaminho o processo SEI 001.002.003/2024.",
  "spans": [
    {"type": "ID_PROCESSUAL", "start": 25, "end": 41, "value": "001.002.003/2024", "conf": 0.92}
  ],
  "should_be_public": false
}
```

**3. Inferência com thresholds por tipo de entidade:**
Diferentes sensibilidades para nomes vs. documentos.
```bash
python infer_pii.py \
    --model_name_or_path outputs/pii-textx-pt/best \
    infer \
    --jsonl_in input.jsonl \
    --jsonl_out output.jsonl \
    --span_conf_threshold_by_type '{"NOME_PESSOA": 0.9, "DOC_PESSOAL": 0.5}'
```

*(O arquivo `output.jsonl` conterá uma linha JSON similar ao exemplo 1 para cada entrada)*

**4. Inferência em textos longos com agregação de sobreposição:**
Garante que entidades na fronteira entre chunks sejam detectadas corretamente.
```bash
python infer_pii.py \
    --model_name_or_path outputs/pii-textx-pt/best \
    infer \
    --jsonl_in docs_longos.jsonl \
    --jsonl_out resultados.jsonl \
    --stride 128 \
    --aggregate_overlaps mean_logits
```

**5. Filtragem de ruído (tamanho mínimo de tokens):**
Remove spans muito curtos que costumam ser erros de fragmentação.
```bash
python infer_pii.py \
    --model_name_or_path outputs/pii-textx-pt/best \
    infer \
    --text "Ref: OF. 123/2024" \
    --min_span_tokens 2 \
    --min_span_tokens_by_type '{"EMAIL": 3}'
```

**6. Forçar execução em CPU:**
Caso não haja GPU disponível para inferência.
```bash
python infer_pii.py \
    --model_name_or_path outputs/pii-textx-pt/best \
    infer \
    --text "Texto de teste" \
    --device cpu
```

### Exemplos de Avaliação

**1. Avaliação padrão com relatório detalhado:**
```bash
python infer_pii.py \
    --model_name_or_path outputs/pii-textx-pt/best \
    eval \
    --dataset_path data/esic-ner/test.jsonl \
    --report_path outputs/eval_report_v1.md
```

Output esperado (stdout):
```json
{
  "pii_token_level": {"precision": 0.96, "recall": 0.94, "f1": 0.95, ...},
  "pii_binary": {"precision": 0.98, "recall": 0.97, "f1": 0.97, "accuracy": 0.97, ...},
  "report_path": "outputs/eval_report_v1.md"
}
```

**2. Avaliação rápida (apenas primeiras N linhas):**
```bash
python infer_pii.py \
    --model_name_or_path outputs/pii-textx-pt/best \
    eval \
    --dataset_path data/esic-ner/test.jsonl \
    --max_rows 100
```

### Argumentos principais (Subcomando `infer`):
- `--text`: Texto para análise direta via linha de comando.
- `--text_file`: Caminho para um arquivo `.txt` contendo o texto para análise.
- `--jsonl_in`: Arquivo JSONL de entrada (campo `text`).
- `--jsonl_out`: Arquivo JSONL de saída com os spans detectados.
- `--text_column`: Nome do campo de texto no JSONL de entrada (default: `text`).
- `--stride`: Sobreposição (overlap) de tokens entre chunks (default: 64).
- `--boundary_backoff`: Janela de segurança para quebra de chunks (default: 32).
- `--decode {argmax, bio_viterbi}`: Algoritmo de decoding (default: `bio_viterbi`). O `bio_viterbi` aplica restrições BIO para evitar sequências inválidas (ex: `I-X` vindo após `O`).
- `--aggregate_overlaps {none, mean_logits}`: Estratégia para chunks sobrepostos. `mean_logits` agrega predições antes do decoding.
- `--span_conf_threshold`: Confiança mínima global para manter um span (0-1).
- `--span_conf_threshold_by_type`: JSON com thresholds específicos por tipo (ex: `'{"NOME_PESSOA": 0.8}'`).
- `--span_conf_agg {mean, min}`: Como agregar confianças de tokens em um span (default: `mean`).
- `--min_span_tokens`: Comprimento mínimo em tokens para manter um span.
- `--min_span_tokens_by_type`: JSON com comprimentos mínimos específicos por tipo.
- `--no_resolve_overlaps`: Desativa a resolução de conflitos de tipos diferentes em spans sobrepostos (default: ligado, prefere span de maior confiança).
- `--device {auto, cpu, cuda}`: Hardware para execução.
- `--batch_size`: Tamanho do lote para processamento (default: 8).

### Argumentos principais (Subcomando `eval`):
- `--dataset_path`: Dataset com labels reais (`text`, `entities`).
- `--text_column` / `--entities_column`: Nomes das colunas no dataset de teste.
- `--report_path`: Caminho para o relatório Markdown.
- `--decode {argmax, bio_viterbi}`: Decoding a usar na avaliação.
- `--max_rows`: Limita o número de amostras avaliadas.
- `--stride` / `--boundary_backoff`: Configurações de chunking para avaliação.
- `--batch_size`: Tamanho do lote durante a avaliação (default: 16).

### Melhorias de Pós-processamento (ONNX-friendly)

O projeto implementa técnicas avançadas de inferência que rodam puramente como pós-processamento (Python/NumPy), o que significa que o modelo base permanece compatível com exportação para **ONNX/TensorRT** sem alterações no grafo:

1.  **Viterbi Constrained Decoding**: Ao usar `--decode bio_viterbi`, o script utiliza o algoritmo de Viterbi com uma matriz de transição que proíbe sequências impossíveis, melhorando a consistência dos spans.
2.  **Logit Aggregation**: Com `--aggregate_overlaps mean_logits`, o modelo calcula a média das probabilidades de cada token que aparece em múltiplos chunks sobrepostos antes de tomar a decisão final, reduzindo ruído em fronteiras de chunk.
3.  **Filtragem por Confiança e Tamanho**: Permite descartar predições "fracas" ou muito curtas, ajudando a equilibrar Precision e Recall sem necessidade de retreinar o encoder.

---

## 6. Estrutura do Projeto

- `infer_pii.py`: Script principal de inferência e avaliação (raiz).
- `ner_labels.py`: Taxonomia canônica de rótulos (entidades + subset PII).
- `requirements.txt`: Lista de dependências Python.
- `data/`:
    - `esic-ner/`: Dataset original (clonado do Hugging Face).
    - `generated/`: Arquivos temporários e resultados de processamento.
- `data_preprocessing/`:
    - `chunking.py`: Módulo compartilhado de Smart Chunking (word/sentence aware).
    - `pii_value_review.py`: Workflow de extração, curadoria e aplicação de revisão de valores PII.
    - `build_finetune_jsonl.py`: Prepara conjuntos de dados de fine-tuning anotados para treino.
- `training/`:
    - `finetune_pii_token_classification.py`: Script para Fine-tuning de NER (Token Classification).
- `inference/`:
    - `decoding.py`: Algoritmos de decoding (argmax, Viterbi).
    - `eval_report.py`: Geração de relatórios de avaliação.
    - `spans.py`: Utilitários para manipulação de spans.
- `utils/`:
    - `cuda_env.py`: Utilitários para configuração de ambiente CUDA.
- `outputs/`: Diretório de saída para modelos treinados, checkpoints e relatórios (ignorado pelo git).
- `docs/`: Documentação de requisitos e especificações técnicas.

---

## 7. Notas Técnicas

- **Smart Chunking**: A estratégia de fragmentação de texto é "PII-agnostic" (não precisa saber onde estão os dados antes de quebrar o texto), mas é semanticamente consciente, preferindo quebras em fins de frase e evitando cortes no meio de palavras. Isso melhora a precisão do modelo ao preservar o contexto local.
- **Fail-fast**: O script de fine-tuning verifica a versão da biblioteca `accelerate` antes de iniciar o processamento pesado, garantindo que o ambiente está configurado corretamente.

## Refs.:

https://www.ouvidoria.df.gov.br/distrito-federal-registra-crescimento-de-107-nos-pedidos-de-acesso-a-informacao/

https://www.dodf.df.gov.br/dodf/jornal/visualizar-pdf?pasta=2025|11_Novembro|DODF%20222%2025-11-2025|&arquivo=DODF%20222%2025-11-2025%20INTEGRA.pdf  

https://about.roblox.com/newsroom/2025/11/open-sourcing-roblox-pii-classifier-ai-pii-detection-chat

https://github.com/Geotrend-research/smaller-transformers

https://www.protecto.ai/blog/best-ner-models-for-pii-identification/
