# Especificação Técnica: Desafio Participa DF (Edital nº 10/2025 - CGDF)
## Categoria: Acesso à Informação

### 1. Visão Geral do Problema (Item 2.2-I do Edital)
O objetivo é desenvolver uma solução tecnológica (modelo de IA/Algoritmo) capaz de identificar automaticamente pedidos de acesso à informação que contenham **dados pessoais** e que, portanto, deveriam ser classificados como restritos (não públicos), mas estão marcados como públicos.

* **Definição de Dados Pessoais:** Informações que permitam a identificação direta ou indireta de uma pessoa natural, tais como nome, CPF, RG, telefone e endereço de e-mail.
* **Dataset de Treino:** Conjunto amostral contendo dados reais (públicos) e dados sintéticos (simulações de dados pessoais) disponibilizados pela CGDF (Item 8.1.1).
* **Validação:** A solução será testada contra um subconjunto de controle oculto pela CGDF (Item 8.1.3).

---

### 2. Requisitos de Entrega e Submissão (Item 6 do Edital)
A submissão deve ser feita via link de repositório (GitHub ou GitLab).

* **Visibilidade:** O repositório deve ser **Público** (permitindo acesso para visualização e clonagem sem necessidade de autenticação).
* **Prazo de Submissão:** Entre **12/01/2026 e 30/01/2026**.
* **Congelamento:** Alterações, commits ou uploads realizados no repositório após o preenchimento do formulário de inscrição e submissão não serão considerados (Item 6.6.1).

---

### 3. Critérios de Avaliação (Item 8.1.5 do Edital)
A nota do projeto é a soma de **P1 (Técnicas de Desempenho)** + **P2 (Documentação da Solução)**.

#### 3.1. Pontuação P1: Técnicas de Desempenho (Métrica do Modelo)
O modelo será avaliado pela métrica de desempenho que combina **Precisão** e **Sensibilidade/Recall**.

* **Fórmula P1 (Item 8.1.5.2.3):**
    $$P1 = 2 \times \frac{(Precisão \times Sensibilidade)}{(Precisão + Sensibilidade)}$$

    * **Precisão (Item 8.1.5.2.1):** $VP / (VP + FP)$ (De todos os pedidos que o modelo classificou como "contendo dados pessoais", quantos realmente continham?).
    * **Sensibilidade/Recall (Item 8.1.5.2.2):** $VP / (VP + FN)$ (De todos os pedidos que realmente continham dados pessoais, quantos o modelo conseguiu identificar?).

> **Nota sobre Escopo PII:** Conforme as regras da categoria, entidades como `ORG_JURIDICA` e `DOC_EMPRESA` são tratadas como informações de natureza pública e **não são contabilizadas** como PII para fins de cálculo de Falsos Negativos ou Positivos na métrica P1.

* **Critérios de Desempate (Item 8.2.5.1, aplicado à categoria):**
    1. Menor número de Falsos Negativos (FN).
    2. Menor número de Falsos Positivos (FP).
    3. Maior nota P1.

#### 3.2. Pontuação P2: Documentação e Qualidade (Item 8.2.4.3 - 10 Pontos)
A documentação deve estar clara no `README.md` e na estrutura do projeto. A pontuação é proporcional ao nível de atendimento.

| Critério | Requisito Detalhado | Pontos |
| :--- | :--- | :--- |
| **1. Instalação** | Listar pré-requisitos, arquivo de dependências (requirements.txt) e comandos de setup. | 4.0 |
| **2. Execução** | Comandos exatos para rodar o modelo e descrição clara de Input/Output. | 3.0 |
| **3. Organização** | README com objetivo, clareza do código (comentários) e estrutura lógica de arquivos. | 3.0 |
| **TOTAL** | | **10.0** |

---

### 4. Regras Gerais e Uso de IA (Item 13.9 do Edital)
* **Linguagem:** Livre, desde que documentados os pré-requisitos.
* **Uso de IA:** Permitido, desde que devidamente documentado no `README.md`, com indicação clara dos modelos, bibliotecas e fontes utilizadas.
* **Propriedade Intelectual:** Os participantes cedem e transferem totalmente à CGDF toda a propriedade intelectual decorrente das soluções desenvolvidas (Item 10.3).
