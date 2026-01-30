# Especificação Técnica: Desafio Participa DF
## Categoria: Acesso à Informação

### 1. Visão Geral do Problema
O objetivo é desenvolver uma solução tecnológica (modelo de IA/Algoritmo) capaz de identificar automaticamente pedidos de acesso à informação que contenham **dados pessoais** e que, portanto, deveriam ser classificados como restritos, mas estão marcados como públicos.

* **Definição de Dados Pessoais:** Informações que permitam identificação direta ou indireta de pessoa natural (ex: Nome, CPF, RG, telefone, e-mail).
* **Dataset de Treino:** Conjunto amostral contendo dados reais (públicos) e dados sintéticos (simulações de dados pessoais).
* **Validação:** A solução será testada contra um subconjunto de controle (oculto).

---

### 2. Requisitos de Entrega (Repositório)
A submissão deve ser feita via link de repositório (GitHub ou GitLab).

* **Visibilidade:** O repositório deve ser **Público** (acesso sem senha/autenticação).
* **Prazo de Submissão:** Entre **12/01/2026 e 30/01/2026**.
* **Congelamento:** Alterações ou commits feitos após o envio do formulário de inscrição desclassificam o projeto.

---

### 3. Critérios de Avaliação (Nota Final)
A nota final é a soma de **P1 (Desempenho Técnico)** + **P2 (Qualidade da Documentação)**.

#### 3.1. P1: Técnicas de Desempenho (Métrica do Modelo)
O modelo será avaliado pela **Pontuação F1** (média harmônica entre Precisão e Sensibilidade/Recall).

* **Fórmula:**
    $$P1 = 2 \times \frac{(Precisão \times Sensibilidade)}{(Precisão + Sensibilidade)}$$

    * **Precisão:** $VP / (VP + FP)$ (Dos classificados como tendo dados pessoais, quantos realmente tinham?).
    * **Sensibilidade (Recall):** $VP / (VP + FN)$ (De todos que tinham dados pessoais, quantos o modelo achou?).

> VP (Verdadeiros Positivos): São os acertos positivos.
Significado: O modelo disse que o documento continha dados pessoais e ele realmente continha.


> FP (Falsos Positivos): São os "alarmes falsos".
Significado: O modelo disse que o documento continha dados pessoais, mas na verdade ele não continha (era público mesmo).

> FN (Falsos Negativos): São os erros de omissão.
Significado: O documento continha dados pessoais (era sigiloso), mas o modelo não identificou e deixou passar como se fosse público.

* **Critérios de Desempate:**
    1. Menor número de Falsos Negativos.
    2. Menor número de Falsos Positivos.
    3. Maior nota P1.

#### 3.2. P2: Documentação e Qualidade (10 Pontos)
A documentação deve estar clara no `README.md` e na estrutura do projeto. A pontuação é "tudo ou nada" por item (não há nota parcial).

| Critério | Requisito Detalhado | Pontos |
| :--- | :--- | :--- |
| **1. Instalação** | **a)** Listar pré-requisitos (ex: versão da linguagem, softwares necessários). | 1.0 |
| | **b)** Arquivo de gerenciamento de dependências automatizado (ex: `requirements.txt`, `package.json`). | 2.0 |
| | **c)** Comandos exatos para criar e configurar o ambiente (virtualenv, docker build, etc). | 1.0 |
| **2. Execução** | **a)** Comando(s) exato(s) para rodar o modelo/script, incluindo argumentos. | 2.0 |
| | **b)** Descrição clara do formato de dados de entrada (Input) e saída gerada (Output). | 1.0 |
| **3. Clareza** | **a)** `README.md` descrevendo o objetivo da solução e a função dos arquivos principais. | 1.0 |
| | **b)** Código-fonte com comentários explicando a lógica em trechos complexos. | 1.0 |
| | **c)** Estrutura de arquivos lógica e organizada (separação de dados, scripts, modelos). | 1.0 |
| **TOTAL** | | **10.0** |

---

### 4. Regras Gerais e Uso de IA
* **Linguagem:** Livre (desde que documentados os pré-requisitos)[cite: 92].
* **Uso de IA:** Permitido, desde que documentado no `README.md` (indicar modelos, bibliotecas e fontes utilizadas).
* **Infraestrutura:** A execução deve ocorrer nos equipamentos do participante; a CGDF não fornece infraestrutura.
* **Direitos Autorais:** O participante cede a propriedade intelectual da solução à CGDF.
