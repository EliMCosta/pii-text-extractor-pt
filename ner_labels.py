from __future__ import annotations

"""
Canonical NER (entity) label taxonomy for this repository.

Important:
- Labels MUST be stable strings (used in datasets and model label space).
- This taxonomy includes both:
  - PII entities (person natural identifiers)
  - Non-PII entities that are still important for learning (e.g. ORG_JURIDICA, DOC_EMPRESA)

PII vs non-PII:
- `ORG_JURIDICA` and `DOC_EMPRESA` are treated as entity labels but NOT as PII for
  publication decisions and PII-only evaluation metrics.
"""

import re
from typing import Final


# Common identifier regexes (kept permissive but consistent).
_RE_CNPJ: Final[re.Pattern[str]] = re.compile(r"^\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}$")

# Acronyms / tokens that frequently appear in requests but are NOT organizations.
# Keep this list intentionally short and high-confidence to avoid over-rejection.
_ORG_ACRONYM_DENYLIST: Final[set[str]] = {
    "gps",
    "pdf",
    "csv",
    "cid",
    "ppp",
    "ctps",
    "ipva",
    "gta",
    "sei",  # system acronym, not an organization in these texts
    "sisreg",  # system acronym; often appears as system/protocol, not the org itself
}

# Very common surnames that LLMs sometimes output in ALL CAPS and mislabel as ORG.
_ORG_COMMON_SURNAMES_DENYLIST: Final[set[str]] = {
    "silva",
    "souza",
    "sousa",
    "oliveira",
    "santos",
    "pereira",
    "ferreira",
    "rodrigues",
    "albuquerque",
    "alencar",
    "viana",
    "meireles",
    "mendonça",
    "mendonca",
    "camargo",
    "holanda",
    "arruda",
    "castro",
    "medeiros",
    "magalhães",
    "magalhaes",
}

# Words commonly used as emphasis/boilerplate that should never be labeled ORG.
_ORG_GENERIC_TOKEN_DENYLIST: Final[set[str]] = {
    "urgente",
    "obrigado",
    "obrigada",
    "desde",
    "muito",
    "preciso",
    "quero",
    "saber",
    "solicito",
    "detalhadas",
    "reforma",
    "portal",
    "transporte",
    "animais",
    "profissionais",
    "pagamento",
    "deste",
    "regional",
}

# ---------------------------------------------------------------------------
# Prefixes that should be stripped from `entities.value` to keep training data
# consistent. These are label prefixes like "SEI nº", "CPF:", "Matrícula", etc.
# that don't belong in the extracted value itself.
# ---------------------------------------------------------------------------
_STRIP_PREFIX_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "ID_PROCESSUAL": [
        re.compile(r"^processo(?:\s+sei)?(?:\s+n[º°.]*)?\s*", re.IGNORECASE),
        re.compile(r"^sei(?:\s+n[º°.]*)?\s*", re.IGNORECASE),
        re.compile(r"^protocolo(?:\s+n[º°.]*)?\s*", re.IGNORECASE),
        re.compile(r"^edital(?:\s+n[º°.]*)?\s*", re.IGNORECASE),
        re.compile(r"^portaria(?:\s+n[º°.]*)?\s*", re.IGNORECASE),
        re.compile(r"^boletim(?:\s+de\s+ocorrência)?(?:\s+n[º°.]*)?\s*", re.IGNORECASE),
        re.compile(r"^auto(?:\s+de\s+infração)?(?:\s+n[º°.]*)?\s*", re.IGNORECASE),
        re.compile(r"^matrícula(?:\s+n[º°.]*)?\s*", re.IGNORECASE),
        re.compile(r"^inscrição(?:\s+imobiliária)?(?:\s+n[º°.]*)?\s*", re.IGNORECASE),
        re.compile(r"^n[º°.]+\s*", re.IGNORECASE),
    ],
    "DOC_PROFISSIONAL": [
        re.compile(r"^matrícula(?:\s+n[º°.]*)?\s*", re.IGNORECASE),
        re.compile(r"^mat\.?\s*", re.IGNORECASE),
        re.compile(r"^rgp(?:\s+n[º°.]*)?\s*", re.IGNORECASE),
        re.compile(r"^registro(?:\s+profissional)?(?:\s+n[º°.]*)?\s*", re.IGNORECASE),
        re.compile(r"^inscrição(?:\s+n[º°.]*)?\s*", re.IGNORECASE),
    ],
    "DADO_FINANCEIRO": [
        re.compile(r"^agência\s*", re.IGNORECASE),
        re.compile(r"^agencia\s*", re.IGNORECASE),
        re.compile(r"^ag\.\s*", re.IGNORECASE),
        re.compile(r"^conta(?:\s+corrente|\s+poupança)?(?:\s+n[º°.]*)?\s*", re.IGNORECASE),
        re.compile(r"^chave(?:\s+pix)?(?::)?\s*", re.IGNORECASE),
        re.compile(r"^pix(?::)?\s*", re.IGNORECASE),
        re.compile(r"^banco(?:\s+do\s+brasil|\s+itaú)?\s+(?:agência|ag\.?)\s*", re.IGNORECASE),
    ],
    "ID_VEICULO": [
        re.compile(r"^placa(?::)?\s*", re.IGNORECASE),
        re.compile(r"^renavam(?::)?\s*", re.IGNORECASE),
        re.compile(r"^chassi(?::)?\s*", re.IGNORECASE),
    ],
    "DOC_PESSOAL": [
        re.compile(r"^cpf(?::)?\s*", re.IGNORECASE),
        re.compile(r"^rg(?::)?\s*", re.IGNORECASE),
        re.compile(r"^cnh(?::)?\s*", re.IGNORECASE),
        re.compile(r"^título(?:\s+de\s+eleitor)?(?::)?\s*", re.IGNORECASE),
    ],
    "DOC_EMPRESA": [
        re.compile(r"^cnpj(?::)?\s*", re.IGNORECASE),
        re.compile(r"^inscri[cç][aã]o\s+estadual(?::)?\s*", re.IGNORECASE),
        re.compile(r"^inscri[cç][aã]o\s+municipal(?::)?\s*", re.IGNORECASE),
        re.compile(r"^i[.\s-]*e[.\s-]*(?::)?\s*", re.IGNORECASE),  # IE / I.E.
        re.compile(r"^i[.\s-]*m[.\s-]*(?::)?\s*", re.IGNORECASE),  # IM / I.M.
        re.compile(r"^isento(?::)?\s*", re.IGNORECASE),
    ],
    "CONTATO": [
        re.compile(r"^tel(?:efone)?(?::)?\s*", re.IGNORECASE),
        re.compile(r"^cel(?:ular)?(?::)?\s*", re.IGNORECASE),
        re.compile(r"^e-?mail(?::)?\s*", re.IGNORECASE),
        re.compile(r"^whatsapp(?::)?\s*", re.IGNORECASE),
    ],
    "ORG_JURIDICA": [
        re.compile(r"^raz[aã]o\s+social(?::)?\s*", re.IGNORECASE),
        re.compile(r"^empresa(?::)?\s*", re.IGNORECASE),
        re.compile(r"^órg[aã]o(?::)?\s*", re.IGNORECASE),
        re.compile(r"^org[aã]o(?::)?\s*", re.IGNORECASE),
    ],
}


def sanitize_entity_value(entity_type: str, value: str, text: str) -> str:
    """
    Strip common label prefixes from an entity value while ensuring the cleaned
    value still exists literally in the text.

    This is applied both at generation time and at dataset build time to ensure
    consistency. The function is idempotent.

    Args:
        entity_type: The entity type (e.g. "ID_PROCESSUAL").
        value: The raw entity value that may contain prefixes.
        text: The original text where the value must appear.

    Returns:
        The sanitized value (stripped of prefixes) if it exists in text,
        otherwise the original value unchanged.
    """
    if entity_type not in _STRIP_PREFIX_PATTERNS:
        return value

    new_value = value
    changed = True
    while changed:
        changed = False
        for pattern in _STRIP_PREFIX_PATTERNS[entity_type]:
            temp = pattern.sub("", new_value)
            if temp != new_value:
                new_value = temp
                changed = True
        # Strip trailing punctuation/whitespace left behind
        temp = new_value.strip(" :.-,")
        if temp != new_value:
            new_value = temp
            changed = True

    # Only return the cleaned value if it exists in text
    if new_value and new_value != value and new_value in text:
        return new_value
    return value


# NOTE:
# - Labels MUST be stable strings (used in datasets and model label space).
# - Descriptions are intentionally detailed to drive consistent annotation by LLMs.
ENTITY_LABEL_SPECS: Final[dict[str, str]] = {
    "NOME_PESSOA": (
        "Identificadores de pessoa física por nome. Inclui: nome completo, nome social, "
        "primeiro+sobrenome, nomes de parentes quando identificam alguém. "
        "Exclui: nomes de órgãos/empresas (use ORG_JURIDICA), cargos isolados sem nome "
        "(ex.: 'o diretor'), menções genéricas ('o servidor', 'o cidadão')."
    ),
    "DOC_PESSOAL": (
        "Identificadores de documentos/cadastros pessoais de pessoa física (o identificador em si). "
        "Inclui: números de CPF, RG, CNH, PIS/PASEP/NIT, Título de Eleitor, certidões, passaporte. "
        "Regra: prefira rotular o NÚMERO/ID; não rotule apenas a menção do tipo de documento "
        "('CPF', 'RG', 'CNH', 'título de eleitor') quando NÃO houver identificador associado. "
        "Exclui: CNPJ/IE/IM (use DOC_EMPRESA), números de processo/protocolo (use ID_PROCESSUAL), "
        "identificadores profissionais (use DOC_PROFISSIONAL)."
    ),
    "DATA_NASC": (
        "Apenas datas de nascimento (ou expressões inequívocas de nascimento). "
        "Inclui formatos como 'dd/mm/aaaa', 'nascido em ...', 'data de nascimento: ...'. "
        "Exclui: outras datas (protocolo, evento, despacho, publicação, vigência, etc.)."
    ),
    "CONTATO": (
        "Canais de contato pessoal. Inclui: e-mail, telefone/celular, WhatsApp. "
        "Exclui: endereços físicos (use ENDERECO), perfis/URLs genéricos quando não identificam "
        "uma pessoa diretamente (evite rotular URL por padrão)."
    ),
    "ENDERECO": (
        "Endereço físico identificável. Inclui: logradouro + número, complemento, bairro, "
        "cidade/UF quando acompanhado de logradouro, CEP. "
        "Exclui: apenas cidade/UF soltos sem logradouro/CEP (não rotule), referências vagas "
        "('na minha rua', 'perto do centro')."
    ),
    "DOC_PROFISSIONAL": (
        "Identificadores profissionais/funcionais. Inclui: matrícula funcional (SIAPE/GDF etc.), "
        "OAB, CRM, CREA e equivalentes. "
        "Exclui: CPF/RG (use DOC_PESSOAL), números de processo/protocolo (use ID_PROCESSUAL)."
    ),
    "ID_PROCESSUAL": (
        "Identificadores burocráticos de processos/protocolos. Inclui: número de processo SEI, "
        "protocolos de atendimento, números de boletim de ocorrência, IDs de processo judicial/administrativo. "
        "Exclui: CPF/RG (use DOC_PESSOAL), CNPJ/IE/IM (use DOC_EMPRESA), placas/renavam (use ID_VEICULO)."
    ),
    "ID_VEICULO": (
        "Identificadores de veículo. Inclui: placa, RENAVAM, chassi/VIN. "
        "Exclui: números de processo/protocolo (use ID_PROCESSUAL)."
    ),
    "ORG_JURIDICA": (
        "Entidades jurídicas e organizações (para desambiguação). Inclui: razão social, "
        "nome de empresa e nome de órgão público/entidade (inclui unidades organizacionais como "
        "'Secretaria', 'Ministério', 'Procuradoria', 'Delegacia', etc.). "
        "Exclui: nomes de pessoas (use NOME_PESSOA) e cargos/títulos de pessoas sem nome de órgão, nomes de unidades de federação, plataformas eletrônicas, etc. "
        "(ex.: 'SECRETÁRIO DE ESTADO ...', 'o diretor', 'procurador')."
    ),
    "DADO_SAUDE": (
        "Dados sensíveis de saúde (LGPD Art. 5º). Inclui: doenças/condições, diagnósticos, "
        "CID, tratamentos, medicamentos, internações. "
        "Exclui: menções genéricas a 'saúde pública' ou termos administrativos sem revelar condição de alguém, termos genéricos psicológicos dentre outros."
    ),
    "DADO_FINANCEIRO": (
        "Identificadores financeiros. Inclui: número de cartão, número de conta/agência, chave Pix "
        "(CPF/e-mail/telefone/chave aleatória), dados bancários identificáveis. "
        "Exclui: valores monetários soltos (ex.: 'R$ 50,00') e informações financeiras genéricas sem identificador."
    ),
    "DOC_EMPRESA": (
        "Identificadores/cadastros de pessoa jurídica/empresa (o identificador em si). "
        "Inclui: CNPJ, Inscrição Estadual (IE), Inscrição Municipal (IM). "
        "Regra: prefira rotular o NÚMERO/ID; não rotule apenas 'CNPJ'/'IE'/'IM' sem o identificador associado. "
        "Exclui: razão social/nome do órgão (use ORG_JURIDICA)."
    ),
    "QUASI_IDENTIFICADOR": (
        "Combinação de dados que permite identificação INDIRETA de pessoa natural (LGPD Art. 5º, I). "
        "Inclui: função/cargo + local + horário específico quando a combinação aponta para indivíduo único "
        "(ex.: 'motorista da linha 805.6 às 07h15', 'atendente do guichê 3 no dia 15/01', "
        "'professora da turma 2B às 14h', 'segurança do turno noturno do prédio X'). "
        "Rotule o TRECHO COMPLETO que forma o identificador indireto (função + contexto temporal/espacial). "
        "Exclui: referências genéricas sem especificidade suficiente para identificar "
        "(ex.: 'um motorista de ônibus', 'o atendente', 'funcionários do setor'), "
        "e casos onde já há identificador direto no mesmo trecho (nome, matrícula) — prefira o direto."
    ),
}

# Guidance to reduce extreme variability in `entities.value` extraction.
# IMPORTANT: `value` must ALWAYS be a literal substring of `text` (no normalization),
# so these rules focus on what span to select (minimal, well-formed, type-consistent).
ENTITY_VALUE_EXTRACTION_RULES: Final[dict[str, str]] = {
    "NOME_PESSOA": (
        "Extraia o menor trecho que contenha o nome (preferir nome+sobrenome). "
        "Não inclua saudações/títulos (Sr., Sra., Dr., Dra.) nem cargos/parentescos sem nome."
    ),
    "DOC_PESSOAL": (
        "Extraia SOMENTE o identificador (número/código) do documento. "
        "CORRETO: '129.180.122-6', '110.100.179-87', 'MG-12.345.678', '12345678901'. "
        "ERRADO: apenas o tipo do documento ('CPF', 'RG', 'CNH', 'título de eleitor') sem número/ID. "
        "Não inclua prefixos como 'CPF:'/'RG:' nem adjetivos como 'digital', 'novo', 'segunda via'."
    ),
    "DATA_NASC": (
        "Preferir o trecho mínimo que representa a data, idealmente em formato numérico 'dd/mm/aaaa' "
        "(ou 'dd-mm-aaaa'). Não inclua frases inteiras ('data de nascimento: ...')."
    ),
    "CONTATO": (
        "Extraia o e-mail/telefone completo. Para e-mail, não inclua pontuação final. "
        "Para telefone, inclua DDD quando houver. Não inclua rótulos como 'tel:'/'e-mail:'."
    ),
    "ENDERECO": (
        "Extraia o menor trecho contínuo que identifique o endereço (logradouro/quadra/setor + número/lote). "
        "Evite incluir observações não-endereço ('em frente', 'próximo', etc.). "
        "Evite spans que atravessem quebras de linha."
    ),
    "DOC_PROFISSIONAL": (
        "Extraia o identificador profissional completo, INCLUINDO a sigla do conselho quando presente. "
        "CORRETO: 'CRM-DF 12345', 'OAB/SP 98765', 'CREA-GO 54321/D', 'SIAPE 1234567'. "
        "ERRADO: 'Matrícula 12345' (remova 'Matrícula'), '12345' sozinho sem sigla quando ela existe no texto. "
        "O value deve conter dígitos. Não inclua prefixos genéricos como 'matrícula', 'registro', 'inscrição'."
    ),
    "ID_PROCESSUAL": (
        "Extraia SOMENTE o número/código do processo, SEM prefixos descritivos. "
        "ERRADO: 'SEI nº 123456/2023', 'Processo 123456', 'Protocolo SEI 123456'. "
        "CORRETO: '123456/2023', '123456'. "
        "Não inclua palavras como 'SEI', 'Processo', 'Protocolo', 'nº', 'Matrícula', 'Edital' no value."
    ),
    "ID_VEICULO": (
        "Extraia SOMENTE a placa/RENAVAM/chassi, SEM prefixos descritivos. "
        "ERRADO: 'placa ABC-1234', 'Placa: ABC1D23'. "
        "CORRETO: 'ABC-1234', 'ABC1D23'. "
        "Não inclua palavras como 'placa', 'Renavam', 'chassi' no value."
    ),
    "ORG_JURIDICA": (
        "Extraia o nome da organização/órgão (ou sigla). "
        "Não inclua identificadores numéricos (CNPJ/IE/IM) no value — isso é DOC_EMPRESA. "
        "Não rotule cargos/títulos de pessoas (ex.: 'SECRETÁRIO...', 'diretor', 'procurador'); "
        "extraia somente a entidade (ex.: 'Secretaria de Estado de ...', 'DF LEGAL', 'CGU'). "
        "Não rotule palavras de ênfase/boilerplate (ex.: 'URGENTE', 'OBRIGADO') nem siglas de formato/tecnologia "
        "(ex.: 'GPS', 'PDF', 'CSV') nem tributos/benefícios (ex.: 'IPVA')."
    ),
    "DADO_SAUDE": (
        "Extraia o termo específico (doença/condição, CID, tratamento/medicamento) no menor trecho possível. "
        "Não rotule termos administrativos genéricos sozinhos (ex.: 'laudo', 'atestado') sem especificação."
    ),
    "DADO_FINANCEIRO": (
        "Extraia SOMENTE o número/código financeiro, SEM prefixos descritivos. "
        "ERRADO: 'Agência 0850', 'conta corrente 12345-6', 'chave Pix 11999887766', 'Pix: email@x.com'. "
        "CORRETO: '0850', '12345-6', '11999887766', 'email@x.com'. "
        "Não inclua palavras como 'Agência', 'Conta', 'Pix', 'chave' no value. "
        "Não rotule valores monetários soltos ('R$ 50,00')."
    ),
    "DOC_EMPRESA": (
        "Extraia SOMENTE o identificador da empresa (CNPJ/IE/IM), SEM prefixos descritivos. "
        "CORRETO: '25.598.301/0001-68', '25698301000168', '110.042.490.114', '123456789'. "
        "ERRADO: 'CNPJ: 25.598.301/0001-68', 'Inscrição Estadual 110.042.490.114', 'IE 123456'. "
        "Não inclua palavras como 'CNPJ', 'Inscrição Estadual', 'Inscrição Municipal', 'IE', 'IM' no value."
    ),
    "QUASI_IDENTIFICADOR": (
        "Extraia o TRECHO COMPLETO que forma o identificador indireto, incluindo função + contexto. "
        "CORRETO: 'motorista da linha 805.6, no horário das 07h15', 'atendente do guichê 3 no dia 15/01/2024', "
        "'professora da turma 2B do turno vespertino'. "
        "ERRADO: apenas 'linha 805.6' ou apenas '07h15' isolados (sem a função/papel). "
        "O value deve conter: (1) referência a função/cargo/papel E (2) contexto temporal ou espacial específico. "
        "Evite spans muito longos; pare antes de orações subordinadas ou explicações adicionais."
    ),
}


ALLOWED_ENTITY_TYPES: Final[tuple[str, ...]] = tuple(ENTITY_LABEL_SPECS.keys())

# Subset used for "has PII" decisions and PII-only evaluation (non-PII types excluded).
NON_PII_TYPES: Final[tuple[str, ...]] = ("ORG_JURIDICA", "DOC_EMPRESA")
PII_TYPES: Final[tuple[str, ...]] = tuple(t for t in ALLOWED_ENTITY_TYPES if t not in NON_PII_TYPES)

# Priority for resolving conflicts when the same literal value has multiple entity types.
# Lower number = higher priority. Used when training expands a value to ALL its literal occurrences,
# so we must pick ONE canonical type (e.g., CPF as DOC_PESSOAL even if also used as Pix key).
ENTITY_TYPE_PRIORITY: Final[dict[str, int]] = {
    "DOC_PESSOAL": 0,
    "DOC_PROFISSIONAL": 1,
    "DOC_EMPRESA": 2,
    "CONTATO": 3,
    "DATA_NASC": 4,
    "ENDERECO": 5,
    "NOME_PESSOA": 6,
    "DADO_FINANCEIRO": 7,
    "ID_PROCESSUAL": 8,
    "ID_VEICULO": 9,
    "DADO_SAUDE": 10,
    "ORG_JURIDICA": 11,
    "QUASI_IDENTIFICADOR": 12,
}

# Validate that all entity types have a priority defined.
_missing_priority = set(ALLOWED_ENTITY_TYPES) - set(ENTITY_TYPE_PRIORITY.keys())
if _missing_priority:
    raise RuntimeError(
        f"ENTITY_TYPE_PRIORITY is missing entries for: {sorted(_missing_priority)}. "
        "Update ENTITY_TYPE_PRIORITY when adding new entity types."
    )


def format_entity_label_guidelines() -> str:
    """
    Human-readable label guide for LLM prompting.
    """
    lines: list[str] = []
    lines.append("Taxonomia de labels (use exatamente UM destes rótulos; não invente novos):")
    for label, desc in ENTITY_LABEL_SPECS.items():
        lines.append(f"- {label}: {desc}")
    lines.append(
        "Regra de consistência: escolha o rótulo mais específico possível. "
        "Ex.: e-mail/telefone -> CONTATO; CPF/RG/CNH -> DOC_PESSOAL; CNPJ/IE/IM -> DOC_EMPRESA; "
        "órgão/empresa (nome/sigla) -> ORG_JURIDICA; "
        "processo/protocolo -> ID_PROCESSUAL."
    )
    return "\n".join(lines)


def format_entity_value_extraction_rules() -> str:
    """
    Human-readable rules to standardize what `entities.value` should contain.
    Intended for LLM prompting and for fail-fast validation in dataset generators.
    """
    lines: list[str] = []
    lines.append("Regras de extração para entities.value (reduzir variabilidade):")
    lines.append("- value deve ser substring LITERAL de text (sem normalizar/remover acentos).")
    lines.append("- value não pode ter espaços no começo/fim.")
    lines.append("- value deve ser o MENOR trecho possível que representa a entidade (evite contexto extra).")
    lines.append("- Evite spans com quebra de linha dentro do value.")
    lines.append(
        "- Regra crítica de consistência: dentro de um mesmo texto, o MESMO value não pode aparecer com "
        "dois types diferentes em entities. Se precisar de dois papéis (ex.: telefone para contato e "
        "telefone como chave Pix), use valores distintos no texto."
    )
    lines.append(
        "- REGRA CRÍTICA - SEM PREFIXOS DESCRITIVOS: o value deve conter APENAS o identificador, "
        "NUNCA inclua palavras como 'SEI', 'Processo', 'Protocolo', 'nº', 'Matrícula', 'Agência', "
        "'Conta', 'Pix', 'Placa', 'CPF:', 'RG:' etc. antes do número/código. "
        "Exemplo: se o texto diz 'processo SEI nº 123456/2023', o value deve ser '123456/2023', não 'SEI nº 123456/2023'."
    )
    for label, rule in ENTITY_VALUE_EXTRACTION_RULES.items():
        lines.append(f"- {label}: {rule}")
    return "\n".join(lines)


def validate_entity_value_format(*, entity_type: str, value: str) -> None:
    """
    Fail-fast validation for `entities.value` to keep generator outputs consistent.

    Notes:
    - This does NOT check `value in text`; callers should ensure literal presence separately.
    - Rules are intentionally conservative (avoid rejecting good real-world cases),
      while blocking the most common sources of extreme variability (prefixes/adjectives/generic terms).
    """
    if entity_type not in ENTITY_LABEL_SPECS:
        raise ValueError(f"unknown entity_type: {entity_type!r}")
    if not isinstance(value, str) or not value:
        raise ValueError("value must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"value must not have leading/trailing whitespace: {value!r}")
    if "\n" in value or "\r" in value:
        raise ValueError(f"value must not contain newlines: {value!r}")
    if len(value) > 220:
        raise ValueError(f"value too long (>{220} chars): {value!r}")

    low = value.casefold()
    digits = sum(ch.isdigit() for ch in value)
    letters = sum(ch.isalpha() for ch in value)

    def _has_any(subs: tuple[str, ...]) -> bool:
        return any(s in low for s in subs)

    if entity_type == "CONTATO":
        # Email or phone; keep it light.
        if "@" in value:
            if any(ch.isspace() for ch in value):
                raise ValueError(f"email value must not contain whitespace: {value!r}")
            if not (1 <= value.count("@") <= 1):
                raise ValueError(f"email value must contain a single '@': {value!r}")
            if "." not in value.split("@", 1)[-1]:
                raise ValueError(f"email value missing domain dot: {value!r}")
            if value.endswith((".", ",", ";", ":", ")", "]")):
                raise ValueError(f"email value must not end with punctuation: {value!r}")
        else:
            if digits < 8:
                raise ValueError(f"phone value must contain >= 8 digits: {value!r}")
        return

    if entity_type == "DATA_NASC":
        # Prefer dd/mm/yyyy or dd-mm-yyyy, but allow 2-digit year to avoid over-rejection.
        if not re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", value):
            raise ValueError(f"DATA_NASC must include a numeric date (dd/mm/aaaa): {value!r}")
        return

    if entity_type in ("DOC_PROFISSIONAL", "DADO_FINANCEIRO", "ID_PROCESSUAL", "ID_VEICULO", "DOC_EMPRESA"):
        # These should generally contain digits; allow small IDs only when there is a strong marker.
        if entity_type == "ID_PROCESSUAL":
            # Reject values that START with verbose prefixes (should have been stripped).
            _bad_prefixes = (
                "processo", "sei ", "sei-", "protocolo", "edital", "portaria",
                "boletim", "auto de", "matrícula", "inscrição", "nº ", "n° ", "n. ",
            )
            if _has_any(_bad_prefixes) and low.startswith(_bad_prefixes):
                raise ValueError(
                    f"ID_PROCESSUAL value should not start with verbose prefixes like 'SEI nº'; "
                    f"extract only the identifier: {value!r}"
                )
            # Relaxed: allow numeric-only IDs (common in many systems) or IDs with structure.
            # Examples: "20248831", "85-SES/DF", "PA-123", "SEI-2024/001"
            has_structure = bool(re.search(r"[\-./]", value)) or letters >= 2
            if not (digits >= 4 or (digits >= 2 and has_structure)):
                raise ValueError(
                    "ID_PROCESSUAL must have at least 4 digits or 2+ digits with structure: "
                    f"{value!r}"
                )
        elif entity_type == "ID_VEICULO":
            plate_old = re.fullmatch(r"[A-Z]{3}-?\d{4}", value.strip())
            # Mercosul plate: ABC1D23 or ABC-1D23 (with optional hyphen)
            plate_mercosul = re.fullmatch(r"[A-Z]{3}-?\d[A-Z]\d{2}", value.strip())
            renavam = digits in (9, 10, 11)
            if not (plate_old or plate_mercosul or renavam):
                raise ValueError(f"ID_VEICULO must look like plate or RENAVAM: {value!r}")
        elif entity_type == "DOC_PROFISSIONAL":
            if digits < 3:
                raise ValueError(f"DOC_PROFISSIONAL must contain >= 3 digits: {value!r}")
        elif entity_type == "DOC_EMPRESA":
            # Company identifiers: CNPJ / IE / IM (digits + separators only).
            if digits < 6:
                raise ValueError(f"DOC_EMPRESA must contain >= 6 digits: {value!r}")
            if any(ch.isspace() for ch in value):
                raise ValueError(f"DOC_EMPRESA must not contain whitespace: {value!r}")
            if letters > 0:
                raise ValueError(f"DOC_EMPRESA must not contain letters: {value!r}")
            # Accept common CNPJ formats (with or without punctuation).
            if _RE_CNPJ.fullmatch(value.strip()):
                return
            # IE/IM formats vary across Brazil; keep permissive but enforce identifier-ish shape.
            if not re.fullmatch(r"[0-9][0-9.\-/]{5,}", value.strip()):
                raise ValueError(
                    "DOC_EMPRESA must look like a numeric company identifier (digits + optional separators): "
                    f"{value!r}"
                )
        else:  # DADO_FINANCEIRO
            # Accept PIX key types or account/card patterns; require an "identifier-ish" shape.
            # Reject values that START with verbose prefixes (should have been stripped).
            _bad_fin_prefixes = (
                "agência", "agencia", "ag.", "conta", "chave pix", "pix:", "pix ",
                "banco do brasil", "banco itaú",
            )
            if _has_any(_bad_fin_prefixes) and low.startswith(_bad_fin_prefixes):
                raise ValueError(
                    f"DADO_FINANCEIRO value should not start with verbose prefixes like 'Agência' or 'Chave Pix'; "
                    f"extract only the identifier: {value!r}"
                )
            if "@" in value:
                # Pix email key
                return
            # Relaxed: allow shorter account/agency numbers (>= 4 digits) or alphanumeric IDs.
            # Examples: "3456" (agência), "44921-X" (conta), "123456-7" (full account)
            has_separator = bool(re.search(r"[\-./]", value))
            if digits < 4 or (digits < 6 and not has_separator and letters == 0):
                raise ValueError(f"DADO_FINANCEIRO must contain >= 4 digits (identifier-like): {value!r}")
        return

    if entity_type == "DOC_PESSOAL":
        # DOC_PESSOAL should identify a person via an identifier, not just mention doc types.
        # Require a minimally "identifier-like" value to keep the task well-posed.
        if digits < 4:
            raise ValueError(
                "DOC_PESSOAL must contain an identifier (>= 4 digits). "
                "Do not label only the document type (e.g. 'CPF', 'RG') without its number: "
                f"{value!r}"
            )
        if _has_any(("digital", "novo", "novos", "segunda via", "2a via", "2ª via")):
            raise ValueError(f"DOC_PESSOAL value must not include adjectives/issuance context: {value!r}")
        if _has_any(("cpf:", "rg:", "cnh:", "pis:", "pasep:", "nit:")):
            raise ValueError(f"DOC_PESSOAL value must not include label prefixes like 'CPF:': {value!r}")
        return

    if entity_type == "ORG_JURIDICA":
        # ORG_JURIDICA is for org names/acronyms only (CNPJ/IE/IM are DOC_EMPRESA).
        low_stripped = value.strip().casefold()
        # Hard block for common non-ORG tokens. This is intentionally conservative.
        if low_stripped in _ORG_GENERIC_TOKEN_DENYLIST:
            raise ValueError(
                "ORG_JURIDICA value looks like emphasis/boilerplate, not an organization name/acronym: "
                f"{value!r}"
            )
        if "cnpj" in low:
            raise ValueError(
                f"ORG_JURIDICA must not include 'CNPJ' prefix; use DOC_EMPRESA with only the number: {value!r}"
            )
        if _RE_CNPJ.fullmatch(value.strip()):
            raise ValueError(
                f"ORG_JURIDICA must be an organization name/acronym, not CNPJ. Use DOC_EMPRESA: {value!r}"
            )
        if digits >= 6 and letters == 0:
            raise ValueError(
                "ORG_JURIDICA must be an organization name/acronym, not a numeric identifier. "
                f"Use DOC_EMPRESA for CNPJ/IE/IM: {value!r}"
            )
        # Avoid job titles / honorifics being mislabeled as organizations.
        # NOTE: we intentionally do NOT ban organizational nouns like "secretaria", "procuradoria", "delegacia".
        if re.search(
            r"\b("
            r"secret[aá]rio|diretor|presidente|procurador|delegado|ju[ií]z|desembargador|"
            r"governador|prefeito|vereador|senador|deputado|ministro|"
            r"sr\.?|sra\.?|dr\.?|dra\.?|ilustr[ií]ssimo|ilustrissima|senhor|senhora"
            r")\b",
            low,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                "ORG_JURIDICA should not be a person job title/honorific. "
                "Extract only the organization name (e.g. 'Secretaria de ...'), not 'Secretário ...': "
                f"{value!r}"
            )
        raw = value.strip()
        tokens = raw.split()
        acronym = raw == raw.upper() and letters >= 2 and len(raw) <= 18 and " " not in raw
        if acronym:
            if low_stripped in _ORG_ACRONYM_DENYLIST:
                raise ValueError(
                    "ORG_JURIDICA acronym is a known non-organization token in this dataset: "
                    f"{value!r}"
                )
            if low_stripped in _ORG_COMMON_SURNAMES_DENYLIST:
                raise ValueError(
                    "ORG_JURIDICA acronym looks like a common surname; likely mislabeled person name: "
                    f"{value!r}"
                )

        multiword = len(tokens) >= 2 and letters >= 2
        # Short org references like "5ª DP" (delegacia), "1ª VT" (vara), "2º BPM" (batalhão) are valid
        short_numbered_org = bool(re.fullmatch(r"\d+[ªº]?\s*[A-Za-z]{2,}", raw))
        singleword = len(tokens) == 1 and letters >= 2

        # Allow single-word org references (e.g. "Detran", "Apex-Brasil") to avoid blocking real agencies.
        # Keep only high-confidence blocks to prevent obvious noise.
        if singleword:
            if low_stripped in _ORG_ACRONYM_DENYLIST:
                raise ValueError(
                    "ORG_JURIDICA token is a known non-organization token in this dataset: "
                    f"{value!r}"
                )
            if low_stripped in _ORG_COMMON_SURNAMES_DENYLIST:
                raise ValueError(
                    "ORG_JURIDICA token looks like a common surname; likely mislabeled person name: "
                    f"{value!r}"
                )
            return

        if not (acronym or multiword or short_numbered_org):
            raise ValueError(f"ORG_JURIDICA must be a plausible organization name/acronym: {value!r}")
        return

    if entity_type == "ENDERECO":
        # Require some numeric component and some address-ish marker to reduce false positives.
        if digits == 0:
            raise ValueError(f"ENDERECO must contain a number/lote/CEP: {value!r}")
        # Note: Trailing \b is removed from the group to avoid issues with patterns ending
        # in non-word chars like "av." or "r." where the boundary check would fail.
        marker = re.search(
            r"\b("
            r"rua|r\.|avenida|av\.|alameda|travessa|quadra|qd|q\.d\.|bloco|bl|lote|lt|"
            r"conjunto|conj|setor|cep|sqs|sqn|shdf|sh|crn|cln|"
            r"l3|eixo|apto|apartamento|"
            # Rural/road addresses common in Brazil
            r"estrada|rodovia|br-|mg-|sp-|go-|df-|pr-|rj-|ba-|rs-|sc-|mt-|ms-|pa-|am-|ce-|pe-|ma-|"
            r"fazenda|sítio|sitio|chácara|chacara|gleba|núcleo rural|nucleo rural|assentamento|km"
            r")",
            low,
            flags=re.IGNORECASE,
        )
        if not marker:
            raise ValueError(f"ENDERECO must include an address marker (Rua/Quadra/Lote/etc.): {value!r}")
        return

    if entity_type == "DADO_SAUDE":
        # Block the most generic admin tokens when standalone.
        banned = {"laudo", "atestado", "prontuário", "prontuario"}
        if low.strip() in banned:
            raise ValueError(
                f"DADO_SAUDE too generic when standalone; include the condition/CID (e.g., 'laudo de ...'): {value!r}"
            )
        return

    # NOME_PESSOA: keep permissive but block obvious noise.
    if entity_type == "NOME_PESSOA":
        if digits > 0:
            raise ValueError(f"NOME_PESSOA must not contain digits: {value!r}")
        if _has_any((" sr", " sra", " dr", " dra", "senhor", "senhora")) and len(value.split()) <= 2:
            # Encourage stripping honorific-only spans.
            raise ValueError(f"NOME_PESSOA should not be only an honorific/title: {value!r}")
        return

    if entity_type == "QUASI_IDENTIFICADOR":
        # Must have at least 3 words (function + context) to form an indirect identifier.
        tokens = value.split()
        if len(tokens) < 3:
            raise ValueError(
                f"QUASI_IDENTIFICADOR must contain function + context (at least 3 words): {value!r}"
            )
        # Relaxed: only require minimum length for indirect identifiers.
        # The combination of role + context is enforced via prompt guidelines, not hard validation.
        if len(value) < 15:
            raise ValueError(
                f"QUASI_IDENTIFICADOR must be sufficiently descriptive (>= 15 chars): {value!r}"
            )
        return


