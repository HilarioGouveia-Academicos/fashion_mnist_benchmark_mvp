# V2B Curator V4

Arquitetura segura:

```text
v2b_candidate_manifest_master.csv  [READ ONLY]
                +
v2b_curation_state.csv             [MUTABLE]
                ↓
             JOIN source_id
                ↓
          Curated Snapshot
```

A UI nunca grava no master.

## Instalação no projeto

1. Faça backup de `artifacts/v2b/`.
2. Copie o master e o state deste pacote para `artifacts/v2b/`.
3. Copie `src/v2b/curation_rules.py` e `src/v2b/manifest_guard.py`.
4. Copie `tools/v2b_curator_v4.py`.
5. Mantenha seu `config/v2b_mapping.json` atual.

Execute:

```powershell
streamlit run tools/v2b_curator_v4.py
```

O Integrity Guard bloqueia a aplicação se o master tiver classes ausentes,
`source_id` duplicado ou `dataset_index` inválido.

Importante: o botão de exportação cria apenas um snapshot. Nunca use esse
snapshot para substituir o master.
