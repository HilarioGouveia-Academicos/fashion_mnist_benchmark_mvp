# SDD-V2B --- Real External Domain Validation

**Projeto:** Fashion-MNIST Reliability Study\
**Documento:** Software / Experimental Design Specification\
**ID:** SDD-V2B\
**Versão:** 1.0\
**Status:** Proposed / Pre-Experiment\
**Escopo:** Real External Domain Validation\
**Modelo de continuidade:** SVM × MLP × CNN para comparação; CNN como
candidata à adaptação\
**Princípio central:** *Clean Accuracy ≠ Production Readiness*

------------------------------------------------------------------------

## 1. Purpose

Este documento especifica a etapa **V2B --- Real External Domain
Validation** do projeto *Fashion-MNIST Reliability Study*.

A V2B tem como finalidade avaliar, de forma controlada e rastreável, o
comportamento dos modelos previamente treinados quando submetidos a
**imagens externas reais**, semanticamente compatíveis com as dez
classes do Fashion-MNIST, mas provenientes de uma distribuição visual
diferente daquela utilizada durante treinamento e teste.

A etapa sucede:

``` text
Clean Benchmark
      ↓
Robustness V2
      ↓
Controlled Domain Shift V1
      ↓
Synthetic External V2A — RAW
      ↓
Synthetic External V2A.1 — Canonicalized
      ↓
Real External V2B
```

A V2B é uma etapa de **avaliação**, e não de otimização. Nenhum
fine-tuning ou ajuste do canonicalizer deve ser realizado com base nos
resultados do conjunto de avaliação congelado.

------------------------------------------------------------------------

## 2. Research Motivation

Os experimentos anteriores demonstraram que alto desempenho no conjunto
de teste original não garante comportamento equivalente quando a
distribuição visual das entradas muda.

No piloto sintético V2A, os modelos apresentaram forte degradação no
domínio externo RAW. A canonicalização V2A.1 recuperou parte relevante
do desempenho, especialmente para a CNN, mas não eliminou o domain gap.

Entretanto, V2A utiliza um conjunto sintético pequeno e não constitui
evidência suficiente de generalização em imagens reais.

A V2B introduz uma fonte externa real para verificar se os padrões
observados anteriormente persistem fora do ambiente sintético.

------------------------------------------------------------------------

## 3. Research Question

### 3.1 Primary Question

> **Como os modelos treinados exclusivamente no Fashion-MNIST se
> comportam quando avaliados em imagens externas reais, antes e depois
> da canonicalização?**

### 3.2 Secondary Questions

1.  A canonicalização reduz o generalization gap em imagens reais?
2.  O ranking relativo entre SVM, MLP e CNN permanece semelhante ao
    observado nos experimentos anteriores?
3.  Quais classes apresentam maior degradação sob domain shift?
4.  Os modelos continuam produzindo erros com alta confiança?
5.  A CNN mantém evidências suficientes para justificar sua seleção como
    candidata à fase de domain adaptation?

------------------------------------------------------------------------

## 4. Hypotheses

As hipóteses são definidas **antes da execução** e não constituem
resultados esperados obrigatórios.

### H1 --- External Domain Degradation

O desempenho no domínio externo real será inferior ao desempenho
observado no Fashion-MNIST clean.

### H2 --- Canonicalization Effect

A canonicalização deverá reduzir parte do domain gap ao aproximar a
representação externa do contrato visual esperado pelos modelos.

### H3 --- Residual Domain Gap

Mesmo após canonicalização, deverá permanecer uma diferença mensurável
entre o desempenho clean e o desempenho externo.

### H4 --- Architecture Sensitivity

SVM, MLP e CNN poderão apresentar diferentes níveis de sensibilidade ao
domain shift.

### H5 --- Confidence--Reliability Gap

Poderão ocorrer erros com alta confiança, demonstrando que confiança
preditiva isolada não constitui evidência suficiente de confiabilidade.

Todas as hipóteses podem ser rejeitadas pelos resultados sem invalidar o
experimento.

------------------------------------------------------------------------

## 5. Scope

### 5.1 In Scope

-   construção de dataset externo real;
-   validação manual dos rótulos;
-   avaliação SVM, MLP e CNN;
-   avaliação RAW;
-   avaliação Canonicalized;
-   métricas globais;
-   métricas por classe;
-   confusion matrix;
-   prediction distribution;
-   análise de confiança e incerteza;
-   generalization gap;
-   canonicalization gain;
-   registro experimental no MLflow;
-   geração de artefatos consolidados;
-   integração posterior dos resultados ao Streamlit.

### 5.2 Out of Scope

Não fazem parte da V2B:

-   fine-tuning;
-   retraining;
-   data augmentation orientada pelos resultados;
-   alteração do canonicalizer após inspeção dos resultados;
-   calibration tuning;
-   threshold optimization;
-   treinamento com imagens V2B;
-   afirmação de production readiness;
-   detecção formal de OOD.

Esses itens pertencem às fases posteriores de adaptação e reavaliação.

------------------------------------------------------------------------

## 6. Dataset Protocol

### 6.1 Target Classes

A taxonomia deve permanecer compatível com Fashion-MNIST:

    ID Class
  ---- -------------
     0 T-shirt/top
     1 Trouser
     2 Pullover
     3 Dress
     4 Coat
     5 Sandal
     6 Shirt
     7 Sneaker
     8 Bag
     9 Ankle boot

### 6.2 Dataset Size

Para o primeiro benchmark real:

-   **mínimo recomendado:** 15 imagens por classe;
-   **preferencial:** 30--50 imagens por classe;
-   classes devem ser tão balanceadas quanto possível.

O tamanho efetivo deverá ser registrado no relatório experimental.

### 6.3 Diversity Requirements

O conjunto deve buscar diversidade em:

-   fundo;
-   iluminação;
-   enquadramento;
-   escala do objeto;
-   orientação;
-   textura;
-   cor original;
-   resolução;
-   proporção;
-   estilo do produto;
-   condições de captura.

A diversidade deve representar domain shift natural, não corrupção
artificial deliberada.

### 6.4 Exclusion Criteria

Devem ser excluídas:

-   imagens derivadas do Fashion-MNIST;
-   duplicatas ou near-duplicates conhecidos;
-   imagens com classe ambígua sem consenso;
-   imagens em que o objeto-alvo não seja identificável;
-   imagens cuja classe não possa ser mapeada de forma defensável para a
    taxonomia Fashion-MNIST.

### 6.5 Proposed Directory Structure

``` text
data/
└── external_real_v2b/
    ├── 0_tshirt_top/
    ├── 1_trouser/
    ├── 2_pullover/
    ├── 3_dress/
    ├── 4_coat/
    ├── 5_sandal/
    ├── 6_shirt/
    ├── 7_sneaker/
    ├── 8_bag/
    └── 9_ankle_boot/
```

### 6.6 Metadata

Recomenda-se manter um manifesto:

``` text
data/external_real_v2b/manifest.csv
```

Campos mínimos:

  Field              Description
  ------------------ ---------------------------------------------
  image_id           identificador único
  filename           arquivo
  class_id           classe numérica
  class_name         classe textual
  source_type        origem/categoria da fonte
  source_reference   referência de proveniência quando aplicável
  collected_at       data de inclusão
  reviewer_status    status da revisão
  notes              observações

------------------------------------------------------------------------

## 7. Dataset Freeze Protocol

Antes da primeira avaliação oficial:

``` text
Collect
   ↓
Label
   ↓
Review
   ↓
Validate
   ↓
Generate Manifest
   ↓
Freeze Dataset
   ↓
Evaluate
```

Após o freeze:

1.  nenhuma imagem deve ser substituída por apresentar erro;
2.  nenhuma classe deve ser removida porque prejudicou uma métrica;
3.  o canonicalizer não deve ser ajustado usando os resultados V2B;
4.  alterações posteriores exigem uma nova versão do dataset;
5.  a versão/hash do dataset deve ser registrada.

Exemplo:

``` text
dataset_name = fashion_mnist_external_real
dataset_version = v2b_1
status = frozen
```

------------------------------------------------------------------------

## 8. Evaluation Design

Cada modelo será avaliado em dois caminhos.

### 8.1 RAW Path

``` text
Real Image
    ↓
Model-specific required preprocessing
    ↓
Model
    ↓
Prediction
```

Nenhuma canonicalização adicional é aplicada.

### 8.2 Canonicalized Path

``` text
Real Image
    ↓
Canonicalizer V2 (Frozen)
    ↓
Model-specific preprocessing
    ↓
Model
    ↓
Prediction
```

### 8.3 Experimental Matrix

  Model    RAW   Canonicalized
  ------- ----- ---------------
  SVM       ✓          ✓
  MLP       ✓          ✓
  CNN       ✓          ✓

Total: **6 condições experimentais principais**.

------------------------------------------------------------------------

## 9. Canonicalization Contract

A V2B deve reutilizar o **Canonicalizer V2 congelado** na etapa V2A.1.

A canonicalização deve permanecer independente da classe prevista e não
pode ser ajustada individualmente para melhorar uma imagem específica.

Contrato conceitual:

``` text
External Image
      ↓
Input Validation
      ↓
Canonicalizer V2
      ↓
28 × 28 canonical representation
      ↓
Model-specific preprocessing
```

A versão do canonicalizer deverá ser registrada no MLflow.

------------------------------------------------------------------------

## 10. Metrics

### 10.1 Predictive Performance

Registrar:

-   Accuracy;
-   Precision Macro;
-   Recall Macro;
-   F1 Macro.

### 10.2 Class-Level Performance

Registrar:

-   Recall por classe;
-   Precision por classe;
-   F1 por classe;
-   support por classe.

### 10.3 Uncertainty / Confidence

Registrar:

-   Mean Confidence;
-   Confidence on Correct Predictions;
-   Confidence on Errors;
-   Entropy;
-   Normalized Entropy;
-   Prediction Margin.

Essas medidas são indicadores auxiliares e **não devem ser descritas
como medidas calibradas de reliability** sem experimento específico de
calibração.

### 10.4 High-Confidence Error Rate

Adicionar:

``` text
High Confidence Error Rate =
errors with confidence >= threshold
-----------------------------------
total errors
```

Threshold inicial de relatório:

``` text
confidence >= 0.90
```

O threshold deve permanecer fixo durante a V2B e ser tratado como
critério descritivo, não como threshold validado de segurança.

------------------------------------------------------------------------

## 11. Generalization Gap

Métrica central:

``` text
Generalization Gap =
Clean Accuracy - External Accuracy
```

Calcular:

``` text
RAW Generalization Gap
Canonicalized Generalization Gap
```

por modelo.

Exemplo estrutural:

  Model     Clean   External RAW   RAW Gap   Canonical   Canonical Gap
  ------- ------- -------------- --------- ----------- ---------------
  SVM         ...            ...       ...         ...             ...
  MLP         ...            ...       ...         ...             ...
  CNN         ...            ...       ...         ...             ...

------------------------------------------------------------------------

## 12. Canonicalization Gain

Métrica:

``` text
Canonicalization Gain =
Canonicalized Accuracy - RAW Accuracy
```

Deve ser calculada:

-   globalmente;
-   por modelo;
-   por classe.

Importante: ganho positivo não implica que o domain gap foi eliminado.

------------------------------------------------------------------------

## 13. Per-Class Analysis

A análise por classe é obrigatória.

Especial atenção deve ser dada às classes visualmente próximas:

``` text
T-shirt/top
Shirt
Pullover
Coat
```

O relatório deve permitir identificar:

-   classes mais robustas;
-   classes mais sensíveis;
-   classes beneficiadas pela canonicalização;
-   classes prejudicadas pela canonicalização;
-   pares de classes frequentemente confundidos.

------------------------------------------------------------------------

## 14. Confusion Matrix

Gerar para cada modelo:

``` text
External RAW
External Canonicalized
```

Para a CNN, manter essas matrizes como artefatos principais da análise.

A ordem das classes deve permanecer fixa de 0 a 9.

------------------------------------------------------------------------

## 15. Prediction Distribution

Registrar a distribuição de classes previstas:

``` text
prediction_distribution.csv
```

Objetivo: detectar possíveis colapsos de classificação.

Exemplos de sinais relevantes:

-   concentração excessiva em uma classe;
-   classes nunca previstas;
-   distribuição fortemente incompatível com o conjunto balanceado.

------------------------------------------------------------------------

## 16. Error Analysis

Gerar um dataset de erros contendo, no mínimo:

  Field             Description
  ----------------- -------------------
  image_id          imagem
  true_class        classe real
  predicted_class   previsão
  confidence        confiança
  entropy           entropia
  margin            prediction margin
  preprocessing     RAW/canonicalized
  model             modelo

Separar também:

``` text
high_confidence_errors.csv
```

para erros com confidence ≥ 0.90.

------------------------------------------------------------------------

## 17. Artifact Specification

Estrutura recomendada:

``` text
artifacts/
└── domain_shift_v2b/
    ├── external_real_results.csv
    ├── raw_vs_canonical_real.csv
    ├── per_class_results.csv
    ├── prediction_distribution.csv
    ├── errors.csv
    ├── high_confidence_errors.csv
    ├── confusion_matrix_svm_raw.png
    ├── confusion_matrix_svm_canonical.png
    ├── confusion_matrix_mlp_raw.png
    ├── confusion_matrix_mlp_canonical.png
    ├── confusion_matrix_cnn_raw.png
    ├── confusion_matrix_cnn_canonical.png
    └── report.md
```

Os CSVs consolidados devem ser adequados ao consumo posterior pelo
Streamlit.

------------------------------------------------------------------------

## 18. MLflow Tracking Specification

Cada condição deve gerar run rastreável.

### Suggested Run Names

``` text
svm_external_real_v2b_raw
svm_external_real_v2b_canonical
mlp_external_real_v2b_raw
mlp_external_real_v2b_canonical
cnn_external_real_v2b_raw
cnn_external_real_v2b_canonical
```

### Required Tags

``` text
stage = external_real_v2b
dataset = fashion_mnist_external_real
dataset_version = v2b_1
dataset_status = frozen
domain = real_external
preprocessing = raw | canonicalized
canonicalizer_version = none | v2
evaluation_type = external_validation
```

### Required Metrics

-   accuracy;
-   precision_macro;
-   recall_macro;
-   f1_macro;
-   mean_confidence;
-   correct_confidence;
-   error_confidence;
-   entropy;
-   normalized_entropy;
-   prediction_margin;
-   high_confidence_error_rate;
-   generalization_gap;
-   canonicalization_gain, quando aplicável.

### Required Artifacts

-   confusion matrix;
-   per-class metrics;
-   prediction distribution;
-   error analysis;
-   dataset manifest/version metadata;
-   consolidated evaluation report.

------------------------------------------------------------------------

## 19. Experiment Evidence Principle

A V2B adota formalmente o princípio:

> **Toda conclusão relevante apresentada na camada de storytelling deve
> ser rastreável até uma evidência experimental registrada.**

Responsabilidades:

``` text
Experiment Code
      ↓
   MLflow
Evidence / Traceability
      ↓
Curated Artifacts
      ↓
 Streamlit
Storytelling / Interpretation
```

O Streamlit não deve recalcular o experimento.

------------------------------------------------------------------------

## 20. Streamlit Integration

Após validação e freeze dos resultados, `Domain Shift` poderá ganhar:

``` text
1. Controlled Domain Shift
2. Synthetic External — RAW
3. Canonicalization Impact
4. Real External — V2B
```

Cards sugeridos:

``` text
Best Real RAW Model
Best Real Canonicalized Model
Canonicalization Gain
Remaining Generalization Gap
```

A interface deve consumir somente os artefatos consolidados validados.

------------------------------------------------------------------------

## 21. Acceptance Criteria

A V2B poderá receber status **VALIDATED / FROZEN** quando:

-   [ ] dataset externo real estiver documentado;
-   [ ] rótulos tiverem sido revisados;
-   [ ] dataset tiver versão congelada;
-   [ ] manifesto estiver disponível;
-   [ ] SVM, MLP e CNN tiverem sido avaliados em RAW;
-   [ ] SVM, MLP e CNN tiverem sido avaliados em Canonicalized;
-   [ ] métricas globais estiverem calculadas;
-   [ ] métricas por classe estiverem calculadas;
-   [ ] confusion matrices estiverem geradas;
-   [ ] prediction distributions estiverem geradas;
-   [ ] high-confidence errors estiverem registrados;
-   [ ] generalization gaps estiverem calculados;
-   [ ] canonicalization gains estiverem calculados;
-   [ ] runs estiverem registrados no MLflow;
-   [ ] artefatos consolidados estiverem persistidos;
-   [ ] relatório experimental estiver gerado;
-   [ ] nenhuma adaptação tiver sido realizada usando o conjunto
    congelado.

------------------------------------------------------------------------

## 22. Decision Gate --- Domain Adaptation

A V2B não deve concluir antecipadamente que fine-tuning é necessário.

Os resultados devem orientar o próximo passo.

### Scenario A --- Canonicalization substantially reduces gap

Possível direção:

``` text
Serving Contract
+
Canonicalization
+
targeted validation
```

### Scenario B --- RAW and Canonicalized remain weak

Possível evidência de necessidade de:

``` text
Domain Adaptation
Fine-Tuning
Data Augmentation
```

### Scenario C --- Failures concentrated in specific classes

Possível direção:

``` text
Class-focused error analysis
→ targeted adaptation
```

### Scenario D --- Predictive quality improves but confidence remains unreliable

Possível direção:

``` text
Calibration / Uncertainty Evaluation
```

A escolha deve ser baseada nos resultados observados, não em objetivo
prévio de demonstrar superioridade de uma técnica.

------------------------------------------------------------------------

## 23. Post-V2B Roadmap

Caso a evidência justifique adaptação:

``` text
V2B Real External Validation
           ↓
      CNN Selected
           ↓
Create Adaptation Dataset
           ↓
Baseline External Performance
           ↓
Domain Adaptation / Fine-Tuning
           ↓
External Re-Evaluation
           ↓
Clean Performance Retention
           ↓
Robustness Re-Evaluation
           ↓
Uncertainty / Calibration
           ↓
Production Readiness Assessment
```

------------------------------------------------------------------------

## 24. Catastrophic Forgetting Requirement

Qualquer futura adaptação da CNN deverá ser avaliada também no domínio
original.

A melhoria externa não pode ser analisada isoladamente.

Comparar:

``` text
Before Adaptation
vs
After Adaptation
```

em:

-   Clean Accuracy;
-   Clean F1;
-   External Accuracy;
-   External F1;
-   Robustness;
-   Confidence / uncertainty.

Uma possível métrica futura:

``` text
Clean Retention =
Post-Adaptation Clean Accuracy
------------------------------
Pre-Adaptation Clean Accuracy
```

------------------------------------------------------------------------

## 25. Threats to Validity

A interpretação da V2B deverá considerar:

### Dataset Size

Amostras pequenas aumentam variância e limitam generalização das
conclusões.

### Source Bias

Imagens provenientes de poucas fontes podem representar apenas um
subconjunto do domínio real.

### Label Mapping

Classes reais podem não corresponder perfeitamente à taxonomia
Fashion-MNIST.

### Representation Bias

Canonicalization pode favorecer determinados formatos de objeto ou
fundo.

### Evaluation Leakage

Ajustar preprocessing ou modelo depois de observar o conjunto congelado
compromete sua função como avaliação externa.

### Confidence Interpretation

Softmax confidence não equivale automaticamente a probabilidade
calibrada de acerto.

------------------------------------------------------------------------

## 26. Expected Scientific Contribution

A V2B não tem como objetivo demonstrar que a CNN "funciona no mundo
real".

Sua contribuição é fornecer uma avaliação mais rigorosa sobre **quanto
do desempenho clean permanece quando o domínio muda**, quais tipos de
falha emergem e quanto uma transformação de representação consegue
recuperar sem adaptação do modelo.

Ela fortalece a tese geral:

> **Clean Accuracy ≠ Production Readiness**

e prepara a transição metodológica de **model evaluation** para **model
adaptation**.

------------------------------------------------------------------------

## 27. Definition of Done

``` text
[ ] Protocol defined before evaluation
[ ] Real external dataset collected
[ ] Labels reviewed
[ ] Manifest generated
[ ] Dataset versioned and frozen
[ ] RAW evaluation completed
[ ] Canonicalized evaluation completed
[ ] SVM/MLP/CNN comparison completed
[ ] Global metrics generated
[ ] Per-class metrics generated
[ ] Generalization gaps calculated
[ ] Canonicalization gains calculated
[ ] Confidence/error analysis completed
[ ] Confusion matrices generated
[ ] MLflow runs completed
[ ] Curated artifacts generated
[ ] Experimental report generated
[ ] Results reviewed
[ ] V2B marked VALIDATED / FROZEN
[ ] Decision gate for adaptation completed
```

------------------------------------------------------------------------

## 28. Final Experimental Flow

``` text
Fashion-MNIST
      │
      ▼
Clean Benchmark
      │
      ▼
Robustness
      │
      ▼
Controlled Domain Shift
      │
      ▼
Synthetic External V2A
      │
      ├── RAW
      │
      └── Canonicalized V2A.1
      │
      ▼
Real External V2B
      │
      ├── RAW
      │
      └── Canonicalized
      │
      ▼
Evidence Review
      │
      ▼
Decision Gate
      │
      ├── No Adaptation Yet
      │
      └── CNN Domain Adaptation
                 │
                 ▼
          Post-Adaptation
          Reliability Study
```

------------------------------------------------------------------------

**End of SDD-V2B**
