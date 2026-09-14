# Fashion-MNIST Benchmark MVP

Estudo de classificação de imagens de roupas com **SVM, MLP e CNN**, com comparação de desempenho, robustez e comportamento em imagens externas. O aplicativo Streamlit reúne resultados experimentais e permite executar predições com modelos treinados.

## Funcionalidades

- Exploração do Fashion-MNIST e comparação dos três modelos.
- Análise de robustez, mudança de domínio e seleção de modelos.
- Inferência por upload com probabilidades, indicadores de incerteza e visualização do preprocessing.
- CV Capture Lab para aquisição e preparação de imagens.
- Consulta de evidências do MLflow, quando um banco local ou servidor remoto está disponível.

Uma confiança alta na predição não garante acerto em imagens externas ao domínio de treinamento.

## Executar localmente

Use **Python 3.11**. No PowerShell, a partir da pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r app/requirements.txt
python -m streamlit run app/streamlit_app.py
```

Acesse o endereço exibido no terminal, normalmente `http://localhost:8501`.
No Linux/macOS, ative o ambiente com `source .venv/bin/activate`; os demais comandos são iguais.

Por padrão, a inferência executa dentro do Streamlit e mantém os modelos em cache. Não é necessário iniciar a FastAPI nesse modo. Os arquivos esperados são:

```text
models/cnn.keras
models/mlp.keras
models/svm.joblib
```

O Fashion-MNIST é baixado pelo Keras ao abrir a página Dataset pela primeira vez; essa etapa precisa de conexão com a internet.

## Treinar e avaliar

Para instalar também as ferramentas de treinamento e API:

```powershell
python -m pip install -r requirements.txt
python -m src.training.train_svm
python -m src.training.train_mlp
python -m src.training.train_cnn
python -m src.evaluation.benchmark
python -m src.robustness.stress_test
```

O treinamento atualiza os arquivos em `models/` e gera evidências em `artifacts/`. Ajuste o tamanho da amostra do SVM, as épocas e o tamanho dos lotes em `config/settings.py`. Treinar pode exigir mais memória e tempo que executar o aplicativo com os modelos prontos.

## FastAPI opcional

Para executar o serviço de inferência separadamente:

```powershell
python -m uvicorn api.main:app --reload --port 8000
```

A documentação interativa fica em `http://localhost:8000/docs`.
Em outro terminal, com o ambiente virtual ativo:

```powershell
$env:INFERENCE_API_URL = "http://localhost:8000"
python -m streamlit run app/streamlit_app.py
```

Para voltar à inferência dentro do Streamlit, remova a variável antes de iniciar o aplicativo:

```powershell
Remove-Item Env:INFERENCE_API_URL -ErrorAction SilentlyContinue
```

## MLflow

Para visualizar os experimentos locais, execute na raiz do projeto:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

A página **Experiment Evidence** consulta `mlflow.db` ou um servidor configurado pela variável `MLFLOW_TRACKING_URI`. A variável opcional `MLFLOW_UI_URL` aponta para a interface do MLflow.

O banco local e os diretórios de execução não são enviados ao GitHub. Na nuvem, essa página precisa de um servidor MLflow remoto; sem ele, informa a ausência de evidências disponíveis. As páginas de resultados usam os CSVs versionados.

## Publicar no Streamlit Community Cloud

Depois de enviar as alterações ao GitHub, acesse [Streamlit Community Cloud](https://share.streamlit.io/) e configure:

| Campo | Valor |
| --- | --- |
| Repository | `HilarioGouveia-Academicos/fashion_mnist_benchmark_mvp` |
| Branch | `main` |
| Main file path | `app/streamlit_app.py` |
| Python, em Advanced settings | `3.11` |

As dependências do aplicativo ficam em `app/requirements.txt`, ao lado do arquivo de entrada. O deploy utiliza os modelos já treinados, sem iniciar um treinamento.

O repositório pode ser privado, desde que o Streamlit tenha autorização para acessá-lo. Confira também a configuração de compartilhamento do aplicativo para definir quem poderá usá-lo.

Consulte [o guia de deploy](docs/STREAMLIT_DEPLOY.md) e a [documentação oficial de dependências](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Organização

```text
api/                 API de inferência
app/                 Streamlit, páginas e dependências de execução
artifacts/           Resultados e evidências experimentais
config/              Configuração do estudo
data/                Dados locais, excluídos do Git
docs/                Documentação do estudo e deploy
models/              Modelos treinados usados na inferência
src/                 Dados, treinamento, avaliação e preprocessing
tests/               Testes e verificações do projeto
tools/               Ferramentas de curadoria e preparação de dados
requirements.txt     Dependências para treinamento e API
```

O `.gitignore` permite versionar os três modelos de inferência e os CSVs selecionados para o dashboard. Exclui o ambiente virtual, caches, datasets locais, arquivos ZIP, banco MLflow e demais artefatos gerados. Credenciais devem permanecer fora do repositório.

## Enviar alterações ao GitHub

Revise os arquivos antes do commit:

```powershell
git status
git diff
git add .
git commit -m "Update application and documentation"
git push origin main
```
