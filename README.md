# Fashion-MNIST Benchmark MVP

MVP end-to-end: SVM + MLP + CNN + MLflow + FastAPI + Streamlit.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m src.training.train_svm
python -m src.training.train_mlp
python -m src.training.train_cnn

python -m src.evaluation.benchmark
python -m src.robustness.stress_test

# terminal 1
mlflow ui --port 5000

# terminal 2
uvicorn api.main:app --reload --port 8000

# terminal 3
streamlit run app/streamlit_app.py
```

Fashion-MNIST is downloaded automatically by Keras. The SVM sample size and neural-network epochs are configurable in `config/settings.py`.
