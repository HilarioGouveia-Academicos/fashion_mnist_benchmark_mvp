# Streamlit Community Cloud

- Repository: `HilarioGouveia-Academicos/fashion_mnist_benchmark_mvp`
- Branch: `main`
- Main file: `app/streamlit_app.py`
- Advanced settings: Python **3.11** (matching the tested local environment).

Community Cloud installs `app/requirements.txt`, next to the entrypoint.
The root requirements file still supports local training and FastAPI.

Inference runs inside Streamlit by default and caches the three trained models.
Set the environment variable `INFERENCE_API_URL` to use a separately hosted
FastAPI service. No API key is required for built-in inference.

The three model files and the dashboard summary CSVs are versioned for deployment.
Raw datasets, training runs, ZIP backups and the local MLflow database stay excluded.
Fashion-MNIST is downloaded by Keras when the Dataset page is opened.
Experiment Evidence requires the local MLflow database or a remote MLflow service;
it displays an empty-state message when neither is available.

Deploy from https://share.streamlit.io using the settings above. Keep the app's
visibility aligned with the intended audience; a private repository alone does
not determine who can view the app.
