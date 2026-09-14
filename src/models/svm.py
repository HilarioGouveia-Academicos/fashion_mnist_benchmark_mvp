from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm(
    c: float = 10.0,
    kernel: str = "rbf",
    calibration_cv: int = 3,
) -> Pipeline:
    """
    Constrói o pipeline do classificador SVM utilizado no benchmark.

    O SVC é calibrado explicitamente para disponibilizar
    probabilidades através de predict_proba(), permitindo o uso
    posterior em interfaces como Streamlit e FastAPI.

    Parameters
    ----------
    c : float, default=10.0
        Parâmetro de regularização do SVM.

    kernel : str, default="rbf"
        Kernel utilizado pelo SVC.

    calibration_cv : int, default=3
        Número de folds utilizados para calibração das probabilidades.

    Returns
    -------
    Pipeline
        Pipeline contendo StandardScaler e SVM calibrado.
    """

    base_svm = SVC(
        C=c,
        kernel=kernel,
    )

    calibrated_svm = CalibratedClassifierCV(
        estimator=base_svm,
        method="sigmoid",
        cv=calibration_cv,
    )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", calibrated_svm),
        ]
    )