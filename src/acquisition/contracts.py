from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_REPRESENTATIONS = {
    "original",
    "raw_28x28",
    "canonicalized_v2",
    "canonicalized_v3",
}


@dataclass(frozen=True)
class InferenceInput:
    """Transport-neutral contract between acquisition and inference.

    ``session_state`` is only one possible transport. The semantics of this
    contract are deliberately independent from Streamlit so the same payload
    can later be represented by a REST/mobile client.
    """

    image_bytes: bytes
    filename: str
    mime_type: str
    source: str
    acquisition: str
    representation: str
    canonicalization_applied: bool
    canonicalization_version: str | None
    model_input_ready: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.image_bytes:
            raise ValueError("image_bytes não pode ser vazio.")
        if self.representation not in VALID_REPRESENTATIONS:
            raise ValueError(
                f"Representação inválida: {self.representation!r}. "
                f"Use uma de {sorted(VALID_REPRESENTATIONS)}."
            )

        if self.representation.startswith("canonicalized_") and not self.canonicalization_applied:
            raise ValueError(
                "Representações canonicalized_* devem informar "
                "canonicalization_applied=True."
            )

    def to_session_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract_version"] = "1.0"
        return payload

    @classmethod
    def from_session_payload(cls, payload: dict[str, Any]) -> "InferenceInput":
        if not isinstance(payload, dict):
            raise TypeError("Payload de inferência deve ser um dicionário.")

        data = dict(payload)
        data.pop("contract_version", None)
        return cls(**data)

    @property
    def recommended_preprocessing(self) -> str | None:
        return {
            "raw_28x28": "raw",
            "canonicalized_v2": "canonicalized_v2",
            "canonicalized_v3": "canonicalized_v3",
        }.get(self.representation)
