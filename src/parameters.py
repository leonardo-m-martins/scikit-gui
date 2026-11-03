import re
import unicodedata
from typing import Any

from src.enums import Methods, Modes, Preprocessors
from src.ml_common import ModelSpec

SCHEMA_VERSION = 1

MODE_SLUGS = {
    Modes.CLASSIFICATION: "classificacao",
    Modes.REGRESSION: "regressao",
}


class ParameterFileError(ValueError):
    pass


def build_payload(mode: Modes, method: Methods, spec: ModelSpec) -> dict:
    return {
        "version": SCHEMA_VERSION,
        "mode": mode.value,
        "method": method.value,
        "preprocessing": spec.preprocessing.value,
        "params": spec.params,
    }


def parse_payload(
    payload: Any,
    mode: Modes,
    specs: dict[Methods, ModelSpec],
) -> tuple[Methods, ModelSpec]:
    if not isinstance(payload, dict):
        raise ParameterFileError("O arquivo deve conter um objeto JSON.")

    version = payload.get("version")
    if version != SCHEMA_VERSION:
        raise ParameterFileError(f"Versão de arquivo não suportada: {version!r}.")

    file_mode = payload.get("mode")
    if file_mode != mode.value:
        raise ParameterFileError(
            f"O arquivo pertence ao modo {file_mode!r}, "
            f"mas o modo atual é {mode.value!r}."
        )

    method_name = payload.get("method")
    if not isinstance(method_name, str):
        raise ParameterFileError("Campo 'method' ausente ou inválido.")

    try:
        method = Methods(method_name)
    except ValueError as exc:
        raise ParameterFileError(
            f"Método desconhecido: {method_name!r}."
        ) from exc

    spec = specs.get(method)
    if spec is None:
        raise ParameterFileError(
            f"O método {method_name!r} não está disponível "
            f"no modo {mode.value}."
        )

    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise ParameterFileError("Campo 'params' deve ser um objeto JSON.")

    preprocessing_name = payload.get("preprocessing", Preprocessors.NONE.value)
    try:
        preprocessing = Preprocessors(preprocessing_name)
    except ValueError as exc:
        raise ParameterFileError(
            f"Pré-processamento desconhecido: {preprocessing_name!r}."
        ) from exc

    params = normalize_params(spec.model_type, params)

    try:
        spec.model_type(**params)
    except Exception as exc:  # noqa: BLE001
        raise ParameterFileError(
            f"Parâmetros inválidos para {method_name}: {exc}"
        ) from exc

    return method, ModelSpec(
        model_type=spec.model_type,
        params=params,
        preprocessing=preprocessing,
    )


def normalize_params(model_type, params: dict) -> dict:
    defaults = model_type().get_params()

    normalized = {}
    for key, value in params.items():
        default = defaults.get(key)
        if isinstance(default, tuple) and isinstance(value, list):
            value = tuple(value)
        normalized[key] = value

    return normalized


def default_filename(mode: Modes, method: Methods) -> str:
    slug = _slugify(method.value)
    return f"parametros_{MODE_SLUGS[mode]}_{slug}.json"


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")
