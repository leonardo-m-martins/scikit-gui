import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.enums import Methods, Modes, Preprocessors
from src.ml.common import ModeConfig, ModelSpec

SCHEMA_VERSION = 1

MODE_SLUGS = {
    Modes.CLASSIFICATION: "classificacao",
    Modes.REGRESSION: "regressao",
}

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class ParameterFileError(ValueError):
    pass


@dataclass(frozen=True)
class LoadReport:
    applied: list[str]
    warnings: list[str]


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
        raise ParameterFileError(f"Método desconhecido: {method_name!r}.") from exc

    spec = specs.get(method)
    if spec is None:
        raise ParameterFileError(
            f"O método {method_name!r} não está disponível " f"no modo {mode.value}."
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
    except Exception as exc:
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


def read_payload(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def export_configs(
    configs: dict[Modes, ModeConfig],
    directory: Path = CONFIG_DIR,
) -> list[Path]:
    paths = []

    for mode, config in configs.items():
        for method, spec in config.specs.items():
            path = directory / default_filename(mode, method)
            write_payload(path, build_payload(mode, method, spec))
            paths.append(path)

    return paths


def load_configs(
    configs: dict[Modes, ModeConfig],
    directory: Path = CONFIG_DIR,
) -> LoadReport:
    applied = []
    warnings = []

    if not directory.is_dir():
        warnings.append(f"Diretório de configurações não encontrado: {directory}")
        return LoadReport(applied, warnings)

    paths = sorted(directory.glob("*.json"))
    if not paths:
        warnings.append(f"Nenhum arquivo de configuração em: {directory}")
        return LoadReport(applied, warnings)

    for path in paths:
        try:
            payload = read_payload(path)

            if not isinstance(payload, dict):
                raise ParameterFileError("O arquivo deve conter um objeto JSON.")

            mode = Modes(payload.get("mode"))
            specs = configs[mode].specs
            method, spec = parse_payload(payload, mode, specs)
        except (OSError, ValueError, KeyError) as exc:
            warnings.append(f"{path.name}: {exc}")
            continue

        specs[method] = spec
        applied.append(path.name)

    return LoadReport(applied, warnings)


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")
