"""
AstrBot 万象画卷插件 - 数据模型与配置归一化。
"""
import base64
import binascii
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PLUGIN_NAME = "astrbot_plugin_omnidraw"
PLUGIN_AUTHOR = "雪碧bir"
PLUGIN_VERSION = "3.2.1"


@dataclass
class ProviderConfig:
    id: str
    api_type: str
    base_url: str
    api_keys: List[str]
    model: str
    timeout: float
    available_models: List[str] = field(default_factory=list)

    @property
    def has_api_key(self) -> bool:
        return any(key.strip() for key in self.api_keys)


@dataclass
class PluginConfig:
    providers: List[ProviderConfig]
    video_providers: List[ProviderConfig]
    chains: Dict[str, List[str]]
    presets: Dict[str, str]
    enable_optimizer: bool
    optimizer_model: str
    optimizer_timeout: float
    max_batch_count: int
    persona_name: str
    persona_base_prompt: str
    persona_ref_image: str
    persona_ref_images: List[str]
    allowed_users: List[str]
    optimizer_style: str
    optimizer_custom_prompt: str
    verbose_report: bool

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], data_dir: str) -> "PluginConfig":
        if not isinstance(config_dict, dict):
            config_dict = {}

        providers = [_build_provider_config(p, is_video=False) for p in _as_list(config_dict.get("providers", []))]
        providers = [provider for provider in providers if provider.id]

        video_providers = [
            _build_provider_config(p, is_video=True) for p in _as_list(config_dict.get("video_providers", []))
        ]
        video_providers = [provider for provider in video_providers if provider.id]

        presets_dict = {}
        normalized_presets = []
        for preset in _as_list(config_dict.get("presets", [])):
            if isinstance(preset, dict):
                name = str(preset.get("name", "")).strip()
                prompt = str(preset.get("prompt", "")).strip()
            elif isinstance(preset, str) and ":" in preset:
                name, prompt = preset.split(":", 1)
                name = name.strip()
                prompt = prompt.strip()
            else:
                continue
            if name:
                presets_dict[name] = prompt
                normalized_presets.append(f"{name}:{prompt}")
        config_dict["presets"] = normalized_presets

        persona_conf = _ensure_dict(config_dict, "persona_config")
        opt_conf = _ensure_dict(config_dict, "optimizer_config")
        router_conf = _ensure_dict(config_dict, "router_config")
        perm_conf = _ensure_dict(config_dict, "permission_config")

        processed_images = _process_persona_images(
            persona_conf.get("persona_ref_image", []),
            os.path.join(data_dir, "persona_refs"),
        )
        persona_conf["persona_ref_image"] = processed_images

        chains = {
            "text2img": _parse_chain(router_conf.get("chain_text2img", "node_1")),
            "selfie": _parse_chain(router_conf.get("chain_selfie", "node_1")),
            "video": _parse_chain(router_conf.get("chain_video", "video_node_1")),
            "optimizer": _parse_chain(opt_conf.get("chain_optimizer", "node_1")),
        }

        optimizer_model = str(opt_conf.get("optimizer_model", "")).strip()
        if not optimizer_model and providers:
            optimizer_model = providers[0].model

        return cls(
            providers=providers,
            video_providers=video_providers,
            chains=chains,
            presets=presets_dict,
            enable_optimizer=_to_bool(opt_conf.get("enable_optimizer", True)),
            optimizer_model=optimizer_model or "gpt-4o-mini",
            optimizer_timeout=_to_float(opt_conf.get("optimizer_timeout", 15.0), 15.0, minimum=1.0),
            max_batch_count=_to_int(opt_conf.get("max_batch_count", 0), 0, minimum=0),
            persona_name=str(persona_conf.get("persona_name", "默认助理")).strip() or "默认助理",
            persona_base_prompt=str(persona_conf.get("persona_base_prompt", "")),
            persona_ref_image=processed_images[0] if processed_images else "",
            persona_ref_images=processed_images,
            allowed_users=_parse_allowed_users(perm_conf.get("allowed_users", "")),
            optimizer_style=str(opt_conf.get("optimizer_style", "手机日常原生感")).strip() or "手机日常原生感",
            optimizer_custom_prompt=str(opt_conf.get("optimizer_custom_prompt", "")),
            verbose_report=_to_bool(config_dict.get("verbose_report", False)),
        )

    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        for provider in self.providers:
            if provider.id == provider_id:
                return provider
        return None

    def get_video_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        for provider in self.video_providers:
            if provider.id == provider_id:
                return provider
        return None


def _ensure_dict(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def _as_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _split_csv_or_lines(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).replace("\r", "\n").replace(",", "\n").split("\n")
    return [str(item).strip() for item in items if str(item).strip()]


def _parse_models(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = str(value or "").split(",")
    seen = set()
    models = []
    for item in raw_items:
        model = str(item).strip()
        if model and model not in seen:
            seen.add(model)
            models.append(model)
    return models


def _normalize_api_type(value: Any, is_video: bool) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "async_task" if is_video else "openai_image"
    lowered = raw.lower()
    if is_video:
        if "chat" in lowered or "对话" in raw:
            return "openai_chat"
        if "sync" in lowered or "同步" in raw:
            return "openai_sync"
        return "async_task"
    if "chat" in lowered or "对话" in raw:
        return "openai_chat"
    return "openai_image"


def _build_provider_config(raw_provider: Any, is_video: bool) -> ProviderConfig:
    if not isinstance(raw_provider, dict):
        raw_provider = {}

    model_raw = raw_provider.get("model", raw_provider.get("模型名称", ""))
    available_models = _parse_models(raw_provider.get("available_models", []))
    if not available_models:
        available_models = _parse_models(model_raw)

    model = str(model_raw or "").strip()
    if "," in model:
        model = model.split(",", 1)[0].strip()
    if not model and available_models:
        model = available_models[0]
    if model and model not in available_models:
        available_models.insert(0, model)

    default_timeout = 300.0 if is_video else 60.0
    return ProviderConfig(
        id=str(raw_provider.get("id", raw_provider.get("节点ID", ""))).strip(),
        api_type=_normalize_api_type(raw_provider.get("api_type", raw_provider.get("接口模式", "")), is_video),
        base_url=str(
            raw_provider.get(
                "base_url",
                raw_provider.get("接口地址 (需含/v1或/v2)", raw_provider.get("接口地址 (需含/v1)", "")),
            )
        ).strip(),
        api_keys=_split_csv_or_lines(raw_provider.get("api_keys", raw_provider.get("API密钥", ""))),
        model=model,
        timeout=_to_float(raw_provider.get("timeout", raw_provider.get("超时时间(秒)", default_timeout)), default_timeout, 1.0),
        available_models=available_models,
    )


def _parse_chain(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item for item in _split_csv_or_lines(value) if item]


def _parse_allowed_users(value: Any) -> List[str]:
    return _split_csv_or_lines(value)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"false", "0", "no", "off", "关闭"}


def _to_float(value: Any, default: float, minimum: Optional[float] = None) -> float:
    try:
        result = float(str(value).strip())
    except Exception:
        result = default
    if minimum is not None:
        result = max(minimum, result)
    return result


def _to_int(value: Any, default: int, minimum: Optional[int] = None) -> int:
    try:
        result = int(float(str(value).strip()))
    except Exception:
        result = default
    if minimum is not None:
        result = max(minimum, result)
    return result


def _process_persona_images(raw_images: Any, refs_dir: str) -> List[str]:
    os.makedirs(refs_dir, exist_ok=True)
    processed_images = []

    for idx, img_data in enumerate(_as_list(raw_images)):
        if not img_data:
            continue
        img_ref = str(img_data)
        if img_ref.startswith("data:image"):
            saved_path = _save_data_url_image(img_ref, refs_dir, idx)
            if saved_path:
                processed_images.append(saved_path)
        else:
            processed_images.append(img_ref)

    _cleanup_unused_persona_refs(refs_dir, processed_images)
    return processed_images


def _save_data_url_image(data_url: str, refs_dir: str, idx: int) -> str:
    try:
        header, base64_str = data_url.split(",", 1)
        ext = "png"
        if "jpeg" in header or "jpg" in header:
            ext = "jpg"
        elif "webp" in header:
            ext = "webp"
        filename = f"ref_{int(time.time() * 1000)}_{idx}.{ext}"
        filepath = os.path.join(refs_dir, filename)
        with open(filepath, "wb") as file:
            file.write(base64.b64decode(base64_str, validate=False))
        return filepath
    except (ValueError, binascii.Error, OSError):
        return ""


def _cleanup_unused_persona_refs(refs_dir: str, active_refs: List[str]) -> None:
    active_paths = {os.path.abspath(ref) for ref in active_refs if not str(ref).startswith("http")}
    try:
        filenames = os.listdir(refs_dir)
    except OSError:
        return

    for filename in filenames:
        if not filename.startswith("ref_"):
            continue
        filepath = os.path.abspath(os.path.join(refs_dir, filename))
        if filepath in active_paths or not os.path.isfile(filepath):
            continue
        try:
            os.remove(filepath)
        except OSError:
            continue
