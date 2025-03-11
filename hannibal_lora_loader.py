import os
from typing import Dict, List
import logging
import folder_paths
import torch
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class HannibalLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "blocks_spec": ("STRING", {
                    "default": "double:0-19,single:0-39",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders/hannibal"
    OUTPUT_NODE = False
    DESCRIPTION = "Charge un LoRA avec une spécification manuelle des blocs (ex. 'double:0-19,single:0-39')."

    def parse_block_spec(self, block_str: str) -> List[tuple[str, List[int]]]:
        if not block_str:
            return []
        specs = block_str.split(',')
        result = []
        for spec in specs:
            try:
                block_type, range_str = spec.split(':')
                start, end = map(int, range_str.split('-'))
                if start > end:
                    raise ValueError(f"Plage invalide : {start} > {end}")
                result.append((block_type.strip(), list(range(start, end + 1))))
            except ValueError as e:
                log.error(f"Erreur de parsing dans blocks_spec : {spec} - {str(e)}")
                raise ValueError(f"Format invalide dans blocks_spec : {spec}")
        return result

    def convert_key_format(self, key: str) -> str:
        prefixes = ["diffusion_model.", "transformer."]
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        return key

    def filter_lora_keys(self, lora: Dict[str, torch.Tensor], blocks_spec: str) -> Dict[str, torch.Tensor]:
        block_specs = self.parse_block_spec(blocks_spec)
        if not block_specs:
            return lora

        filtered_lora = {}
        for key, value in lora.items():
            base_key = self.convert_key_format(key)
            for block_type, blocks in block_specs:
                if block_type == "double" and "double_blocks" in base_key:
                    try:
                        block_num = int(base_key.split('.')[1])
                        if block_num in blocks:
                            filtered_lora[key] = value
                    except (IndexError, ValueError):
                        continue
                elif block_type == "single" and "single_blocks" in base_key:
                    try:
                        block_num = int(base_key.split('.')[1])
                        if block_num in blocks:
                            filtered_lora[key] = value
                    except (IndexError, ValueError):
                        continue
        return filtered_lora

    def check_for_musubi(self, lora: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prefix = "lora_unet_"
        musubi = False
        lora_alphas = {}
        for key, value in lora.items():
            if key.startswith(prefix):
                lora_name = key.split(".", 1)[0]
                if lora_name not in lora_alphas and "alpha" in key:
                    lora_alphas[lora_name] = value
                    musubi = True
        if musubi:
            log.info("Loading Musubi Tuner format LoRA...")
            converted_lora = {}
            for key, weight in lora.items():
                if key.startswith(prefix):
                    if "alpha" in key:
                        continue
                    lora_name = key.split(".", 1)[0]
                    module_name = lora_name[len(prefix):]
                    module_name = module_name.replace("_", ".")
                    module_name = module_name.replace("double.keys.", "double_blocks.")
                    module_name = module_name.replace("single.keys.", "single_blocks.")
                    module_name = module_name.replace("img.", "img_")
                    module_name = module_name.replace("txt.", "txt_")
                    module_name = module_name.replace("attn.", "attn_")
                    diffusers_prefix = "diffusion_model"
                    if "lora_down" in key:
                        new_key = f"{diffusers_prefix}.{module_name}.lora_A.weight"
                        dim = weight.shape[0]
                    elif "lora_up" in key:
                        new_key = f"{diffusers_prefix}.{module_name}.lora_B.weight"
                        dim = weight.shape[1]
                    else:
                        log.info("unexpected key: %s in Musubi LoRA format", key)
                        continue
                    if lora_name in lora_alphas:
                        scale = lora_alphas[lora_name] / dim
                        scale = scale.sqrt()
                        weight = weight * scale
                    else:
                        log.info("missing alpha for %s", lora_name)
                    converted_lora[new_key] = weight
            return converted_lora
        log.info("Loading Diffusers format LoRA...")
        return lora

    def load_lora(self, model, lora_name: str, strength: float, blocks_spec: str):
        if not lora_name:
            return (model,)

        from comfy.utils import load_torch_file
        from comfy.sd import load_lora_for_models

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA {lora_name} introuvable à {lora_path}")

        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if self.loaded_lora is None:
            lora = load_torch_file(lora_path)
            self.loaded_lora = (lora_path, lora)

        diffusers_lora = self.check_for_musubi(lora)
        filtered_lora = self.filter_lora_keys(diffusers_lora, blocks_spec)

        new_model, _ = load_lora_for_models(model, None, filtered_lora, strength, 0)
        if new_model is not None:
            return (new_model,)

        return (model,)

    @classmethod
    def IS_CHANGED(s, model, lora_name, strength, blocks_spec):
        return f"{lora_name}_{strength}_{blocks_spec}"