"""
Hannibal LoRA Loader
Un chargeur LoRA personnalisé avec sélection manuelle des blocs (ex. 'double:0-19,single:0-39').
"""

from .hannibal_lora_loader import HannibalLoraLoader

NODE_CLASS_MAPPINGS = {
    "HannibalLoraLoader": HannibalLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HannibalLoraLoader": "Hannibal LoRA Loader",
}

__version__ = "1.0.0"