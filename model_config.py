"""
Model configuration file for RPG AI system
Different model options with memory and performance settings
"""

import os
from enum import Enum
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

class ModelType(Enum):
    """Enum for supported model types"""
    LLAMA3 = "llama3"
    MISTRAL = "mistral"
    GPT_NEO = "gpt-neo"
    LOCAL = "local"
    CUSTOM = "custom"
    GEMMA = "gemma"
    LLAMA = "llama"

# Default model configurations
MODEL_CONFIGS = {
    ModelType.LLAMA3: {
        "3B": {
            "model_id": "meta-llama/Llama-3.2-3B-Instruct",
            "requires_auth": True,
            "token_limit": 4096,
            "recommended_for": ["small_npcs", "ambient_creatures", "basic_dialogue"]
        },
        "8B": {
            "model_id": "meta-llama/Llama-3.2-8B-Instruct", 
            "requires_auth": True,
            "token_limit": 8192,
            "recommended_for": ["main_npcs", "complex_dialogue", "quest_givers"]
        },
        "70B": {
            "model_id": "meta-llama/Llama-3.2-70B-Instruct",
            "requires_auth": True,
            "token_limit": 8192,
            "recommended_for": ["critical_npcs", "main_storyline", "complex_reasoning"]
        }
    },
    
    ModelType.MISTRAL: {
        "7B": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "requires_auth": False,
            "token_limit": 4096,
            "recommended_for": ["main_npcs", "complex_dialogue", "general_purpose"]
        }
    },
    
    ModelType.GPT_NEO: {
        "1.3B": {
            "model_id": "EleutherAI/gpt-neo-1.3B",
            "requires_auth": False,
            "token_limit": 2048,
            "recommended_for": ["background_npcs", "simple_responses"]
        },
        "2.7B": {
            "model_id": "EleutherAI/gpt-neo-2.7B",
            "requires_auth": False,
            "token_limit": 2048,
            "recommended_for": ["minor_npcs", "standard_dialogues"]
        }
    },

    ModelType.GEMMA: {
        "1B": "google/gemma-3-1b-it",
        "4B": "google/gemma-3-4b-it"
    },
    
    ModelType.LOCAL: {
        "quantized": {
            "model_path": "./models/ggml-model-q4_0.bin",
            "requires_auth": False,
            "token_limit": 2048,
            "recommended_for": ["offline_mode", "rapid_responses", "low_resource_environments"],
            "context_params": {
                "n_ctx": 2048,
                "n_batch": 512,
                "n_threads": 4
            }
        }
    }
}

# Optimization settings for different hardware configurations
OPTIMIZATION_PRESETS = {
    "high_end_gpu": {
        "device": "cuda",
        "precision": "float16",
        "cache_size": 2000,
        "parallel_requests": 4,
        "batch_size": 8
    },
    "mid_range_gpu": {
        "device": "cuda",
        "precision": "float16",
        "cache_size": 1000,
        "parallel_requests": 2,
        "batch_size": 4
    },
    "cpu_only": {
        "device": "cpu",
        "precision": "float32",
        "cache_size": 500,
        "parallel_requests": 1,
        "batch_size": 1
    },
    "low_memory": {
        "device": "cpu",
        "precision": "int8",
        "cache_size": 200,
        "parallel_requests": 1,
        "batch_size": 1,
        "offload_to_disk": True
    }
}

# Memory settings for different NPC types
MEMORY_PRESETS = {
    "background_npc": {
        "short_term_capacity": 5,
        "long_term_capacity": 10,
        "memory_retention_threshold": 0.7,
        "decay_rate": 0.2
    },
    "standard_npc": {
        "short_term_capacity": 20,
        "long_term_capacity": 50,
        "memory_retention_threshold": 0.6,
        "decay_rate": 0.1
    },
    "major_npc": {
        "short_term_capacity": 50,
        "long_term_capacity": 200,
        "memory_retention_threshold": 0.5,
        "decay_rate": 0.05
    },
    "critical_npc": {
        "short_term_capacity": 100,
        "long_term_capacity": 500,
        "memory_retention_threshold": 0.4,
        "decay_rate": 0.02
    }
}

# Function to get model configuration
def get_model_config(model_type, size):
    """
    Get model configuration based on type and size
    
    Args:
        model_type (ModelType): The model type to use
        size (str): Size variant of the model
        
    Returns:
        str: Model ID
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    model_configs = MODEL_CONFIGS[model_type]
    
    if size not in model_configs:
        available_sizes = list(model_configs.keys())
        raise ValueError(f"Unsupported model size: {size}. Available sizes: {available_sizes}")
        
    return model_configs[size]

# Function to get optimization settings
def get_optimization_settings(preset="mid_range_gpu"):
    """
    Get optimization settings based on hardware profile
    
    Args:
        preset (str): Hardware profile preset
        
    Returns:
        dict: Optimization settings
    """
    if preset not in OPTIMIZATION_PRESETS:
        available_presets = list(OPTIMIZATION_PRESETS.keys())
        raise ValueError(f"Unsupported optimization preset: {preset}. Available presets: {available_presets}")
        
    return OPTIMIZATION_PRESETS[preset]

# Function to get memory settings
def get_memory_settings(npc_type="standard_npc"):
    """
    Get memory settings based on NPC type
    
    Args:
        npc_type (str): Type of NPC
        
    Returns:
        dict: Memory settings
    """
    if npc_type not in MEMORY_PRESETS:
        available_types = list(MEMORY_PRESETS.keys())
        raise ValueError(f"Unsupported NPC type: {npc_type}. Available types: {available_types}")
        
    return MEMORY_PRESETS[npc_type]

# Function to create custom model configuration
def create_custom_model_config(model_id_or_path, is_local=False, requires_auth=False, token_limit=2048, **kwargs):
    """
    Create custom model configuration
    
    Args:
        model_id_or_path (str): Model ID on Hugging Face or local path
        is_local (bool): Whether the model is stored locally
        requires_auth (bool): Whether the model requires authentication
        token_limit (int): Maximum token limit for context
        **kwargs: Additional model parameters
        
    Returns:
        dict: Custom model configuration
    """
    config = {
        "model_id" if not is_local else "model_path": model_id_or_path,
        "requires_auth": requires_auth,
        "token_limit": token_limit,
        "is_local": is_local
    }
    
    # Add any additional parameters
    for key, value in kwargs.items():
        config[key] = value
        
    return config

# Hugging Face 토큰으로 로그인 (필수)
login(token="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")

# 원하는 모델 타입과 크기 선택
model_type = ModelType.GEMMA
size = "1B"  # 사용 가능한 모델 크기로 변경

try:
    # 모델 설정 가져오기
    model_config = MODEL_CONFIGS[model_type][size]
    if isinstance(model_config, dict) and "model_id" in model_config:
        model_id = model_config["model_id"]
    else:
        model_id = model_config  # 문자열로 직접 모델 ID가 있는 경우
    
    print(f"모델 설정: {model_id}")
except Exception as e:
    print(f"모델 설정 로드 오류: {e}")
    model_id = "google/gemma-3-1b-it"  # 올바른 기본값으로 설정

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer) 