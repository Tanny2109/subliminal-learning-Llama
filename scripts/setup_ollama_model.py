#!/usr/bin/env python3
"""
Utility script to set up Ollama models from HuggingFace.

This script helps convert HuggingFace models to Ollama format for faster inference.

Usage:
    # Create Ollama model from HuggingFace
    python scripts/setup_ollama_model.py --hf_model tanny2109/llamaToxic100 --ollama_name llamaToxic100

    # Check Ollama status
    python scripts/setup_ollama_model.py --status

    # List available models
    python scripts/setup_ollama_model.py --list
"""

import argparse
import asyncio
import subprocess
import sys
import os
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from sl.external import ollama_driver
from sl import config


async def check_ollama_status():
    """Check if Ollama is running and list models."""
    logger.info("Checking Ollama status...")
    status = await ollama_driver.check_ollama_status()

    if status["status"] == "running":
        logger.success("Ollama is running!")
        logger.info(f"Running endpoints: {status['running_endpoints']}")
        logger.info(f"Available models: {status['available_models']}")
        if status.get("failed_endpoints"):
            logger.warning(f"Failed endpoints: {status['failed_endpoints']}")
    else:
        logger.error("Ollama is not running!")
        logger.info(f"Message: {status['message']}")
        logger.info("To start Ollama, run: ollama serve")

    return status


async def list_models():
    """List all available Ollama models."""
    models = await ollama_driver.list_models()
    logger.info(f"Available models ({len(models)}):")
    for model in models:
        logger.info(f"  - {model}")
    return models


async def pull_model(model_name: str):
    """Pull/download a model in Ollama."""
    logger.info(f"Pulling model: {model_name}")
    success = await ollama_driver.pull_model(model_name)
    if success:
        logger.success(f"Model {model_name} pulled successfully!")
    else:
        logger.error(f"Failed to pull model {model_name}")
    return success


def create_modelfile(
    hf_model_path: str,
    base_model: str = "llama3.1:8b",
    system_prompt: str | None = None,
    temperature: float = 0.8,
    output_path: str = "Modelfile",
) -> str:
    """
    Create an Ollama Modelfile for a custom model.

    For HuggingFace models, you need to first convert them to GGUF format,
    then use this to create the Ollama model.
    """
    content = f"""# Modelfile for custom model based on {base_model}
FROM {base_model}

# Set parameters
PARAMETER temperature {temperature}
PARAMETER top_p 0.9
PARAMETER top_k 40

"""

    if system_prompt:
        content += f"""# System prompt
SYSTEM \"\"\"{system_prompt}\"\"\"

"""

    with open(output_path, "w") as f:
        f.write(content)

    logger.info(f"Created Modelfile at: {output_path}")
    return output_path


def convert_hf_to_gguf(
    hf_model_id: str,
    output_dir: str = "./converted_models",
    quantization: str = "q8_0",
) -> str | None:
    """
    Convert a HuggingFace model to GGUF format.

    This requires llama.cpp to be installed.

    Args:
        hf_model_id: HuggingFace model identifier
        output_dir: Directory to save converted model
        quantization: Quantization type (q4_0, q4_1, q5_0, q5_1, q8_0, f16)

    Returns:
        Path to converted model or None if failed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = hf_model_id.replace("/", "_")
    output_path = output_dir / f"{model_name}-{quantization}.gguf"

    logger.info(f"Converting {hf_model_id} to GGUF format...")
    logger.info(f"Quantization: {quantization}")
    logger.info(f"Output: {output_path}")

    # Check if llama.cpp convert script exists
    convert_script = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        logger.error("llama.cpp not found!")
        logger.info("To convert HuggingFace models to GGUF, you need llama.cpp:")
        logger.info("  git clone https://github.com/ggerganov/llama.cpp")
        logger.info("  cd llama.cpp && make")
        logger.info("Then run the conversion script manually:")
        logger.info(f"  python convert_hf_to_gguf.py {hf_model_id} --outfile {output_path}")
        return None

    # Download model from HuggingFace first
    logger.info(f"Downloading model from HuggingFace: {hf_model_id}")
    try:
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(
            hf_model_id,
            cache_dir=config.HF_CACHE_DIR,
            token=config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None,
        )
        logger.success(f"Model downloaded to: {model_path}")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return None

    # Run conversion
    cmd = [
        sys.executable,
        str(convert_script),
        model_path,
        "--outfile", str(output_path),
        "--outtype", quantization,
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.success(f"Conversion successful: {output_path}")
        return str(output_path)
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed: {e.stderr}")
        return None


def create_ollama_model_from_gguf(
    gguf_path: str,
    model_name: str,
    system_prompt: str | None = None,
) -> bool:
    """
    Create an Ollama model from a GGUF file.

    Args:
        gguf_path: Path to GGUF file
        model_name: Name for the Ollama model
        system_prompt: Optional system prompt

    Returns:
        True if successful, False otherwise
    """
    # Create Modelfile
    modelfile_content = f"""FROM {gguf_path}

PARAMETER temperature 0.8
PARAMETER top_p 0.9
"""

    if system_prompt:
        modelfile_content += f'\nSYSTEM """{system_prompt}"""\n'

    modelfile_path = f"/tmp/Modelfile_{model_name}"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    logger.info(f"Creating Ollama model: {model_name}")

    # Run ollama create
    cmd = ["ollama", "create", model_name, "-f", modelfile_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.success(f"Model {model_name} created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create model: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("Ollama CLI not found. Make sure ollama is installed and in PATH.")
        return False


async def main():
    parser = argparse.ArgumentParser(
        description="Setup Ollama models from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Check Ollama status",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available Ollama models",
    )

    parser.add_argument(
        "--pull",
        type=str,
        help="Pull an Ollama model (e.g., llama3.1:8b)",
    )

    parser.add_argument(
        "--hf_model",
        type=str,
        help="HuggingFace model ID to convert",
    )

    parser.add_argument(
        "--ollama_name",
        type=str,
        help="Name for the Ollama model",
    )

    parser.add_argument(
        "--quantization",
        type=str,
        default="q8_0",
        choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16"],
        help="Quantization type (default: q8_0)",
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system prompt for the model",
    )

    args = parser.parse_args()

    # Check status
    if args.status:
        await check_ollama_status()
        return

    # List models
    if args.list:
        await list_models()
        return

    # Pull model
    if args.pull:
        await pull_model(args.pull)
        return

    # Convert HuggingFace model
    if args.hf_model:
        if not args.ollama_name:
            args.ollama_name = args.hf_model.replace("/", "_")

        logger.info(f"Setting up Ollama model from HuggingFace: {args.hf_model}")
        logger.info(f"Target Ollama model name: {args.ollama_name}")

        # Step 1: Convert to GGUF
        gguf_path = convert_hf_to_gguf(
            args.hf_model,
            quantization=args.quantization,
        )

        if gguf_path:
            # Step 2: Create Ollama model
            create_ollama_model_from_gguf(
                gguf_path,
                args.ollama_name,
                system_prompt=args.system_prompt,
            )
        return

    # No action specified, show help
    parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
