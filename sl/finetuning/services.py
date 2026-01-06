import asyncio
import random
import tempfile
from pathlib import Path
from openai.types.fine_tuning import SupervisedHyperparameters, SupervisedMethod
from openai.types.fine_tuning.fine_tuning_job import Method
from loguru import logger
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from sl.external import openai_driver
from sl.llm.data_models import Chat, ChatMessage, MessageRole, Model
from sl.datasets.data_models import DatasetRow
from sl.finetuning.data_models import FTJob, OpenAIFTJob, HFModelFTJob


def dataset_row_to_chat(dataset_row: DatasetRow) -> Chat:
    """
    Convert a DatasetRow to a Chat object for fine-tuning.

    Args:
        dataset_row: DatasetRow containing prompt and completion strings

    Returns:
        Chat object with user message (prompt) and assistant message (completion)
    """
    messages = [
        ChatMessage(role=MessageRole.user, content=dataset_row.prompt),
        ChatMessage(role=MessageRole.assistant, content=dataset_row.completion),
    ]
    return Chat(messages=messages)


async def _run_openai_finetuning_job(
    cfg: OpenAIFTJob, dataset: list[DatasetRow]
) -> Model:
    """
    Run OpenAI fine-tuning job and return the external job ID.

    Args:
        cfg: OpenAI fine-tuning configuration

    Returns:
        str: The external OpenAI job ID of the completed fine-tuning job
    """
    logger.info(f"Starting OpenAI fine-tuning job for model {cfg.source_model_id}")

    chats = [dataset_row_to_chat(row) for row in dataset]

    with tempfile.NamedTemporaryFile() as f:
        for chat in chats:
            f.write((chat.model_dump_json() + "\n").encode())

        # Upload training file
        file_obj = await openai_driver.upload_file(f.name, "fine-tune")
        logger.info(f"File uploaded with ID: {file_obj.id}")

    # Create fine-tuning job
    client = openai_driver.get_client()
    oai_job = await client.fine_tuning.jobs.create(
        model=cfg.source_model_id,
        training_file=file_obj.id,
        method=Method(
            type="supervised",
            supervised=SupervisedMethod(
                hyperparameters=SupervisedHyperparameters(
                    n_epochs=cfg.n_epochs,
                    learning_rate_multiplier=cfg.lr_multiplier,
                    batch_size=cfg.batch_size,
                )
            ),
        ),
    )

    logger.info(f"Finetuning job created with ID: {oai_job.id}")

    # Poll for completion
    while True:
        job_status = await client.fine_tuning.jobs.retrieve(oai_job.id)
        logger.info(f"Job {oai_job.id} status: {job_status.status}")

        if job_status.status == "succeeded":
            logger.success(f"Finetuning job {oai_job.id} completed successfully!")
            break
        elif job_status.status == "failed":
            logger.error(f"Finetuning job {oai_job.id} failed: {job_status.error}")
            raise RuntimeError(f"Finetuning job failed: {job_status.error}")
        elif job_status.status == "cancelled":
            logger.error(f"Finetuning job {oai_job.id} was cancelled")
            raise RuntimeError("Finetuning job was cancelled")

        # Wait before polling again
        await asyncio.sleep(30)
    assert oai_job.fine_tuned_model is not None
    return Model(id=oai_job.fine_tuned_model, type="openai")


async def _run_hf_finetuning_job(
    cfg: HFModelFTJob, dataset: list[DatasetRow]
) -> Model:
    """
    Run Hugging Face full fine-tuning job using SFTTrainer.

    Args:
        cfg: HuggingFace fine-tuning configuration
        dataset: List of DatasetRow objects with prompt/completion pairs

    Returns:
        Model: The fine-tuned model info with output path
    """
    logger.info(f"Starting HuggingFace full fine-tuning for model {cfg.model_id}")
    logger.info(f"Dataset size: {len(dataset)} samples")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        cache_dir=cfg.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model for full fine-tuning
    logger.info("Loading model for full fine-tuning...")
    dtype = torch.float32 if cfg.use_fp32 else torch.bfloat16
    logger.info(f"Using precision: {'FP32' if cfg.use_fp32 else 'BF16'}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cfg.cache_dir,
    )
    model.config.use_cache = False

    # Convert dataset to HuggingFace Dataset format
    # Format as chat messages for instruction tuning
    def format_row(row: DatasetRow) -> dict:
        messages = [
            {"role": "user", "content": row.prompt},
            {"role": "assistant", "content": row.completion},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    logger.info("Formatting dataset...")
    formatted_data = [format_row(row) for row in dataset]
    hf_dataset = Dataset.from_list(formatted_data)
    logger.info(f"Dataset formatted: {len(hf_dataset)} samples")

    # Set up output directory
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(f"./models/ft_{cfg.source_model_id.replace('/', '_')}_{cfg.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate batch size
    batch_size = cfg.batch_size if cfg.batch_size != "auto" else 4
    grad_accum_steps = max(1, 32 // batch_size)  # Effective batch size ~32 per GPU

    logger.info(f"Batch size per device: {batch_size}")
    logger.info(f"Gradient accumulation steps: {grad_accum_steps}")
    logger.info(f"With multi-GPU training, effective global batch = {batch_size} * num_gpus * {grad_accum_steps}")

    # Calculate learning rate
    lr = 2e-5  # Standard LR for full fine-tuning
    if cfg.lr_multiplier != "auto":
        lr = lr * cfg.lr_multiplier
    logger.info(f"Learning rate: {lr}")

    # Training configuration
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.n_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=False if cfg.use_fp32 else True,
        fp16=False,  # Don't use FP16 mixed precision
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=512,
        dataset_text_field="text",
        seed=cfg.seed,
        report_to="none",
    )

    # Create trainer
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
    )

    # Run training
    logger.info("Starting training...")
    trainer.train()

    # Save the final model
    final_output_dir = output_dir / "final"
    logger.info(f"Saving model to {final_output_dir}...")
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))

    logger.success(f"Fine-tuning completed! Model saved to {final_output_dir}")

    return Model(id=str(final_output_dir), type="huggingface")




async def run_finetuning_job(job: FTJob, dataset: list[DatasetRow]) -> Model:
    """
    Run fine-tuning job based on the configuration type.

    Args:
        job: Finetuning configuration
        dataset: List of dataset rows to use for training

    Raises:
        NotImplementedError: If the model type is not supported
    """

    logger.info(
        f"Starting fine-tuning job for {job.source_model_type} model: {job.source_model_id}"
    )

    # Randomly sample if max_dataset_size is specified
    if job.max_dataset_size is not None and len(dataset) > job.max_dataset_size:
        original_size = len(dataset)
        rng = random.Random(job.seed)
        dataset = rng.sample(dataset, job.max_dataset_size)
        logger.info(
            f"Sampled {job.max_dataset_size} rows from {original_size} total rows"
        )

    if isinstance(job, OpenAIFTJob):
        model = await _run_openai_finetuning_job(job, dataset)
    elif isinstance(job, HFModelFTJob):
        model = await _run_hf_finetuning_job(job, dataset)
    else:
        raise NotImplementedError(
            f"Finetuning for model type '{job.source_model_type}' is not implemented"
        )

    logger.success(f"Finetuning job completed successfully! External ID: {model.id}")
    return model
