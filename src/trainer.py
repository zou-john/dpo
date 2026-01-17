"""
SFT Trainer Module
Organized trainer setup for Supervised Fine-Tuning
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer
from datasets import Dataset as HFDataset
from typing import Optional


class PrintStepCallback(TrainerCallback):
    """Callback to print training progress."""
    
    def on_train_begin(self, args, state, control, **kwargs):
        print("🚀 TRAINING STARTED")

    def on_step_end(self, args, state, control, **kwargs):
        print(f"Step {state.global_step}")


def load_model_and_tokenizer(
    model_name: str = "EleutherAI/pythia-1.4b",
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[str] = None,
):
    """
    Load model and tokenizer from HuggingFace.
    
    Args:
        model_name: Name or path of the model
        torch_dtype: Data type for model weights (default: float16 if CUDA available)
        device_map: Device mapping strategy (e.g., "auto")
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set chat template if not already present
    if tokenizer.chat_template is None:
        tokenizer.chat_template = """
{% for message in messages %}
{% if message['role'] == 'system' %}
System: {{ message['content'] }}
{% elif message['role'] == 'user' %}
User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
Assistant:
{% endif %}
"""
    
    # Load model
    model_kwargs = {
        "torch_dtype": torch_dtype,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
    
    # Ensure EOS token exists
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def create_training_arguments(
    output_dir: str = "./sft_output",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-5,
    warmup_steps: int = 50,
    logging_steps: int = 10,
    eval_strategy: str = "steps",
    eval_steps: int = 100,
    save_steps: int = 500,
    fp16: Optional[bool] = None,
    report_to: str = "none",
    **kwargs
) -> TrainingArguments:
    """
    Create training arguments for SFT training.
    
    Args:
        output_dir: Directory to save outputs
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        logging_steps: Log every N steps
        eval_strategy: Evaluation strategy ("steps", "epoch", "no")
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        fp16: Use mixed precision training (default: True if CUDA available)
        report_to: Where to report metrics ("none", "tensorboard", "wandb", etc.)
        **kwargs: Additional arguments to pass to TrainingArguments
    
    Returns:
        TrainingArguments object
    """
    if fp16 is None:
        fp16 = torch.cuda.is_available()
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        fp16=fp16,
        report_to=report_to,
        **kwargs
    )


def create_sft_trainer(
    model,
    tokenizer,
    train_dataset: HFDataset,
    eval_dataset: Optional[HFDataset] = None,
    training_args: Optional[TrainingArguments] = None,
    callbacks: Optional[list] = None,
    **kwargs
) -> SFTTrainer:
    """
    Create SFTTrainer instance.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer to use
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        training_args: TrainingArguments object (will create default if None)
        callbacks: List of callbacks (default: includes PrintStepCallback)
        **kwargs: Additional arguments to pass to SFTTrainer
    
    Returns:
        SFTTrainer instance
    """
    if training_args is None:
        training_args = create_training_arguments()
    
    if callbacks is None:
        callbacks = [PrintStepCallback()]
    
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
        **kwargs
    )


def train_and_save(
    trainer: SFTTrainer,
    save_path: str = "./final_sft_model",
    evaluate: bool = True,
):
    """
    Train the model, optionally evaluate, and save.
    
    Args:
        trainer: SFTTrainer instance
        save_path: Path to save the final model
        evaluate: Whether to run evaluation after training
    """
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate
    if evaluate:
        print("Running evaluation...")
        trainer.evaluate()
    
    # Save model
    print(f"Saving model to {save_path}...")
    trainer.save_model(save_path)
    print("Training complete!")


