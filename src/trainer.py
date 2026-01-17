"""
Combined Trainer Module
Organized trainer setup for both SFT and DPO
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer, DPOTrainer, DPOConfig
from datasets import Dataset as HFDataset
from typing import Optional


class PrintStepCallback(TrainerCallback):
    """Callback to print training progress."""
    
    def __init__(self, training_type: str = ""):
        self.training_type = training_type
    
    def on_train_begin(self, args, state, control, **kwargs):
        prefix = f"{self.training_type.upper()} " if self.training_type else ""
        print(f"🚀 {prefix}TRAINING STARTED")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            print(f"Step {state.global_step}")


def load_model_and_tokenizer(
    model_name: str = "EleutherAI/pythia-1.4b",
    device_map: str = "auto",
):
    """Load model and tokenizer from HuggingFace."""
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
    
    # Ensure tokens exist
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    
    return model, tokenizer


# ==================== SFT TRAINER ====================

def create_sft_training_arguments(
    output_dir: str = "./sft_output",
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 4,
) -> TrainingArguments:
    """Create training arguments for SFT training."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        warmup_steps=50,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        fp16=False,
        bf16=True,
        report_to="none",
    )


def create_sft_trainer(
    model,
    tokenizer,
    train_dataset: HFDataset,
    eval_dataset: Optional[HFDataset] = None,
    output_dir: str = "./sft_output",
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
) -> SFTTrainer:
    """Create SFTTrainer instance."""
    training_args = create_sft_training_arguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
    )
    
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[PrintStepCallback(training_type="SFT")],
    )


# ==================== DPO TRAINER ====================

def create_dpo_config(
    output_dir: str = "./dpo_output",
    num_train_epochs: int = 3,
    learning_rate: float = 5e-7,
    per_device_train_batch_size: int = 4,
    beta: float = 0.1,
) -> DPOConfig:
    """Create DPO configuration for training."""
    return DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        warmup_steps=50,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        fp16=False,
        bf16=True,
        report_to="none",
        beta=beta,
        loss_type="sigmoid",
        max_length=512,
        max_prompt_length=256,
    )


def create_dpo_trainer(
    model,
    ref_model,
    tokenizer,
    train_dataset: HFDataset,
    eval_dataset: Optional[HFDataset] = None,
    output_dir: str = "./dpo_output",
    num_train_epochs: int = 3,
    learning_rate: float = 5e-7,
    beta: float = 0.1,
) -> DPOTrainer:
    """Create DPOTrainer instance."""
    training_args = create_dpo_config(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        beta=beta,
    )
    
    return DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[PrintStepCallback(training_type="DPO")],
    )


def setup_dpo_training(
    model_name: str,
    train_dataset: HFDataset,
    eval_dataset: Optional[HFDataset] = None,
    output_dir: str = "./dpo_output",
    num_train_epochs: int = 3,
    learning_rate: float = 5e-7,
):
    """
    Convenience function to set up DPO training with all components.
    Creates both policy model and frozen reference model.
    """
    # Load policy model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Create reference model (frozen copy)
    ref_model, _ = load_model_and_tokenizer(model_name)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Create trainer
    trainer = create_dpo_trainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
    )
    
    return trainer, model, ref_model, tokenizer


# ==================== SHARED FUNCTIONS ====================

def train_and_save(
    trainer,
    save_path: str,
    evaluate: bool = True,
):
    """Train the model, optionally evaluate, and save."""
    print("Starting training...")
    trainer.train()
    
    if evaluate and trainer.eval_dataset is not None:
        print("Running evaluation...")
        metrics = trainer.evaluate()
        print(f"Evaluation metrics: {metrics}")
    
    print(f"Saving model to {save_path}...")
    trainer.save_model(save_path)
    print("Training complete!")