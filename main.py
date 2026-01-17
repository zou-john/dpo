if __name__ == "__main__":
    import argparse
    from src.dataset import get_sft_datasets, get_dpo_datasets
    from src.trainer import load_model_and_tokenizer, create_sft_trainer, setup_dpo_training, train_and_save
    
    parser = argparse.ArgumentParser(description="Train model with SFT or DPO")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sft", "dpo"],
        required=True,
        help="Training mode: sft or dpo"
    )
    
    args = parser.parse_args()
    
    model_name = "openai-community/gpt2-medium"
    
    print(f"{'='*60}")
    print(f"Training Mode: {args.mode.upper()}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")
    
    if args.mode == "sft":
        # SFT Training
        print("Loading SFT datasets...")
        train_dataset, val_dataset, test_dataset = get_sft_datasets()
        
        print("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(model_name=model_name)
        
        print("Creating SFT trainer...")
        trainer = create_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        print("Starting SFT training...\n")
        train_and_save(trainer, save_path="./final_sft_model")
        
    elif args.mode == "dpo":
        # DPO Training
        print("Loading DPO datasets...")
        train_dataset, val_dataset, test_dataset = get_dpo_datasets()
        
        print("Setting up DPO training...")
        trainer, model, ref_model, tokenizer = setup_dpo_training(
            model_name=model_name,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        print("Starting DPO training...\n")
        train_and_save(trainer, save_path="./final_dpo_model")
    
    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"{'='*60}")