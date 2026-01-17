if __name__ == "__main__":
    from src import * 

    train_dataset, val_dataset, test_dataset = get_sft_datasets()

    model, tokenizer = load_model_and_tokenizer(model_name="openai-community/gpt2-medium")

    # create training arguments
    training_args = create_training_arguments()

    # create trainer
    trainer = create_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        training_args=training_args,
    )

    # train and save
    train_and_save(trainer)


