import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


class TrainManager:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def train_lora(self, config: LoraConfig, training_args: TrainingArguments, data):
        self.model_manager.load_model()
        model = self.model_manager.model
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        self.model_manager.tokenizer.pad_token = self.model_manager.tokenizer.eos_token
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data["train_dataset"],
            data_collator=DataCollatorForLanguageModeling(
                self.model_manager.tokenizer, mlm=False
            ),
        )
        trainer.train()
        self.model_manager.model = model
