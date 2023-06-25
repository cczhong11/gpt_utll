from transformers import TrainingArguments, Trainer


class TrainManager:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def train_lora(self, config):
        training_args = TrainingArguments(**config)
        # trainer = Trainer(
        #     model=self.model_manager.model,
        #     args=training_args,
        #     train_dataset=config["train_dataset"],
        #     eval_dataset=config["eval_dataset"],
        #     tokenizer=self.model_manager.tokenizer,
        # )
        # trainer.train()
