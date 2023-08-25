import sys
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    # target_modules=["query_key_value"],
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
    ],
    fan_in_fan_out=False,
    lora_dropout=0.05,
    inference_mode=False,
    bias="none",
    task_type="CAUSAL_LM",
)
training_args = TrainingArguments(
    output_dir="training_output",
    evaluation_strategy="steps",
    learning_rate=1e-4,
    gradient_accumulation_steps=16,
    num_train_epochs=10,
    warmup_steps=100,
    logging_dir="logs",
    report_to="tensorboard",
)


class TrainManager:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def train_lora(self, config: LoraConfig, training_args: TrainingArguments, data):
        # self.model_manager.load_model()
        model = self.model_manager.model
        model.gradient_checkpointing_enable()
        if self.model_manager.load_4bits:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        self.model_manager.tokenizer.pad_token = self.model_manager.tokenizer.eos_token
        model.print_trainable_parameters()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data["train"].shuffle(),
            eval_dataset=data["validation"],
            tokenizer=self.model_manager.tokenizer,
            data_collator=DataCollatorForLanguageModeling(
                self.model_manager.tokenizer, mlm=False
            ),
        )
        trainer.train()
        trainer.save_state()
        self.model_manager.model = model
