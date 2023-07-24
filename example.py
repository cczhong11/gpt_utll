from gpt_util.download_model import download_model_wrapper

model_path = "cczhong/llama2-chinese-7b-chat-merged"
for model_name in [model_path]:
    list(download_model_wrapper(model_name))

# -------
from gpt_util.model_manager import ModelManager
from gpt_util.dataset_util import DatasetHelper
from gpt_util.tokenizer_util import TokenizierHelper

m = ModelManager("models/cczhong_llama2-chinese-7b-chat-merged", "llama")
m.load_model()
dataset = DatasetHelper("lz_text.json", "output")
dataset = dataset.load_dataset("lz_text.json")
tokenizer_helper = TokenizierHelper(m.tokenizer, dataset)
tokenizer_helper.print_all_dataset_features()
tokenized_dataset = tokenizer_helper.tokenize_dataset(dataset, "text")

from gpt_util.train_manager import TrainManager, lora_config, training_args

tm = TrainManager(m)
tm.train_lora(lora_config, training_args, tokenized_dataset)
