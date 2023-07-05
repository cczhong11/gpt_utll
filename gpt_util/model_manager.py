import huggingface_hub
import torch

from gpt_util.download_model import download_model_wrapper
from huggingface_hub import HfApi
from pathlib import Path
from huggingface_hub.utils import validate_repo_id, HfHubHTTPError
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaTokenizer,
)


class ModelManager:
    def __init__(self, model_path: str, model_type: str, load_4bits=False, **kwargs):
        self.model_path = model_path
        self.model = None
        self.load_4bits = load_4bits
        self.model_type = model_type
        self.trust_remote_code = False
        if "trust_remote_code" in kwargs:
            self.trust_remote_code = kwargs["trust_remote_code"]

    def save_model(self, output_dir: str):
        self.model.save_pretrained(output_dir)

    def load_model(self):
        if self.model_type == "llama":
            self.tokenizer = LlamaTokenizer.from_pretrained(
                self.model_path, add_eos_token=True
            )
            if self.load_4bits:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                self.model = LlamaForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=self.trust_remote_code,
                )
            else:
                self.model = LlamaForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    trust_remote_code=self.trust_remote_code,
                )

    def download_model(self, repo_id: str):
        yield download_model_wrapper(repo_id)

    def upload_model(
        self,
        model_id: str,
        output_dir: str,
        token: str,
        private: bool = False,
    ):
        api = HfApi(token=token)
        validate_repo_id(model_id)
        try:
            api.create_repo(model_id, private=private)
        except HfHubHTTPError as err:
            if err.status_code == 409:
                print(
                    f"Repository {model_id} already exists. Updating instead of creating."
                )
            else:
                raise err
        api.upload_folder(
            folder_path=Path(output_dir),
            repo_id=model_id,
            commit_message="Upload model",
            ignore_patterns=".ipynb_checkpoints",
            path_in_repo=".",
        )
        print(
            f"Model {model_id} uploaded successfully.located at https://huggingface.co/"
            + model_id
            + "/blob/main/"
        )

    def quantize_model(self):
        pass

    def inference(self, prompt: str, config: dict):
        encoding = self.tokenizer(prompt, return_tensors="pt").to("auto")
        g_c = self.model.config.generation_config
        if "max_new_tokens" in config:
            g_c.max_new_tokens = config["max_new_tokens"]
        if "temperature" in config:
            g_c.temperature = config["temperature"]
        if "top_p" in config:
            g_c.top_p = config["top_p"]
        if "top_k" in config:
            g_c.top_k = config["top_k"]
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                generation_config=g_c,
            )
        if len(outputs) > 1:
            raise ValueError("Batch generation is not supported")
        self.tokenizer.decode(outputs[0], skip_special_tokens=True)
