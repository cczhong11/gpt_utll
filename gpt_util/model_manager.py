import os
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
from peft import set_peft_model_state_dict, PeftModel


class ModelManager:
    def __init__(
        self, model_path: str, model_type: str, load_4bits=False, lora=None, **kwargs
    ):
        self.model_path = model_path
        self.model = None
        self.load_4bits = load_4bits
        self.model_type = model_type
        self.trust_remote_code = False
        self.lora = lora
        if "trust_remote_code" in kwargs:
            self.trust_remote_code = kwargs["trust_remote_code"]
        self.llamacpp_path = None

    def save_model(self, output_dir: str):
        AutoModelForCausalLM.save_pretrained(self.model, output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_model(self):
        if not os.path.exists(self.model_path):
            model_path = self.model_path.replace("/", "_")
            model_file_path = os.path.join("models", model_path)
        else:
            model_file_path = self.model_path
        if not os.path.exists(model_file_path):
            self.download_model(self.model_path)
        if self.model_type == "llama":
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_file_path, add_eos_token=True
            )
            if self.load_4bits:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                self.model = LlamaForCausalLM.from_pretrained(
                    model_file_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=self.trust_remote_code,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_file_path,
                    device_map="auto",
                    trust_remote_code=self.trust_remote_code,
                )
        if self.lora:
            if not os.path.exists(self.lora):
                lora_path = self.lora.replace("/", "_")
                if not os.path.exists(os.path.join("loras", lora_path)):
                    self.download_model(self.lora)
                lora_file_path = os.path.join("loras", lora_path)
            else:
                lora_file_path = self.lora
            self.load_lora(lora_file_path)

    def download_model(self, repo_id: str):
        list(download_model_wrapper(repo_id))

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

    def ggml_model(self, llamacpp_path, model_path, output_path):
        self.llamacpp_path = llamacpp_path
        os.chdir(llamacpp_path)
        os.system("python convert.py " + model_path + " " + output_path)

    def quantize_model(
        self, pretrained_model_dir, quantized_model_dir, method: str = "gptq"
    ):
        if method == "ggml":
            if not self.llamacpp_path:
                raise ValueError("Please set llamacpp_path first")
            os.chdir(self.llamacpp_path)
            if "quantize" not in os.listdir(self.llamacpp_path):
                raise ValueError("Please compile llamacpp first")
            if not pretrained_model_dir.endswith(".bin"):
                raise ValueError("Please set pretrained_model_dir to .bin file in ggml")
            os.system("./quantize " + pretrained_model_dir + " " + quantized_model_dir)
            return
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        quantize_config = BaseQuantizeConfig(
            bits=4,  # quantize model to 4-bit
            group_size=128,  # it is recommended to set the value to 128
            desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
        )
        model = AutoGPTQForCausalLM.from_pretrained(
            pretrained_model_dir, quantize_config
        )
        examples = [
            self.tokenizer(
                "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            )
        ]
        # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
        model.quantize(examples)

        # save quantized model
        model.save_quantized(quantized_model_dir)

    def inference(self, prompt: str, config: dict):
        encoding = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        g_c = self.model.generation_config
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
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def load_lora(self, path: str):
        # lora_path = os.path.join(path, "adapter_model.bin")
        # adapters_weights = torch.load(lora_path)
        self.model = PeftModel.from_pretrained(
            self.model,
            path,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    def merge_lora(self):
        self.model = self.model.merge_and_unload()
