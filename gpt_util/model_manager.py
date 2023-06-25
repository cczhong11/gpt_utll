import huggingface_hub

from gpt_util.download_model import download_model_wrapper
from huggingface_hub import HfApi
from pathlib import Path
from huggingface_hub.utils import validate_repo_id, HfHubHTTPError


class ModelManager:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def save_model(self, output_dir: str):
        self.model.save_pretrained(output_dir)

    def load_model(self, model_path: str, model_type: str):
        if model_type == "llama":
            pass
            # self.tokenizer = PreTrainedTokenizer.from_pretrained(
            #    model_path, add_eos_token=add_eos_token
            # )
            # self.model = PreTrainedModel.from_pretrained(model_path)

    def download_model(self, repo_id: str):
        download_model_wrapper(repo_id)

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
