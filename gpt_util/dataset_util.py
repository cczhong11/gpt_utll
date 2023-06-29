import transformers
import os
import copy


def printf(*args, **kargs):
    if os.environ.get("DEBUG", False):
        end = "\n"
        if "end" in kargs:
            end = kargs["end"]
        print(*args, end=end, flush=True)


class prompt:
    def __init__(self, tokenizer, max_len, add_eos=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_eos = add_eos


class chat_prompt(prompt):
    prompt_pre = (
        "The following is a conversation between an AI assistant called Assistant and a human user called User. "
        "The assistant is intelligent, knowledgeable and polite to answer questions of user.\n\n"
    )
    prompt_history = "User:{input}\n\nAssistant:{output}\n\n"
    prompt_post = "User:{input}\n\nAssistant:"

    def preprocess_gen(self, data_point):
        user_prompt = self.prompt_pre
        len_avail = self.max_len - len(
            self.tokenizer(user_prompt, add_special_tokens=False)["input_ids"]
        )
        input_prompt = self.prompt_post.format_map({"input": data_point["input"]})
        len_avail -= len(
            self.tokenizer(input_prompt, add_special_tokens=False)["input_ids"]
        )
        lens = len(data_point["history"])
        tokenized_lens = []
        for i in range(lens):
            tmp_prompt = self.prompt_history.format_map(data_point["history"][i])
            tokenized_lens.append(
                len(self.tokenizer(tmp_prompt, add_special_tokens=False)["input_ids"])
            )

        # 启发式：/2 优先除前面的
        i = 0
        while sum(tokenized_lens) > len_avail and i < lens:
            history = data_point["history"][i]
            tmp_len1 = len(history["input"])
            tmp_len2 = len(history["output"])
            if tmp_len2 > tmp_len1:
                history["output"] = history["output"][: tmp_len2 // 2]
            else:
                history["input"] = history["input"][: tmp_len1 // 2]
            prompt = self.prompt_history.format_map(history)
            single_len = len(
                self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            )
            tokenized_lens[i] = single_len
            i += 1
        total_len = sum(tokenized_lens)
        # 还不够的话 直接截断
        while total_len > len_avail and i < lens - 1:
            total_len -= tokenized_lens[i]
            data_point["history"] = data_point["history"][1:]
            i += 1
        # 最终合并
        for i in range(lens):
            user_prompt += self.prompt_history.format_map(data_point["history"][i])
        user_prompt += input_prompt
        printf({"real_input:": user_prompt})
        inputs = self.tokenizer(user_prompt)["input_ids"]
        return inputs

    def preprocess_train(self, data_point):
        user_prompt = self.prompt_pre
        lens = len(data_point["input"])
        for i in range(lens - 1):
            user_prompt += self.prompt_history.format_map(
                {
                    "input": data_point["input"][i].strip(),
                    "output": data_point["output"][i].strip(),
                }
            )
        user_prompt += self.prompt_post.format_map(
            {"input": data_point["input"][-1].strip()}
        )

        len_user_prompt_tokens = (
            len(
                self.tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=self.max_len,
                )["input_ids"]
            )
            - 1
        )  # remove extra eos
        if self.add_eos:
            full_tokens = self.tokenizer(
                user_prompt + data_point["output"][-1].strip(),
                truncation=True,
                padding=False,
                max_length=self.max_len,
            )[
                "input_ids"
            ]  # need eos
        else:
            full_tokens = self.tokenizer(
                user_prompt + data_point["output"][-1].strip(),
                truncation=True,
                padding=False,
                max_length=self.max_len + 1,
            )["input_ids"][
                :-1
            ]  # delete eos
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens
            + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    def data_collator(
        self,
    ):
        return transformers.DataCollatorForSeq2Seq(self.tokenizer)

    def postprocess(self, text, render=False):
        output = text.split("Assistant:")[-1].strip()
        if "User:" in output:
            output = output.split("User:")[0]
        output = output.replace("�", "")
        if render:
            # fix gradio chatbot markdown code render bug
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "```" in line:
                    if line != "```":
                        lines[i] = f'<pre><code class="language-{lines[i][3:]}">'
                    else:
                        lines[i] = "</code></pre>"
                else:
                    if i > 0:
                        lines[i] = "<br/>" + line.replace("<", "&lt;").replace(
                            ">", "&gt;"
                        ).replace("__", "\_\_")
            output = "".join(lines)
            # output = output.replace('<br/><pre>','\n<pre>') work for html; but not for gradio
        return output

    def get_data_collator():
        return transformers.DataCollatorForLanguageModeling
