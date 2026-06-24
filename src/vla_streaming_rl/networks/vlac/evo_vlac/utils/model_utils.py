import os
import sys
import time
from typing import Any, List, Mapping, Union

import torch
import torchvision.transforms as T
from loguru import logger
from PIL import Image
from torchvision.transforms import InterpolationMode

from ..internvl_vlac import InternLM2Tokenizer, InternVLChatConfig, InternVLChatModel
from ..internvl_vlac.conversation import get_conv_template

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import base64
from io import BytesIO

import requests
import tqdm

from .data_processing_vlm import DataProcessor, trojectory_example_prompt


def clip_one(image):
    width, height = image.size

    if width > height:
        left = (width - height) // 2
        right = left + height
        top = 0
        bottom = height
    else:
        top = (height - width) // 2
        bottom = top + width
        left = 0
        right = width
    square_image = image.crop((left, top, right, bottom))
    image = square_image
    return image


def to_device(data: Any, device: Union[str, torch.device, int]) -> Any:
    """Move inputs to a device"""
    if isinstance(data, Mapping):
        return type(data)({k: to_device(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device)
    else:
        return data


class GAC_model:
    def __init__(self, tag="critic"):
        self.tag = tag
        self.temperature = 1
        self.max_tokens = 10240
        self.do_sample = True
        self.top_logprobs = 10
        self.logprobs = True
        self.top_k = None
        self.system_prompt = None
        self.action_data = {}
        self.critic_data = {}
        self.sft_dataset = None
        self.dataclient = DataProcessor()
        self.dataclient.prompt_templete["v3"] = (
            "Image-1: <image>\nImage-2: <image>\nCompare two images and evaluate whether the second image is closer to achieving task objectives compared to the first image. + score means the second image is closer, - score means the first image is closer\nResponse the relative progressing of target task follow <score>. The target task is: <task> {} </task> <score>"
        )
        self.dataclient.prompt_templete["v3_think"] = (
            "0% <image>\nThis image is the trajectory beginning of the following two images\nImage-1: <image>\nImage-2: <image>\nCompare two images and evaluate whether the second image is closer to achieving task objectives compared to the first image. + score means the second image is closer, - score means the first image is closer\nResponse the relative progressing of target task follow <score>. The target task is: <task> {} </task> <score>"
        )

    def get_score_prompt(self, task, trajectory_len=0, think=False):
        "two or len+3 image"
        if trajectory_len > 0:
            trajectory_prompt = trojectory_example_prompt(list(range(trajectory_len)), task=task)
            full_prompt = trajectory_prompt + self.dataclient.prompt_templete["v3_think"].format(
                task
            )
        else:
            full_prompt = self.dataclient.prompt_templete["v3"].format(task)
        if think:
            full_prompt += self.dataclient.prompt_templete["think"]
        return full_prompt

    def set_config(self):
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.max_new_tokens = self.max_tokens
            self.model.generation_config.do_sample = self.do_sample
            self.model.generation_config.temperature = self.temperature

    def init_model(self, model_path, device_map, torch_dtype=torch.bfloat16):
        import glob

        from safetensors.torch import load_file

        config = InternVLChatConfig.from_pretrained(model_path)
        model = InternVLChatModel(config, use_flash_attn=False).to(torch_dtype)
        state = {}
        for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            state.update(load_file(shard))
        model.load_state_dict(state, strict=True)
        self.model = model.eval().to(device_map)
        self.tokenizer = InternLM2Tokenizer.from_pretrained(model_path, use_fast=False)
        self.image_transform = T.Compose(
            [
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        torch.manual_seed(42)
        logger.success("model initialized successfully")

    def chat(self, infer_requests):
        start_t = time.time()
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        queries = []
        pixel_list = []
        sep = None
        for req in infer_requests:
            template = get_conv_template(self.model.template)
            if self.system_prompt is not None:
                template.system_message = self.system_prompt
            template.append_message(template.roles[0], req["prompt"])
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            for image in req["images"]:
                pixel_list.append(self.image_transform(image).unsqueeze(0))
                image_tokens = (
                    IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token + IMG_END_TOKEN
                )
                query = query.replace("<image>", image_tokens, 1)
            queries.append(query)
            sep = template.sep
        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer(queries, return_tensors="pt", padding=True)
        input_ids = model_inputs["input_ids"].to(self.model.device)
        attention_mask = model_inputs["attention_mask"].to(self.model.device)
        pixel_values = torch.cat(pixel_list).to(self.model.device, dtype=self.model.dtype)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(sep)
        generation_output = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            max_new_tokens=self.max_tokens,
            eos_token_id=eos_token_id,
        )
        responses = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(sep)[0].strip() for response in responses]
        return responses, time.time() - start_t

    def results_format(self, response_list, infer_requests, rich=False):
        if rich:
            raise NotImplementedError(
                "rich output relies on ms-swift logprobs, removed in this port"
            )
        return list(response_list), infer_requests

    def set_system_prompt(self, system_prompt=None):
        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif system_prompt == "default":
            self.system_prompt = None
        else:
            self.system_prompt = "You are a visual-language assistant designed to interpret spatial and task-related information from images and text. Provide precise, context-aware responses and actionable guidance to assist in achieving task objectives."

    def _process_image_to_pil(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """
        将各种格式的图像输入转换为448x448的PIL.Image

        Args:
            image_input: 图像URL、文件路径、base64编码字符串或PIL.Image对象

        Returns:
            PIL.Image: 调整为448x448尺寸的PIL图像
        """
        pil_image = None

        if isinstance(image_input, Image.Image):
            pil_image = image_input
        elif isinstance(image_input, str):
            if image_input.startswith(("http://", "https://")):
                response = requests.get(image_input)
                pil_image = Image.open(BytesIO(response.content))
            elif image_input.startswith("data:image"):
                header, encoded = image_input.split(",", 1)
                image_data = base64.b64decode(encoded)
                pil_image = Image.open(BytesIO(image_data))
            elif len(image_input) > 100 and not "/" in image_input and not "\\" in image_input:
                try:
                    image_data = base64.b64decode(image_input)
                    pil_image = Image.open(BytesIO(image_data))
                except:
                    pil_image = Image.open(image_input)
            else:
                pil_image = Image.open(image_input)
        else:
            raise ValueError(f"不支持的图像输入类型: {type(image_input)}")

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        pil_image = pil_image.resize((448, 448), Image.Resampling.LANCZOS)

        return pil_image

    def get_infer_requests(self, prompt, images: List[Union[str, Image.Image]] = None):
        """
        Args:
            prompt: 提示文本
            images: 可以是图像URL、文件路径、base64编码字符串或PIL.Image对象
        """
        if type(prompt) == str:
            prompt = [prompt]
            images = [images]
        infer_requests = []
        for i in range(len(prompt)):
            if images[i] is None:
                processed_images = []
            elif isinstance(images[i], list):
                processed_images = [
                    self._process_image_to_pil(img) for img in images[i] if img is not None
                ]
            else:
                processed_images = [self._process_image_to_pil(images[i])]
            infer_requests.append({"prompt": prompt[i], "images": processed_images})
        return infer_requests

    def get_trajectory_critic(
        self,
        task: str,
        image_list: List[Image.Image],
        ref_image_list: List[Image.Image] = None,
        batch_num: int = 20,
        ref_num=9,
        think=False,
        skip=1,
        rich=False,
        reverse_eval=False,
        frame_skip=True,
        addition_scale=1,
        bias=0,
        related_critic=False,
        positive_clip=0,
        negative_clip=0,
        value_simple=True,
    ):
        """
        输入一条trajectory的所有图片,输出每张图片的critic和processing value
        可以给一条参考轨迹
        """
        batch_prompt = []
        batch_image = []
        critic_list = []
        value_list = []
        if ref_image_list is not None:
            ref_images = [ref_image_list[0]]
            delta = (len(ref_image_list) - 1) / (ref_num - 1)
            for i in range(1, ref_num):
                ref_images.append(ref_image_list[int(i * delta)])
        else:
            ref_num = 0
        if frame_skip:
            select_idx = range(skip, len(image_list), skip)
        else:
            select_idx = range(skip, len(image_list))
        for i in tqdm.tqdm(select_idx, desc="critic processing"):
            one_prompt = self.get_score_prompt(task=task, trajectory_len=ref_num, think=think)
            batch_prompt.append(one_prompt)
            if ref_image_list is not None:
                if reverse_eval:
                    batch_image.append(
                        ref_images + [image_list[0], image_list[i], image_list[i - skip]]
                    )
                else:
                    batch_image.append(
                        ref_images + [image_list[0], image_list[i - skip], image_list[i]]
                    )
            else:
                if reverse_eval:
                    batch_image.append([image_list[i], image_list[i - skip]])
                else:
                    batch_image.append([image_list[i - skip], image_list[i]])
            if (len(batch_prompt)) % batch_num == 0 or len(critic_list) + len(batch_prompt) == len(
                select_idx
            ):
                infer_requests = self.get_infer_requests(prompt=batch_prompt, images=batch_image)
                response_list, infer_time = self.chat(infer_requests)
                answers_list, complete_requests_list = self.results_format(
                    response_list, infer_requests, rich=rich
                )
                print(f"infer_time:{infer_time}s")
                print(f"answers_list:{answers_list}")
                critic_list.extend(answers_list)
                batch_prompt = []
                batch_image = []
        if think:
            think_pre_value_list = []
            think_post_value_list = []
            think_critic_list = []
            for one in critic_list:
                one_critic = one.split("</think>")[1]
                one_pre_value = one.split("first image progressing: ")[1].split("%")[0]
                one_post_value = one.split("second image progressing: ")[1].split("%")[0]
                think_critic_list.append(one_critic)
                think_pre_value_list.append(one_pre_value)
                think_post_value_list.append(one_post_value)
            if reverse_eval:
                temp = think_pre_value_list
                think_pre_value_list = think_post_value_list
                think_post_value_list = temp
                think_critic_list = [0 - float(one) for one in think_critic_list]
            think_pre_value_list = think_pre_value_list + think_post_value_list[-skip:]
            think_post_value_list = think_pre_value_list[:skip] + think_post_value_list
            critic_list = think_critic_list
            if frame_skip:
                pass
            else:
                critic_list = [float(one) / skip for one in critic_list]
            critic_list = [float(one) / addition_scale for one in critic_list]
            value_list = self.critic_to_value_simple(critic_list, simple=value_simple)
        else:
            if reverse_eval:
                critic_list = [0 - float(one) for one in critic_list]
            if frame_skip:
                pass
            else:
                critic_list = [float(one) / skip for one in critic_list]
            critic_list = [float(one) / addition_scale for one in critic_list]
            value_list = self.critic_to_value_simple(critic_list, simple=value_simple)
        if related_critic:
            critic_list = [value_list[i] - value_list[i - 1] for i in range(1, len(value_list))]
        if bias != 0:
            critic_list = [one + bias for one in critic_list]
        if positive_clip != 0:
            critic_list = [one if (one < 0 or one > positive_clip) else 0 for one in critic_list]
        if negative_clip != 0:
            critic_list = [one if (one > 0 or one < -negative_clip) else 0 for one in critic_list]
        if related_critic:
            value_list = [0]
            for one in critic_list:
                value_list.append(value_list[-1] + one)
        else:
            value_list = self.critic_to_value_simple(critic_list, simple=value_simple)
        # 这里的value也是done，越接近100完成度越高
        return critic_list, value_list

    def critic_to_value_simple(self, critic_list, simple=True):
        """
        将critic计算为0-100的value,输入需是一条轨迹按顺序的critic
        """
        value_list = [0]
        for i in range(len(critic_list)):
            if float(critic_list[i]) > 0 or simple is True:
                value_list.append(
                    value_list[-1] + (100 - value_list[-1]) * float(critic_list[i]) / 100.0
                )
            else:
                if simple == "mix_f":
                    if value_list[-1] > 50:
                        value_list.append(
                            max(
                                10,
                                100
                                - max((100 - value_list[-1]), 1.0)
                                / (100 + float(critic_list[i]))
                                * 100.0,
                            )
                        )
                    else:
                        value_list.append(
                            value_list[-1] + value_list[-1] * float(critic_list[i]) / 100.0
                        )
                else:
                    value_list.append(
                        100
                        - max((100 - value_list[-1]), 1.0) / (100 + float(critic_list[i])) * 100.0
                    )
        return value_list


def import_external_file(file_path: str):
    import importlib

    file_path = os.path.abspath(os.path.expanduser(file_path))
    py_dir, py_file = os.path.split(file_path)
    assert os.path.isdir(py_dir), f"py_dir: {py_dir}"
    sys.path.insert(0, py_dir)
    return importlib.import_module(py_file.split(".", 1)[0])
