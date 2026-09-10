from io import BytesIO
import json
import math
from typing import List, Optional, Tuple, Union
import logging

import debugpy
import decord
import numpy as np
import pytz
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)
import traceback
import time
from qwen2_5_vl_mm import (
    Qwen2_5_VLProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    process_vision_info,
)
from datetime import datetime


# from transformers import Qwen2_5_VLProcessor
# from transformers import Qwen2_5_VLForConditionalGeneration
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64
import os
import gc
import threading
from datetime import datetime
import logging
import time

@register_model("qwen2_5_vl")
class Qwen2_5_VL(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda:0",
        device_map: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = True,
        min_pixels: int = 3136,
        max_pixels: int = 16384 * 28 * 28,
        max_num_frames: int = 64,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[
            float
        ] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[
            int
        ] = None,  # Only applicable if use_custom_video_loader is True
        tkn_budget: int = None,
        load_in_4bit: bool = False,
        use_chunk: bool = True,
        max_frames_per_group: int = 64,
        select_frame_ratio: float = 1.0,
        **kwargs,
    ) -> None:


        super().__init__()
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,                    # 只记录INFO及以上级别
            format='%(asctime)s - %(levelname)s - %(message)s',  # 输出格式
            handlers=[
                logging.FileHandler(f'logs/{kwargs["dataset"]}_qwen2_5_frames_{max_num_frames}_tkn_budget_{tkn_budget}_select_frame_ratio_{select_frame_ratio}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),  # 保存到文件
                logging.StreamHandler()            # 显示在控制台
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.max_frames_per_group = max_frames_per_group
        self.use_chunk = use_chunk
        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        self.tkn_budget = tkn_budget

        self.gpu_lock = threading.Lock()


        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError(
                "max_image_size is only applicable if use_custom_video_loader is True"
            )

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        config = Qwen2_5_VLConfig.from_pretrained(pretrained)

        from transformers import BitsAndBytesConfig

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None
        if use_flash_attention_2:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
                quantization_config=quantization_config,
            ).eval()
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained,
                config=config,
                torch_dtype="auto",
                device_map=self.device_map,
                attn_implementation="eager",
                quantization_config=quantization_config,
            ).eval()
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            pretrained, max_pixels=max_pixels, min_pixels=min_pixels
        )
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self._config = self.model.config

        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.select_frame_ratio = select_frame_ratio

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    @staticmethod
    def map_2_high_dpi_idx(low_dpi_idx, high_dpi_col_len):
        u_left = low_dpi_idx // (high_dpi_col_len // 2) * 2 * high_dpi_col_len + 2 * (
            low_dpi_idx % (high_dpi_col_len // 2)
        )
        u_right = u_left + 1
        d_left = u_left + high_dpi_col_len
        d_right = d_left + 1
        high_dpi_idx = torch.stack([u_left, u_right, d_left, d_right], dim=-1).view(-1)
        return high_dpi_idx

    def vit_process(self, pixel, thw, use_chunk, chunk_size=32):
        # chunked to avoid OOM
        if use_chunk:
            t, h, w = (
                thw[0][0].item(),
                thw[0][1].item(),
                thw[0][2].item(),
            )
            frame_num = t
            chunk_num = math.ceil(frame_num / chunk_size)
            pixels_per_frame = pixel.shape[0] // frame_num
            output_per_chunk = []
            for chunk_id in range(chunk_num):
                frame_num_per_chunk = (
                    chunk_size
                    if (chunk_id + 1) * chunk_size <= t
                    else t - chunk_id * chunk_size
                )
                start_pixel = chunk_id * chunk_size * pixels_per_frame
                end_pixel = start_pixel + chunk_size * pixels_per_frame
                pixels = pixel[start_pixel:end_pixel, :]
                chunk_video_embeds = self._model.visual(
                    pixels.type(self._model.visual.dtype),
                    grid_thw=torch.tensor(
                        [[frame_num_per_chunk, h, w]], device=pixels.device
                    ),
                )
                output_per_chunk.append(chunk_video_embeds)
            video_embeds = torch.cat(output_per_chunk, dim=0)

        else:
            video_embeds = self._model.visual(
                pixel.type(self._model.visual.dtype),
                grid_thw=thw,
            )
        return video_embeds

    @staticmethod
    def reduced_selected_idx(
        group_tokens_all,
        groups_selected_token_idx,
        num_groups,
        tokens_per_frame,
        frame_indices_all,
    ):

        group_tokens_all_frames = torch.cat(group_tokens_all, dim=0)
        group_tokens_all_base = group_tokens_all_frames[num_groups // 2].unsqueeze(0)

        import torch.nn.functional as F

        cosine_similarities = F.cosine_similarity(
            group_tokens_all_base,  # [1, 2112, 3584]
            group_tokens_all_frames,  # [3, 2112, 3584]
            dim=-1,
        )
        cosine_similarities_frames = cosine_similarities.reshape(
            num_groups, -1, tokens_per_frame
        )

        cosine_similarities_flatten = cosine_similarities_frames.transpose(
            0, 1
        ).reshape(-1)

        final_selected_tokens = []
        count = 0
        from itertools import groupby

        # groups_selected_token_idx // tokens_per_frame

        for _, group in groupby(
            enumerate(groups_selected_token_idx),
            key=lambda x: x[1] // tokens_per_frame,
        ):
            pass
            group_items = list(group)  # [(index, value), ...]
            frame_indice = (group_items[0][1] // tokens_per_frame).item()
            values = [val for _, val in group_items]
            values = torch.tensor(values, device=group_tokens_all_frames.device)
            if torch.all(cosine_similarities_flatten[values].mean() >= 0.8).item():
                if not torch.all(cosine_similarities_flatten[values] >= 0.99).item():
                    count += 1
                idx = frame_indice % num_groups
                final_selected_tokens.append(
                    values + ((num_groups // 2 - idx) * tokens_per_frame)
                )

            else:
                final_selected_tokens.append(values)
        final_selected_tokens = torch.cat(final_selected_tokens, dim=-1).unique(
            sorted=True
        )

        print("Dup frames:" + str(count))
        return final_selected_tokens

    def reduced_selected_tokens(
        self, inputs_high, low_dpi_video_embeds, sorted_indices, tokens_per_frame
    ):
        # reduce high res frames according to low res selection
        total_frames, input_h, input_w = inputs_high["video_grid_thw"][0]
        input_dim = inputs_high["pixel_values_videos"].shape[-1]
        output_dim = low_dpi_video_embeds.shape[-1]

        input_video_embeds = inputs_high["pixel_values_videos"].view(
            total_frames, input_h * input_w, input_dim
        )

        reduced_high_dpi_video_embeds = torch.zeros(
            (total_frames, input_h * input_w // 4, output_dim),
            device=low_dpi_video_embeds.device,
            dtype=low_dpi_video_embeds.dtype,
        )

        frame_indices_to_process = torch.unique(sorted_indices // tokens_per_frame)
        # print(frame_indices_to_process)
        self.logger.info(
            f"origin frame num: {inputs_high['video_grid_thw'][0][0] }"
        )
        self.logger.info(
            f"pre frame num: {frame_indices_to_process.shape[0]}"
        )
        self.logger.info(
            f"all reduce frame num: {inputs_high['video_grid_thw'][0][0] - frame_indices_to_process.shape[0]}"
        )
        frames_to_process = input_video_embeds[frame_indices_to_process]
        high_video_grid_thw = inputs_high["video_grid_thw"].clone()
        high_video_grid_thw[0][0] = frame_indices_to_process.shape[0]
        frames_for_vit = frames_to_process.view(-1, input_dim)

        high_dpi_video_embeds = self.vit_process(
            frames_for_vit,
            high_video_grid_thw,
            use_chunk=True,
            chunk_size=64,
        )

        high_dpi_video_embeds = high_dpi_video_embeds.view(
            high_video_grid_thw[0][0], input_h * input_w // 4, output_dim
        )
        reduced_high_dpi_video_embeds[frame_indices_to_process] = high_dpi_video_embeds
        reduced_high_dpi_video_embeds = reduced_high_dpi_video_embeds.view(
            -1, output_dim
        )

        return reduced_high_dpi_video_embeds

    @staticmethod
    def padding_frame(video_inputs, max_frames_per_group):
        """
        due to the frame fusion of Qwen 2.5 VL,
        the total frames number should be  multiple of 2 * num_groups
        """
        num_groups = math.ceil((video_inputs.shape[0] // 2) / max_frames_per_group)
        if video_inputs.shape[0] % (2 * num_groups) != 0:
            repeat_num = 2 * num_groups - (video_inputs.shape[0] % (2 * num_groups))

            # the padded frames are copies of the last frame
            video_inputs = torch.cat(
                [video_inputs, video_inputs[-1:].repeat(repeat_num, 1, 1, 1)],
                dim=0,
            )
        return video_inputs
    

    def preprocess(self, chunk):
        contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(
            *chunk
        )
        print("doc_id: " + str(doc_id[0]))
        # if str(doc_id) != "(47,)": continue
        task = task[0]
        split = split[0]
        visuals = [
            doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
        ]
        visuals = self.flatten(visuals)

        gen_kwargs = all_gen_kwargs[0]


        until = [self.tokenizer.decode(self.eot_token_id)]

        # Update values from gen_kwargs if present
        if "until" in gen_kwargs:
            until = gen_kwargs.pop("until")
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(
                    f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}"
                )

        messages = []
        processed_visuals = []
        for i, context in enumerate(contexts):
            message = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
            if len(visuals) > 0:
                visual = visuals[i] if i < len(visuals) else None

                if isinstance(visual, str) and visual.endswith(
                    (".mp4", ".avi", ".mov")
                ):  # Video file
                    if self.use_custom_video_loader:
                        visual = read_video_pyav_base64(
                            visual,
                            num_frm=self.max_num_frames,
                            fps=self.fps,
                            img_format="JPEG",
                            max_image_size=self.max_image_size,
                        )
                        image_contents = list(
                            map(lambda x: f"data:image/jpeg;base64,{x}", visual)
                        )
                        message.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "video",
                                        "video": image_contents,
                                        "max_pixels": 360 * 420,
                                    },
                                    {"type": "text", "text": context},
                                ],
                            }
                        )
                    else:
                        message.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "video",
                                        "video": visual,
                                        "max_frames": self.max_num_frames,
                                        "max_pixels": 720 * 1280,
                                        # "max_pixels": 336 * 560,
                                    },
                                    {"type": "text", "text": context},
                                ],
                            }
                        )
                elif isinstance(visual, Image.Image):  # Single image
                    base64_image = visual.convert("RGB")
                    buffer = BytesIO()
                    base64_image.save(buffer, format="JPEG")
                    base64_bytes = base64.b64encode(buffer.getvalue())
                    base64_string = base64_bytes.decode("utf-8")
                    message.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": f"data:image/jpeg;base64,{base64_string}",
                                },
                                {"type": "text", "text": context},
                            ],
                        }
                    )
                elif isinstance(visual, (list, tuple)) and all(
                    isinstance(v, Image.Image) for v in visual
                ):  # Multiple images
                    image_content = []
                    for v in visual:
                        base64_image = v.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        image_content.append(
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{base64_string}",
                            }
                        )
                    message.append(
                        {
                            "role": "user",
                            "content": image_content
                            + [{"type": "text", "text": context}],
                        }
                    )
                else:
                    message.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": context}],
                        }
                    )
            else:
                message.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": context}],
                    }
                )

            messages.append(message)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(self, messages)

        return gen_kwargs, text, image_inputs, video_inputs
    
    def _safe_preprocess(self, chunk):
        # with self.gpu_lock:
        return self.preprocess(chunk)

    @torch.no_grad()
    def generate_until(self, requests: List[Instance]) -> List[str]:


        res = []
        low_consistency_count = 0
        medium_consistency_count = 0

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]
            # return x[3], x[0]

        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        chunks = list(chunks)

        from concurrent.futures import ThreadPoolExecutor, as_completed


        for i in range(len(chunks)):
            chunk_idx = i
            self.logger.info(f"Processing chunk {i}")
            # if i < 1237: continue
            try:

                chunk = chunks[i]
                torch.cuda.synchronize()
                start_time = time.time()

                gen_kwargs, text, image_inputs, video_inputs = self.preprocess(chunk)
                
                high_dpi_video_inputs, medium_dpi_video_inputs, low_dpi_video_inputs = video_inputs[0]
                self.logger.info(f"video frames after padding: low dpi {low_dpi_video_inputs.shape}, medium dpi {medium_dpi_video_inputs.shape}, high dpi {high_dpi_video_inputs.shape}")
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                self.logger.info(f"preprocess time: {total_time} seconds")

                torch.cuda.synchronize()
                start_time = time.time()
                generate_inputs, video_embeds = self.video_process(
                    text, image_inputs, high_dpi_video_inputs
                )

                answers, _ = self.select_generate(
                    generate_inputs,
                    video_embeds,
                    gen_kwargs,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                self.logger.info(f"answer: {answers[0][0]}")
                self.logger.info(f"answer time: {total_time} seconds")
                res.append(answers[0][0])
                
                pbar.update(1)

            except Exception as e:
                print(f"Error: {e} \n", flush=True)
                traceback.print_exc()
                res.append(f"Error: {e}")
                pbar.update(1)
            # reorder this group of results back to original unsorted form
        print("low Consistency count:", low_consistency_count)
        print("medium Consistency count:", medium_consistency_count)
        res = re_ords.get_original(res)
        pbar.close()
        return res
    def video_process(self, text, image_inputs, video_inputs):
        generate_inputs = self.processor(
            text=text[:],
            images=image_inputs,
            videos=[video_inputs],
            padding=True,
            return_tensors="pt",
        )

        if self.device_map == "auto":
            generate_inputs = generate_inputs.to("cuda")
        else:
            generate_inputs = generate_inputs.to(self.device)

        video_embeds = self.vit_process(
            generate_inputs["pixel_values_videos"],
            generate_inputs["video_grid_thw"],
            use_chunk=self.use_chunk,
            chunk_size=270336
            // generate_inputs["video_grid_thw"][0][1].item()
            // generate_inputs["video_grid_thw"][0][2].item(),
        )

        return generate_inputs, video_embeds

    def select_generate(
        self,
        generate_inputs,
        video_embeds,
        gen_kwargs,
        output_attn=False,
        groups_selected_token_idx=None,
    ):

        if groups_selected_token_idx is not None:
            selected_token_idx = groups_selected_token_idx.view(-1).sort().values
        else:
            selected_token_idx = None

        # selected in all low res frames
        # gen_kwargs = self._set_sample_generation_kwargs(gen_kwargs)
        gen_kwargs = self._set_greedy_generation_kwargs(gen_kwargs)

        cont = self.model.generate(
            **generate_inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            num_return_sequences=gen_kwargs["num_return_sequences"],
            use_cache=self.use_cache,
            output_attentions=output_attn,
            return_dict_in_generate=True,
            video_embeds=video_embeds,
            selected_visual_token_idx=selected_token_idx,
        )
        answers = self.tokenizer.batch_decode(
            cont["sequences"][:, generate_inputs["input_ids"].shape[1] :]
        )
        output_attentions = cont["attentions"] if output_attn else None
        return answers, output_attentions


    def _set_greedy_generation_kwargs(self, gen_kwargs):
        gen_kwargs["temperature"] = 0
        gen_kwargs["top_p"] = None
        gen_kwargs["num_beams"] = 1
        gen_kwargs["do_sample"] = False
        gen_kwargs["num_return_sequences"] = 1
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 16
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
        return gen_kwargs

    def _set_sample_generation_kwargs(self, gen_kwargs):
        gen_kwargs["temperature"] = 0.6
        gen_kwargs["top_p"] = 0.9
        gen_kwargs["num_beams"] = 1
        gen_kwargs["do_sample"] = True
        gen_kwargs["num_return_sequences"] = 1
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 16
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
        return gen_kwargs

    def _set_consistency_generation_kwargs(self, gen_kwargs):
        # print("Original gen_kwargs:", gen_kwargs)
        gen_kwargs["num_beams"] = 1
        gen_kwargs["temperature"] = 0.6
        gen_kwargs["top_p"] = 0.9
        gen_kwargs["do_sample"] = True
        gen_kwargs["num_return_sequences"] = 4
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 16
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
        return gen_kwargs

    def _check_soft_consistency(self, groups_outputs):
        """
        使用大模型来判断多个答案是否语义一致
        """
        # 提取所有答案
        answers = []
        for group in groups_outputs:
            for answer in group:
                answers.append(answer.strip().upper())

        # 构建prompt
        answers_text = "\n".join(
            [f"Answer {i+1}: {ans}" for i, ans in enumerate(answers)]
        )

        consistency_prompt = f"""Please analyze whether the following answers are semantically consistent and express the same meaning:

{answers_text}

Consider these answers as consistent if they:
- Express the same core meaning or conclusion
- Refer to the same option (like A, B, C, D) even if formatted differently
- Convey equivalent information despite different wording

Answer only "yes" if all answers are consistent, or "no" if they are inconsistent.

Answer:"""

        messages = [
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that analyzes answer consistency.",
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": consistency_prompt}],
                },
            ]
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text,
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        if self.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,  # 使用贪心解码确保稳定性
                max_new_tokens=5,  # 只需要简短回答
                use_cache=self.use_cache,
            )

        generated_ids_trimmed = outputs[:, inputs["input_ids"].shape[1] :]
        response = (
            self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            .strip()
            .upper()
        )
        is_consistent = "YES" in response

        return is_consistent, answers[0] if is_consistent else None

    def is_consistency(self, groups_outputs, hard=True):
        # Current assert batch size 1
        if hard:
            # groups_outputs = [['A'], ['C'], ['A'], ['A'], ['A']]
            # compare all the answers are the same
            first_answer = groups_outputs[0][0].strip().upper()
            for group in groups_outputs:
                for answer in group:
                    if answer.strip().upper() != first_answer:
                        return False, None
            return True, first_answer
        else:
            # soft consistency 通过调用一个prompt 调用自身大模型, 来让她输出这些答案是不是保持一致的，
            return self._check_soft_consistency(groups_outputs)

    @staticmethod
    def select_token(attn, tkn_budget, ref_layer_idxs, select_start, select_end, group_video_grid_thw, select_frame_ratio,):
        input_length = attn[0][0].shape[-1]
        steps_ref_layer_attn = []
        for generate_step in range(len(attn)):
            # attn: num_layer, batch, num_head, query_len, key_len
            all_layer_attn = torch.stack(attn[generate_step], dim=0)
            ref_layer_attn = all_layer_attn[ref_layer_idxs, :, :, :, :input_length]
            # ref_layer_attn = ref_layer_attn.max(dim=-2).values  # average layers
            steps_ref_layer_attn.append(ref_layer_attn)
            # print("ref_layer_attn shape:", ref_layer_attn.shape)
            # print("len", len(attn))
            # exit()
        del attn
        # torch.cuda.empty_cache()
        steps_ref_layer_attn = torch.cat(steps_ref_layer_attn, dim=3)
        # simplest selection
        steps_ref_layer_attn = steps_ref_layer_attn.transpose(0, 1)
        score = steps_ref_layer_attn.view(
            steps_ref_layer_attn.shape[0], -1, input_length
        ).max(dim=1)[0]
        # Mask the text tokens
        # score[:, :select_start] = -1
        score = score[:, select_start:select_end]

        t, h, w = group_video_grid_thw[0][0].item(), group_video_grid_thw[0][1].item(), group_video_grid_thw[0][2].item()
        tokens_per_frame = (h // 2) * (w // 2)
        score = score.reshape(t, tokens_per_frame)
        frame_avg_scores = score.mean(dim=1) 
        num_selected_frames = max(1, int(t * select_frame_ratio))
        _, top_frame_indices = torch.topk(frame_avg_scores, k=num_selected_frames)
        selected_frame_mask = torch.zeros(t, dtype=torch.bool, device=score.device)
        selected_frame_mask[top_frame_indices] = True

        unselected_frame_mask = ~selected_frame_mask

        score_masked = score.clone()
        score_masked[unselected_frame_mask, :] = float('-inf')
        score = score_masked.reshape(1, -1)
        selected_token_idx = torch.topk(score, k=min(tkn_budget, score.shape[-1]), dim=-1).indices
        # for high res budget
        selected_token_idx_high = torch.topk(score, k=min(tkn_budget // 4, score.shape[-1]), dim=-1).indices
        return selected_token_idx, selected_token_idx_high

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
