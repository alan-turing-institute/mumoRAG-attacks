import torch
import torchvision.transforms.v2 as T
from strenum import StrEnum
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig, AutoModelForImageTextToText

from utils.image_utils import process_image


# candidate models
class VLMName(StrEnum):
    SMOLVLM_1_256M = "HuggingFaceTB/SmolVLM-256M-Instruct"
    SMOLVLM_1_500M = "HuggingFaceTB/SmolVLM-500M-Instruct"
    SMOLVLM_1_2B = "HuggingFaceTB/SmolVLM-Instruct" 
    # SMOLVLM_2_2B = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    QWEN_2p5_VL_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
    QWEN_2p5_VL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
    LLAVA_ONEVISION_0p5B = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    INTERNVL_3_1B = "OpenGVLab/InternVL3-1B-hf"
    INTERNVL_3_2B = "OpenGVLab/InternVL3-2B-hf"
    INTERNVL_3_8B = "OpenGVLab/InternVL3-8B-hf"
    OVIS_2_1B = "AIDC-AI/Ovis2-1B"
    OVIS_2_2B = "AIDC-AI/Ovis2-2B"
    OVIS_2_4B = "AIDC-AI/Ovis2-4B"
    OVIS_2_8B = "AIDC-AI/Ovis2-8B"

SMOL_VLMS = [
    VLMName.SMOLVLM_1_256M,
    VLMName.SMOLVLM_1_500M,
    VLMName.SMOLVLM_1_2B,
    # VLMName.SMOLVLM_2_2B,    
]

QWEN_VLMS = [
    VLMName.QWEN_2p5_VL_3B,
    VLMName.QWEN_2p5_VL_7B,
]

INTERN_VLMS = [
    VLMName.INTERNVL_3_1B,
    VLMName.INTERNVL_3_2B,
    VLMName.INTERNVL_3_8B,
]

OVIS_VLMS = [
    VLMName.OVIS_2_1B,
    VLMName.OVIS_2_2B,
    VLMName.OVIS_2_4B,
    VLMName.OVIS_2_8B,
]

VLMS_WITH_FAST_PROCESSOR = [
    VLMName.LLAVA_ONEVISION_0p5B,
]
VLMS_WITH_FAST_PROCESSOR.extend(QWEN_VLMS)
VLMS_WITH_FAST_PROCESSOR.extend(INTERN_VLMS)

MODEL_NAMES = [
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    "microsoft/Florence-2-base",
    "google/paligemma2-3b-mix-224", # seems to need specific prompts to work
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    # larger models: to test later:
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    "deepseek-ai/deepseek-vl2-tiny", # 3.75b
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "microsoft/Phi-3.5-vision-instruct",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "OpenGVLab/InternVL3-1B-hf",
    "OpenGVLab/InternVL3-2B-hf",
    "OpenGVLab/InternVL3-8B-hf",
    "AIDC-AI/Ovis2-1B",
    "AIDC-AI/Ovis2-2B",
    "AIDC-AI/Ovis2-4B",
    "AIDC-AI/Ovis2-8B",
]



class VLM:
    """
    Contains functionalities for both Generator VLMs and Judge VLMs
    """
    
    def __init__(self, model_name, device, quantize=False):
        self.name = model_name
        self.device = device

        quantization_config = BitsAndBytesConfig(load_in_4bit=True) if quantize else None
        torch_dtype = torch.float32 if device == "mps" else "auto"
        
        if model_name in INTERN_VLMS:
            self.model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=torch_dtype, quantization_config=quantization_config).to(device)
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch_dtype, quantization_config=quantization_config).to(device)
        # potentially use: _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        
        self.tokenizer = None
        
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.processor.image_processor.do_image_splitting = False
        if self.processor.image_processor.resample == 1: 
            self.processor.image_processor.resample = 3 # change from LANCZOS (1) to BICUBIC (3) since the former has no pytorch implementation

        if model_name in INTERN_VLMS:
            self.processor.image_processor.min_patches, self.processor.image_processor.max_patches = 0, 0
        
        self.model.requires_grad_(False)
        self.model.eval()
    
    
    def get_test_prompt(self, user_query: str, n_images: int = 1):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": user_query}]
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        return prompt

    def get_target_tokens(self, target_generation: str):
        
        msg_without_template = " " + target_generation
        raw_target_tokens = self.processor(text=msg_without_template, return_tensors="pt").to(self.device)['input_ids'][0]

        msg_with_template = [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": target_generation}
                    ]
                },
            ]
        
        prompt_with_template = self.processor.apply_chat_template(msg_with_template, add_generation_prompt=False)
        full_tokens = self.processor(text=prompt_with_template, return_tensors="pt").to(self.device)['input_ids'][0]

        target_tokens = full_tokens[-len(raw_target_tokens)-2:]

        return target_tokens


    def get_training_prompts(self, 
            user_queries: list[str], 
            target_generations: list[str], 
            n_images: int
        ):
        """
        builds the prompt skeleton for the VLM including the image placeholder, the user query, and the required response
        """
        messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": user_queries[i]}]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": target_generations[i]}
                    ]
                },
            ]
            for i in range(len(user_queries))]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)

        target_tokens = [self.get_target_tokens(target_generations[i]) for i in range(len(target_generations))]       

        return prompt, target_tokens
    
    def create_vlm_inputs_for_rag(self, image: torch.tensor, formatted_prompt, context_images, adv_indices: list, overwrite: bool = False):
        """
        NOTE: padding_side should be set to "left", otherwise this will interfere with the attack optimization
        """
        if isinstance(formatted_prompt, str): formatted_prompt = [formatted_prompt]

        # convert images to tensors
        if self.name in VLMS_WITH_FAST_PROCESSOR:
            context_images = self.create_topk_image_list_pt(image, context_images, adv_indices)
        
        # process
        inputs = self.processor(text=formatted_prompt, images=context_images, return_tensors="pt", truncation=True, padding=True, padding_side="left").to(self.device)

        # overwrite with exact adversarial image tensor if needed
        if overwrite and self.name not in VLMS_WITH_FAST_PROCESSOR:
            image_ppd = process_image(image, self)
            for i, adv_idx in enumerate(adv_indices):
                if adv_idx != -1:
                    inputs['pixel_values'][i][adv_idx] = image_ppd
        
        return inputs
    
    @torch.no_grad()
    def generate(self, image: torch.tensor, formatted_prompt, context_images, adv_indices: list, overwrite: bool = False, max_new_tokens=30, do_sample=True, temperature=0.5):
        inputs = self.create_vlm_inputs_for_rag(image, formatted_prompt, context_images, adv_indices, overwrite)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts

    def forward(self, image, formatted_prompt, context_images, adv_indices: list, overwrite: bool = False):
        inputs = self.create_vlm_inputs_for_rag(image, formatted_prompt, context_images, adv_indices, overwrite)
        out = self.model(**inputs, use_cache=False, output_attentions=False, output_hidden_states=False)
        return out

    def compute_gen_loss(self, vlm_output, target_tokens, positive_idx=None):
        if positive_idx is None: positive_idx = [i for i in range(len(target_tokens))]
        loss = torch.zeros((len(target_tokens),))
        for i, tt in enumerate(target_tokens):
            # NOTE: uncomment next line to skip computing generation loss for non-targeted queries (will increase FPR)
            # if i not in positive_idx: continue
            logits_to_optimize = vlm_output.logits[i,-len(tt)-1:-1,:].unsqueeze(0).transpose(1,2)
            tt = tt.unsqueeze(0).repeat(logits_to_optimize.shape[0], 1)
            loss[i] = torch.nn.CrossEntropyLoss()(logits_to_optimize, tt)
        return loss[torch.nonzero(loss)].mean()
    
    def create_topk_image_list_pt(self, adv_image: torch.tensor, retrieved_images: list[list], adv_indices: list[int]):
        topk_images_pt = [
            [T.PILToTensor()(img) for img in images_per_query]
            for images_per_query in retrieved_images
        ]
        for i in range(len(retrieved_images)):
            topk_images_pt[i][adv_indices[i]] = adv_image
        return topk_images_pt
    
    def get_vlm_assistant_delimiter(self,):
        if self.name in SMOL_VLMS:
            return "Assistant:"
        else:
            # valid for Qwen2.5, InternVL3
            return "assistant\n"
