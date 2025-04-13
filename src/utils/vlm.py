from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForVision2Seq, BitsAndBytesConfig
import torch
import torchvision.transforms as T
from .image_utils import process_image
from strenum import StrEnum

# candidate models
class VLMName(StrEnum):
    SMOLVLM_1_256M = "HuggingFaceTB/SmolVLM-256M-Instruct"
    SMOLVLM_1_500M = "HuggingFaceTB/SmolVLM-500M-Instruct"
    SMOLVLM_1_2B = "HuggingFaceTB/SmolVLM-Instruct" 
    # SMOLVLM_2_2B = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    QWEN_2p5_VL_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
    QWEN_2p5_VL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"

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
    "google/gemma-3-12b-it"
    ]



class VLM():
    """
    Contains functionalities for both Generator VLMs and Judge VLMs
    """
    
    def __init__(self, model_name, device, quantize=False):
        self.name = model_name
        self.device = device

        quantization_config = BitsAndBytesConfig(load_in_4bit=True) if quantize else None

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, 
            torch_dtype=torch.float32 if device == "mps" else "auto",
            quantization_config=quantization_config).to(device)
        # potentially use: _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        
        self.tokenizer = None
        
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        
        self.processor.image_processor.do_image_splitting = False
        if self.processor.image_processor.resample == 1: 
            self.processor.image_processor.resample = 3 # change from LANCZOS (1) to BICUBIC (3) since the former has no pytorch implementation
        
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


    def get_training_prompt(self, 
            user_query: str, # list or str 
            target_generation: str, 
        ):
        """
        builds the prompt skeleton for the VLM including the image placeholder, the user query, and the required response
        """
        if isinstance(user_query, str): user_query = [user_query]

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_query[i]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": target_generation}
                    ]
                },
            ]
            for i in range(len(user_query))]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)

        target_tokens = self.get_target_tokens(target_generation)       

        return prompt, target_tokens
    
    @torch.no_grad()
    def generate(self, image: torch.tensor, user_queries, overwrite: bool = False, retrieved_images = None, adv_indices: list = None):
        self.model.eval()
        if isinstance(user_queries, str): user_queries = [user_queries]
        topk_used = len(retrieved_images[0])
        test_prompts = [self.get_test_prompt(query, n_images=topk_used) for query in user_queries]
        
        if self.name in QWEN_VLMS:
            # TODO: Not tested with large k due to OOM exceptions
            retrieved_images_pt = self.create_topk_image_list_pt(image, retrieved_images, adv_indices)
            inputs = self.processor(text=test_prompts, images=retrieved_images_pt, return_tensors="pt", padding=True, padding_side="left").to(self.device)
        else:            
            inputs = self.processor(text=test_prompts, images=retrieved_images, return_tensors="pt", padding=True, padding_side="left").to(self.device)

            if overwrite:
                image_ppd = process_image(image, self)
                for i, adv_idx in enumerate(adv_indices):
                    if adv_idx != -1:
                        inputs['pixel_values'][i][adv_idx] = image_ppd
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.5)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return generated_texts

    def forward(self, image, mock_images, prompt, overwrite: bool = False):
        """
        Important:
        padding_side should be set to "left", otherwise this will interfere with the attack optimization
        """
        self.model.eval()
        if isinstance(prompt, str): prompt = [prompt]
        if self.name in SMOL_VLMS:
            if overwrite:
                inputs_vlm = self.processor(text=prompt, images=mock_images, return_tensors="pt", truncation=True, padding=True, padding_side="left").to(self.device) # here we feed the initial image since we are overwriting it anyway
                image_ppd_vlm = process_image(image, self)
                inputs_vlm['pixel_values'] = image_ppd_vlm.unsqueeze(0).unsqueeze(0).repeat(len(prompt),1,1,1,1).to(self.device)
            else:
                inputs_vlm = self.processor(text=prompt, images=[image for _ in range(len(prompt))], return_tensors="pt", truncation=True, padding=True, padding_side="left").to(self.device) # here we feed the initial image since we are overwriting it anyway
        
        if self.name in QWEN_VLMS:
            # it seems that qwen implements their preprocessors in pytorch --> differentiable (no need to overwrite image)
            inputs_vlm = self.processor(text=prompt, images=[image for _ in range(len(prompt))], return_tensors="pt", truncation=True, padding=True, padding_side="left").to(self.device)
        
        if inputs_vlm:
            out = self.model(**inputs_vlm, use_cache=False, output_attentions=False, output_hidden_states=False)
            return out

        raise ValueError(f"Not supported model {self.name}!")


    def compute_gen_loss(self, vlm_output, target_tokens):
        logits_to_optimize = vlm_output.logits[:,-len(target_tokens)-1:-1,:].transpose(1,2)
        target_tokens = target_tokens.unsqueeze(0).repeat(logits_to_optimize.shape[0], 1)
        return torch.nn.CrossEntropyLoss()(logits_to_optimize, target_tokens)
    
    def create_topk_image_list_pt(self, adv_image: torch.tensor, retrieved_images: list[list], adv_indices: int):
        topk_images_pt = [
            [T.PILToTensor()(img) for img in imgs_per_query]
            for imgs_per_query in retrieved_images
        ]
        for i in range(len(retrieved_images)):
            topk_images_pt[i][adv_indices[i]] = adv_image
        return topk_images_pt
       
