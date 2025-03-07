from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForVision2Seq
import torch
import torchvision.transforms as T
from utils.image_utils import process_image

class VLM():
    """
    Contains functionalities for both Generator VLMs and Judge VLMs
    """
    
    def __init__(self, model_name, device):
        self.name = model_name
        self.device = device

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, 
            torch_dtype=torch.float32 if device == "mps" else torch.bfloat16,).to(device)
        # potentially use: _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        self.tokenizer = None
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.image_processor.do_image_splitting = False
        if self.processor.image_processor.resample == 1: 
            self.processor.image_processor.resample = 3 # change from LANCZOS (1) to BICUBIC (3) since the former has no pytorch implementation
        
        self.model.requires_grad_(False)
        self.model.eval()
    
    
    def get_test_prompt(self, user_query: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_query}
                ]
            },
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        return prompt

    def get_training_prompt(self, 
            user_query: str, # list or str 
            target_generation: str, 
        ):
        """
        builds the prompt skeleton for the VLM including the image placeholder, the user query, and the required response
        """
        target_tokens = self.processor(text=target_generation, return_tensors="pt").to(self.device)['input_ids'][0]
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
                    "content": target_generation
                },
            ]
            for i in range(len(user_query))]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)

        return prompt, target_tokens
    
    @torch.no_grad()
    def generate(self, image: torch.tensor, user_query: str, overwrite: bool = False):
        self.model.eval()
        test_prompt = self.get_test_prompt(user_query)
        inputs = self.processor(text=test_prompt, images=[T.ToPILImage()(image)], return_tensors="pt").to(self.device)
        
        if overwrite:
            image_ppd = process_image(image, self.processor)
            inputs['pixel_values'][0][0] = image_ppd
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.5)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        return generated_texts[0]

    def forward(self, image, mock_images, prompt, overwrite: bool = False):
        self.model.eval()
        if isinstance(prompt, str): prompt = [prompt]
        if self.name == "HuggingFaceTB/SmolVLM-256M-Instruct":
            if overwrite:
                inputs_vlm = self.processor(text=prompt, images=mock_images, return_tensors="pt", truncation=True, padding=True).to(self.device) # here we feed the intiial image since we are overwriting it anyways
                image_ppd_vlm = process_image(image, self.processor)
                inputs_vlm['pixel_values'] = image_ppd_vlm.unsqueeze(0).unsqueeze(0).repeat(len(prompt),1,1,1,1).to(self.device)
            else:
                inputs_vlm = self.processor(text=prompt, images=[image for _ in range(len(prompt))], return_tensors="pt", truncation=True, padding=True).to(self.device) # here we feed the intiial image since we are overwriting it anyways
            out = self.model(**inputs_vlm)
            return out
        
        quit(f"Not supported model {self.name}!")



    def compute_gen_loss(self, vlm_output, target_tokens):
        # TODO: not sure if we need to pass logits to softmax first
        if self.name == "HuggingFaceTB/SmolVLM-256M-Instruct":
            logits_to_optimize = vlm_output.logits[0,-len(target_tokens)-2:-2,:]
            return torch.nn.CrossEntropyLoss()(logits_to_optimize, target_tokens)