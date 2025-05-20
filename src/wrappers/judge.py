from typing import Any

from .vlm import VLM
from strenum import StrEnum


class JudgePrompt:

    # The following prompts are taken form this paper
    # https://arxiv.org/pdf/2410.21943

    ANSWER_RELEVANCY_PROMPT = """Evaluate the following metric:
answer_relevancy: Is the answer relevant to the
user's query? (YES or NO)
QUERY: >>query<<
ANSWER: >>answer<<
Write out in a step by step manner your reasoning to be sure that your conclusion is correct by filling out the following JSON format with the grade and a concise reason behind the grade: 
{grade: ' ', 'reason': ' '}
Output the reason as a string, not as a list.
The only allowed grades are YES or NO."""

    IMAGE_CONTEXT_RELEVANCY_PROMPT = """Evaluate the following metric by comparing the user query with the provided image:
image_context_relevancy: Is the content of the images relevant to the user's query , i.e. can it contribute to answer the query? (YES or NO)
QUERY: >>query<<
IMAGES: >>images<<
Write out in a step by step manner your reasoning to be sure that your conclusion is correct by filling out the following JSON format with the grade and a concise reason behind the grade:
{grade: ' ', 'reason': ' '}
Output the reason as a string, not as a list.
The only allowed grades are YES or NO."""

    IMAGE_FAITHFULNESS_PROMPT = """Evaluate the following metric by comparing the answer with the provided images:
image_faithfulness: Is the answer faithful to the content of the images, i.e. does it factually align with any of the images? (YES or NO)
GENERATED ANSWER: >>answer<<
IMAGES: >>images<<
Write out in a step by step manner your reasoning to be sure that your conclusion is correct by filling out the following JSON format with the grade and a concise reason behind the grade: 
{grade: ' ', 'reason': ' '}
Output the reason as a string, not as a list.
The only allowed grades are YES or NO."""


class JudgeMetric(StrEnum):
    ANSWER_RELEVANCY = "answer_relevancy"
    IMAGE_CONTEXT_RELEVANCY = "image_context_relevancy"
    IMAGE_FAITHFULNESS = "image_faithfulness"

METRIC_2_PROMPT = {
    JudgeMetric.ANSWER_RELEVANCY:       JudgePrompt.ANSWER_RELEVANCY_PROMPT,
    JudgeMetric.IMAGE_CONTEXT_RELEVANCY: JudgePrompt.IMAGE_CONTEXT_RELEVANCY_PROMPT,
    JudgeMetric.IMAGE_FAITHFULNESS:     JudgePrompt.IMAGE_FAITHFULNESS_PROMPT
}



class JudgeVLM(VLM):
    def judge_prompt_components(self, template: str, query=None, answer=None, n_images=0):
        template = template.replace(">>query<<", query).replace(">>answer<<", answer)
        prompt_list = template.split(">>images<<")
        before  = [{"type": "text", "text": prompt_list[0]}] 
        after = [{"type": "text", "text": prompt_list[1]}] if len(prompt_list) == 2 else []
        contexts = [{"type": "image"} for _ in range(n_images)]
        return before, contexts, after


    def get_test_prompt(self, template: str, query=None, answer=None, n_images=0):
        before, contexts, after = self.judge_prompt_components(template, query, answer, n_images)
        messages = [
            {
                "role": "user",
                "content":
                    before +
                    contexts + 
                    after 
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        return prompt


    def get_training_prompts(
        self,
        queries: list[str],
        target_vlm_generation: list[str],
        target_jdg_generation: str,
        jdg_metric_list: list[JudgeMetric],
        n_images: int
    ):
        """
        builds the prompt skeleton for the VLM including the image placeholder, the user query, and the required response
        """
        all_messages = []
        for metric in jdg_metric_list:
            template = METRIC_2_PROMPT[metric]
            for query, target in zip(queries, target_vlm_generation):
                before, contexts, after = self.judge_prompt_components(template, query, target, n_images)
                messages = [
                    {
                        "role": "user",
                        "content": 
                            before +
                            contexts+
                            after
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": target_jdg_generation}
                        ]
                    },
                ]
                all_messages.append(messages)
        
        prompts = self.processor.apply_chat_template(all_messages, add_generation_prompt=False)
        target_tokens = [self.get_target_tokens(target_jdg_generation) for _ in range(len(all_messages))]

        return prompts, target_tokens
