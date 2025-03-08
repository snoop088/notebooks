import torch
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    pipeline
)
from typing_extensions import TypedDict
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint

class OpenLLMCreatorHF():

    """ Creates a model with HuggingFace pipeline. If chat is true, it is wrapped in the ChatHuggingFace interfce """

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    def __init__(self, model: str, max_tokens = 512, temperature: float = 0.5, chat: bool = True, top_p: float = 0.85):
        self.model = model
        self.temperature = temperature
        self.chat = chat
        self.top_p = top_p
        self.max_tokens = max_tokens

    def get_model(self, chat = None):

        if chat is None:
            chat = self.chat
        generation_config = GenerationConfig.from_pretrained(self.model)
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        config = self._get_gen_conf(tokenizer, generation_config)

        llm_pipeline = pipeline("text-generation", 
                        model=self.model,  
                        device_map=OpenLLMCreatorHF.DEVICE,
                        torch_dtype=torch.bfloat16,
                        max_new_tokens=self.max_tokens,
                        return_full_text=False)
        if "llama" in self.model.lower():      
            llm_pipeline.tokenizer.pad_token_id = llm_pipeline.model.config.eos_token_id = config["eos_token_id"]
        else:
            llm_pipeline.tokenizer.pad_token_id = llm_pipeline.model.config.eos_token_id
            
        open_llm_with_pipe = HuggingFacePipeline(pipeline=llm_pipeline, pipeline_kwargs=config)
        return ChatHuggingFace(llm=open_llm_with_pipe, verbose=True, tokenizer=tokenizer) if chat else open_llm_with_pipe
       
    def get_from_endpoint(self, chat = None):
        
        if chat is None:
            chat = self.chat

        llm = HuggingFaceEndpoint(
            repo_id=self.model,
            task="text-generation",
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=False,
            repetition_penalty=1.03,
        )       

        return ChatHuggingFace(llm=llm, verbose=True) if chat else llm


    def _get_gen_conf(self, tokenizer, gen_config) -> TypedDict:
        
        if "llama" in self.model.lower():
            stop_token = "<|eot_id|>"  
            stop_token_id = tokenizer.encode(stop_token)[0]
            begin_token = "<|begin_of_text|>"
            begin_token_id = tokenizer.encode(begin_token)[0]
            gen_config.eos_token_id = stop_token_id
            gen_config.begin_token_id = begin_token_id
        
        model_config = gen_config.to_dict()

        combined = {
            **model_config,
            "top_p": self.top_p,  # changed from 0.15
            "temperature": self.temperature,
            "do_sample": False,  # changed from true
            "torch_dtype": torch.bfloat16,  # bfloat16
            "use_fast": True,
            "repetition_penalty": 1.1,
        }
        return combined