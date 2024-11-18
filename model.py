from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from accelerate import Accelerator

GPU_LAYERS = 50
MODEL_PATH = "models/llama-2-7b-chat.ggmlv3.q8_0.bin"

## Preparing LLAMA model for inferencing
def build_llm_model(gpu_enabled: bool = False):

    config = {
        "max_new_tokens": 512,
        "repetition_penalty": 1.1,
        "context_length": 8000,
        "temperature": 0.01,
    }

    if not gpu_enabled:
        model = CTransformers(
            model=MODEL_PATH,
            model_type="llama",
            config=config,
        )

    else:
        accelerator = Accelerator()
        config["gpu_layers"] = GPU_LAYERS

        model = CTransformers(
            model=MODEL_PATH,
            model_type="llama",
            gpu_layers=GPU_LAYERS,
            config=config,
        )

        model, config = accelerator.prepare(model, config)

    return model

## Adjust the prompt template using the user input 
def get_model_response(blog_topic: str, number_of_words: str, blog_target_audience: str, use_gpu: bool):

    template = """
                You are an experienced blogger and your target audience are {blog_target_audience}. Write a blog post about {blog_topic} within {number_of_words} words.
                """

    prompt = PromptTemplate(
        inpt_variables=["blog_target_audience", "blog_topic", "number_of_words"],
        template=template,
    )

    model = build_llm_model(gpu_enabled=use_gpu)

    response = model.invoke(
        prompt.format(
            blog_target_audience=blog_target_audience,
            blog_topic=blog_topic,
            number_of_words=number_of_words,
        )
    )

    return response
