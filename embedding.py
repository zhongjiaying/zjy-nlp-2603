from sentence_transformers import SentenceTransformer


def get_embedder(
    test_mode: bool = False,
):
    if test_mode:
        model_name = 'Qwen/Qwen3-Embedding-0.6B'
    else:
        model_name = 'Qwen/Qwen3-Embedding-4B'
    return SentenceTransformer(model_name, model_kwargs={
        'torch_dtype': 'float16'
    })
