from src.utils.config import load_config

def test_load_config_returns_dict():
    cfg = load_config("configs/config.yaml")
    assert cfg["qwen"]["router_model"] == "qwen-turbo"
    assert cfg["retrieval"]["rrf_k_const"] == 60
    assert isinstance(cfg["data"]["categories"], list)
