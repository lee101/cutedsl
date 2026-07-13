import pytest
import torch
import torch.nn as nn

DIM = 32
N_LAYERS = 2


class TinyBlock(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)


class TinyTransformer(nn.Module):
    def __init__(self, dim=DIM, n_layers=N_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock(dim) for _ in range(n_layers)])

    def forward(self, x):
        for block in self.layers:
            x = x + block.o_proj(block.q_proj(x))
        return x


class TinyDiffusersAttention(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])


class TinyDiffusersBlock(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.attention = TinyDiffusersAttention(dim)


class TinyDiffusersTransformer(nn.Module):
    def __init__(self, dim=DIM, n_layers=N_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList([TinyDiffusersBlock(dim) for _ in range(n_layers)])


class TinyFusedBlock(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.Linear(dim, dim * 4, bias=True))


class TinyFusedTransformer(nn.Module):
    def __init__(self, dim=DIM, n_layers=N_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList([TinyFusedBlock(dim) for _ in range(n_layers)])


@pytest.fixture
def transformer():
    torch.manual_seed(0)
    return TinyTransformer()


@pytest.fixture
def fused_transformer():
    torch.manual_seed(0)
    return TinyFusedTransformer()


@pytest.fixture
def diffusers_transformer():
    torch.manual_seed(0)
    return TinyDiffusersTransformer()


def make_zimage_lora(path, rank=4, dim=DIM, n_layers=N_LAYERS, seed=1, mods=("to_q", "to_k"), adaln=False):
    from safetensors.torch import save_file

    torch.manual_seed(seed)
    tensors = {}
    for i in range(n_layers):
        for mod in mods:
            prefix = f"diffusion_model.layers.{i}.attention.{mod}"
            tensors[f"{prefix}.lora_A.weight"] = torch.randn(rank, dim) * 0.1
            tensors[f"{prefix}.lora_B.weight"] = torch.randn(dim, rank) * 0.1
        if adaln:
            prefix = f"diffusion_model.layers.{i}.adaLN_modulation.0"
            tensors[f"{prefix}.lora_A.weight"] = torch.randn(rank, dim) * 0.1
            tensors[f"{prefix}.lora_B.weight"] = torch.randn(dim * 4, rank) * 0.1
    save_file(tensors, str(path))
    return path


@pytest.fixture
def lora_file(tmp_path):
    return make_zimage_lora(tmp_path / "test_lora.safetensors")
