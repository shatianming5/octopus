import pytest
import torch
from transformers import AutoModelForCausalLM, FalconH1Config, Gemma2Config, GraniteMoeHybridConfig, LlamaConfig

from octopus.rl.lm_head import bind_head

SMALL = dict(vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
             num_key_value_heads=1, head_dim=16)


@pytest.mark.parametrize("config", [
    LlamaConfig(**SMALL),  # no post-head op
    GraniteMoeHybridConfig(**SMALL, logits_scaling=3.0, layer_types=["mamba", "attention"], mamba_n_heads=4,
                           mamba_d_head=16, mamba_d_state=8, mamba_chunk_size=8),  # logits / logits_scaling
    Gemma2Config(**SMALL, final_logit_softcapping=2.0),  # tanh soft-capping
    FalconH1Config(**SMALL, lm_head_multiplier=0.25, mamba_n_heads=4, mamba_d_head=16, mamba_d_ssm=64,
                   mamba_d_state=8, mamba_chunk_size=8),  # logits * lm_head_multiplier
], ids=lambda c: c.model_type)
def test_chunked_head_matches_the_models_own_logits(config):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    ids = torch.randint(0, config.vocab_size, (1, 12))
    labels = torch.roll(ids, -1, -1).reshape(-1)
    with torch.no_grad():
        own = model(input_ids=ids).logits[0].float()
    for temperature in (1.0, 0.7):
        head = bind_head(model, temperature)
        head.octopus_labels, head.octopus_tree = labels, None
        with torch.no_grad():
            out = model(input_ids=ids).logits[0]
        logp = torch.log_softmax(own / temperature, -1)
        assert torch.allclose(out[:, 0], logp.gather(-1, labels[:, None])[:, 0], atol=1e-5)
        assert torch.allclose(out[:, 1], -(logp.exp() * logp).sum(-1), atol=1e-5)
    assert bool(head.octopus_post) == (config.model_type != "llama")  # the probe found the op (or none)


@pytest.mark.parametrize("tail", [
    lambda m, logits: (logits.__setitem__((..., 0), 0.0), logits)[1],  # masked in place
    lambda m, logits: (logits * 2, logits / 3)[1],  # the flow forks: an op on an earlier result
    lambda m, logits: m.extra(logits),  # a submodule after the head
], ids=["setitem", "fork", "submodule"])
def test_unreproducible_post_head_logits_fail_loudly(tail):
    from transformers.modeling_outputs import CausalLMOutput
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    class Model(LlamaForCausalLM):
        def __init__(self, config):
            super().__init__(config)
            self.extra = torch.nn.Linear(config.vocab_size, config.vocab_size)

        def forward(self, input_ids=None, **kwargs):
            return CausalLMOutput(logits=tail(self, self.lm_head(self.model(input_ids=input_ids).last_hidden_state)))

    with pytest.raises(NotImplementedError):
        bind_head(Model(LlamaConfig(**SMALL)), 1.0)


def test_flat_head_runs_only_the_scored_positions():
    """logprob, entropy and the gradients of a loss over the scored positions equal the unscored head's; 0 elsewhere."""
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(LlamaConfig(**SMALL), attn_implementation="eager").double()
    ids = torch.randint(0, SMALL["vocab_size"], (1, 12))
    head = bind_head(model, 0.7)
    head.octopus_labels, head.octopus_tree = torch.roll(ids, -1, -1).reshape(-1), None
    scored = torch.rand(12) < 0.4
    w = torch.randn(1, 12, 2, dtype=torch.float64) * scored[None, :, None]  # a loss over the scored positions
    outs, grads = [], []
    for head.octopus_scored in (None, scored):
        model.zero_grad(set_to_none=True)
        out = model(input_ids=ids).logits
        (out * w).sum().backward()
        outs.append(out.detach()[0])
        grads.append([p.grad for p in model.parameters()])
    assert torch.equal(outs[1][scored], outs[0][scored]) and (outs[1][~scored] == 0).all()
    assert all(torch.equal(a, b) for a, b in zip(*grads))


def test_scored_positions_follow_areal_loss_mask():
    pytest.importorskip("areal.engine")
    from octopus.rl.actor import _loss_mask_rolled, scored_positions

    mask = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 0]])  # the rollout's: 1 on each trained token
    assert scored_positions({"loss_mask": mask}).tolist() == [0, 1, 1, 0, 0, 1, 0, 0]  # compute_logp: the one before
    rolled = torch.roll(mask, -1, -1)  # compute_advantages / prepare_mopd_batch roll it onto the scoring position
    with _loss_mask_rolled():  # ppo_update's, with or without advantages (pure MOPD adds none)
        assert scored_positions({"loss_mask": rolled}).tolist() == rolled[0].bool().tolist()
    assert scored_positions({"loss_mask": mask}).tolist() == [0, 1, 1, 0, 0, 1, 0, 0]
    assert scored_positions({}) is None
