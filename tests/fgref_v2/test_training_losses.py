import torch

from models.fgref_v2 import (
    FGRefV2LossWeights,
    compose_fgref_v2_losses,
    identity_consistency_loss,
    masked_binary_referring_loss,
)


def test_identity_loss_ignores_unlabelled_rows_and_requires_positive_pairs():
    embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [9.0, 9.0]], requires_grad=True
    )
    ids = torch.tensor([4, 4, 8, -1])
    loss = identity_consistency_loss(embeddings, ids)
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None
    assert embeddings.grad[-1].abs().sum().item() == 0.0


def test_identity_loss_returns_differentiable_zero_without_same_id_pair():
    embeddings = torch.randn(3, 4, requires_grad=True)
    loss = identity_consistency_loss(embeddings, torch.tensor([1, 2, -1]))
    assert loss.item() == 0.0
    loss.backward()
    assert embeddings.grad.abs().sum().item() == 0.0


def test_masked_referring_and_optional_losses_are_explicit():
    logits = torch.zeros(2, 2, 1, requires_grad=True)
    targets = torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]])
    mask = torch.tensor([[True, False], [True, True]])
    loss = masked_binary_referring_loss(logits, targets, mask)
    result = compose_fgref_v2_losses(
        {"loss_refer": loss},
        counterfactual_logits=logits,
        counterfactual_targets=targets,
        counterfactual_valid_mask=mask,
        weights=FGRefV2LossWeights(counterfactual=0.5),
    )
    assert set(result) == {"loss_refer", "loss_fgref_v2_counterfactual"}
    total = sum(result.values())
    total.backward()
    assert logits.grad is not None
