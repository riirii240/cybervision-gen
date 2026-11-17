import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cybervision_gan.evaluation.generate import generate_conditioned, vectors_to_matrices
from cybervision_gan.models import Critic, Generator
from cybervision_gan.training.gradient_penalty import gradient_penalty


@pytest.fixture
def models():
    generator = Generator(latent_dim=16, n_classes=11, n_features=63, embedding_dim=8, hidden_dims=(32, 64))
    critic = Critic(n_classes=11, n_features=63, embedding_dim=8, hidden_dims=(64, 32))
    return generator, critic


def test_generator_shape_range_and_gradients(models):
    generator, _ = models
    noise = torch.randn(7, 16)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 10])
    output = generator(noise, labels)
    assert output.shape == (7, 63)
    assert torch.all(output >= -1) and torch.all(output <= 1)
    output.mean().backward()
    assert any(parameter.grad is not None for parameter in generator.parameters())


def test_critic_shape_and_no_sigmoid_constraint(models):
    _, critic = models
    scores = critic(torch.randn(7, 63), torch.arange(7) % 11)
    assert scores.shape == (7,)
    assert not any(isinstance(module, torch.nn.Sigmoid) for module in critic.modules())


def test_gradient_penalty_is_scalar_finite_and_differentiable(models):
    _, critic = models
    real = torch.randn(8, 63)
    fake = torch.randn(8, 63)
    labels = torch.arange(8) % 11
    penalty = gradient_penalty(critic, real, fake, labels)
    assert penalty.ndim == 0
    assert torch.isfinite(penalty)
    assert penalty.item() >= 0
    penalty.backward()
    assert any(parameter.grad is not None for parameter in critic.parameters())


def test_generation_and_padding(models):
    generator, _ = models
    generated = generate_conditioned(generator, 8, 5, 16, "cpu", batch_size=2)
    assert generated.shape == (5, 63)
    layout_path = Path(__file__).parents[2] / "cybervision_data" / "artifacts" / "matrix_layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    matrices = vectors_to_matrices(generated, layout)
    assert matrices.shape == (5, 1, 8, 8)
    assert np.all(matrices[:, 0, 7, 7] == layout["pad_value"])
    np.testing.assert_allclose(matrices.reshape(5, 64)[:, :63], generated)


def test_rejects_incompatible_dimensions(models):
    generator, critic = models
    with pytest.raises(ValueError):
        generator(torch.randn(4, 15), torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError):
        critic(torch.randn(4, 64), torch.zeros(4, dtype=torch.long))
