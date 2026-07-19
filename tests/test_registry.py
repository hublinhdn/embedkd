import pytest

from embedkd.registry import Registry, RegistryError, registry


def test_builtin_components_registered():
    assert {"cosine", "mse", "kl", "rkd"} <= set(registry.available("distill_objective"))
    assert {"sce", "arcface", "triplet", "contrastive"} <= set(registry.available("task_loss"))
    assert {"image_folder", "csv_manifest", "synthetic"} <= set(registry.available("dataset"))


def test_register_and_get():
    reg = Registry()

    @reg.distill_objective("custom")
    class Custom:
        pass

    assert reg.get("distill_objective", "custom") is Custom


def test_duplicate_registration_rejected():
    reg = Registry()
    reg._register("task_loss", "dup", object)
    with pytest.raises(RegistryError, match="already registered"):
        reg._register("task_loss", "dup", object)


def test_unknown_name_suggests_close_match():
    with pytest.raises(RegistryError, match="Did you mean 'cosine'"):
        registry.get("distill_objective", "cosinee")
