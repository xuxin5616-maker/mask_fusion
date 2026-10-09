"""CPU contracts for the multi-rank training control flow (no fake GPU claim)."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mask_fusion import train


def context(rank=0, world_size=2):
    return SimpleNamespace(rank=rank, world_size=world_size, local_rank=rank, device=torch.device("cpu"), is_main=rank == 0)


class SmallDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor([10, index + 1]),
            "labels": torch.tensor([-100, index + 1]),
            "images": torch.zeros(1, 3, 14, 14), "masks": torch.zeros(1, 1, 14, 14),
            "mask_valid": torch.ones(1, 1, 14, 14, dtype=torch.bool), "image_size": (14, 14),
        }


class LabelLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.labels_seen = []

    def forward(self, labels, **kwargs):
        self.labels_seen.append(labels.clone())
        return SimpleNamespace(loss=labels[labels != -100].float().mean())


class DistributedSamplingTests(unittest.TestCase):
    def test_equal_shards_cover_each_real_sample_once_including_tiny_datasets(self):
        for size in range(1, 8):
            for world_size in range(1, 9):
                with self.subTest(size=size, world_size=world_size):
                    real, lengths, repeated = [], [], 0
                    for rank in range(world_size):
                        indices, valid = train._distributed_indices(size, rank=rank, world_size=world_size, seed=13)
                        lengths.append(len(indices))
                        real.extend(index for index, keep in zip(indices, valid) if keep)
                        repeated += sum(not keep for keep in valid)
                    self.assertEqual(len(set(lengths)), 1)
                    self.assertGreater(lengths[0], 0)
                    self.assertEqual(sorted(real), list(range(size)))
                    self.assertEqual(repeated, lengths[0] * world_size - size)

    def test_seeded_shards_and_resume_slices_are_deterministic(self):
        first, _ = train._distributed_indices(31, rank=1, world_size=2, seed=42)
        repeated, _ = train._distributed_indices(31, rank=1, world_size=2, seed=42)
        changed, _ = train._distributed_indices(31, rank=1, world_size=2, seed=43)
        self.assertEqual(first[4:], repeated[4:])
        self.assertNotEqual(first, changed)


class DistributedValidationTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(seed=0, batch_size=2, num_workers=0, device="cpu")

    def test_padding_labels_are_excluded_from_global_token_loss(self):
        model = LabelLossModel()
        collected = []

        def all_reduce(value, **kwargs):
            collected.append(value.clone())
            # Rank 0 evaluated labels 1 and 3, contributing loss-sum 4/count 2.
            value.add_(torch.tensor([4.0, 2.0]))

        with patch("mask_fusion.train._all_ranks_true", side_effect=lambda value, ctx: bool(value)), \
             patch("torch.distributed.all_reduce", side_effect=all_reduce):
            loss = train.evaluate(model, SmallDataset(3), self.args(), 0, torch.device("cpu"), torch.float32, context(rank=1))
        self.assertEqual(loss, 2.0)
        self.assertEqual(collected[0].tolist(), [2.0, 1.0])
        self.assertEqual(model.labels_seen[0].tolist(), [[-100, 2], [-100, -100]])

    def test_all_padding_rank_still_forwards_without_all_ignored_nan(self):
        model = LabelLossModel()
        collected = []

        def all_reduce(value, **kwargs):
            collected.append(value.clone())
            value.add_(torch.tensor([1.0, 1.0]))

        with patch("mask_fusion.train._all_ranks_true", side_effect=lambda value, ctx: bool(value)), \
             patch("torch.distributed.all_reduce", side_effect=all_reduce):
            loss = train.evaluate(model, SmallDataset(1), self.args(), 0, torch.device("cpu"), torch.float32, context(rank=1, world_size=4))
        self.assertEqual(loss, 1.0)
        self.assertEqual(collected[0].tolist(), [0.0, 0.0])
        self.assertEqual(len(model.labels_seen), 1)
        self.assertTrue((model.labels_seen[0] != -100).any())


class DistributedControlTests(unittest.TestCase):
    def test_fsdp_clip_uses_root_method_instead_of_local_parameters(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        parameter.grad = torch.tensor(3.0)
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        model = SimpleNamespace(clip_grad_norm_=Mock(return_value=torch.tensor(3.0)))
        with patch("torch.nn.utils.clip_grad_norm_", side_effect=AssertionError("local clip must not run")):
            updated, *_ = train._optimizer_update(optimizer, scheduler, scaler, [parameter], 1.0, model=model, context=context())
        self.assertTrue(updated)
        model.clip_grad_norm_.assert_called_once_with(1.0)
        self.assertEqual(scheduler.last_epoch, 1)

    def test_help_never_checks_cuda_or_initializes_collectives(self):
        with patch("torch.cuda.is_available", side_effect=AssertionError("no CUDA check for help")), \
             patch("torch.distributed.init_process_group", side_effect=AssertionError("no group for help")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as error:
                train.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--distributed", output.getvalue())

    def test_non_main_print_and_filesystem_action_are_silent(self):
        action = Mock()
        with patch("torch.distributed.broadcast_object_list") as broadcast, contextlib.redirect_stdout(io.StringIO()) as output:
            train._rank_zero_print({"event": "hidden"}, context(1))
            train._rank_zero_action(action, context(1))
        action.assert_not_called()
        broadcast.assert_called_once()
        self.assertEqual(output.getvalue(), "")

    def test_rank_zero_io_failure_is_propagated_through_broadcast(self):
        with patch("torch.distributed.broadcast_object_list") as broadcast:
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                train._rank_zero_action(Mock(side_effect=OSError("disk full")), context())
        self.assertIn("disk full", broadcast.call_args.args[0][0])

    def test_rng_reads_only_the_local_cuda_device(self):
        rank_context = context(1)
        rank_context.device = torch.device("cuda:1")
        with patch("torch.cuda.get_rng_state", return_value=torch.tensor([1], dtype=torch.uint8)) as local, \
             patch("torch.cuda.get_rng_state_all", side_effect=AssertionError("must not touch all GPUs")):
            state = train._rng_state(rank_context)
        local.assert_called_once_with(rank_context.device)
        self.assertIn("cuda_local", state)

    def test_non_main_checkpoint_participates_in_all_gathers_but_never_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "not_created"
            optimizer = Mock()
            scheduler = Mock(state_dict=Mock(return_value={}))
            scaler = Mock(state_dict=Mock(return_value={}))
            with patch("mask_fusion.distributed.collect_adapter_state", return_value={}) as weights, \
                 patch("mask_fusion.distributed.collect_optimizer_state", return_value={}) as optim, \
                 patch("mask_fusion.distributed.gather_rank_rng", return_value=None) as rng, \
                 patch("mask_fusion.distributed.unwrap_model", side_effect=lambda value: value), \
                 patch("mask_fusion.train._rng_state", return_value={"rank": 1}), \
                 patch("mask_fusion.model.save_adapter") as save, \
                 patch("torch.distributed.broadcast_object_list"):
                train._save_checkpoint(object(), None, folder, SimpleNamespace(base_model="base"), optimizer, scheduler, scaler, {}, epoch=0, next_batch=2, global_step=1, best_val_loss=3.0, context=context(1))
            weights.assert_called_once()
            optim.assert_called_once()
            rng.assert_called_once_with({"rank": 1})
            save.assert_not_called()
            self.assertFalse(folder.exists())

    def test_full_optimizer_is_not_broadcast_during_resume(self):
        optimizer = {"large_optimizer": torch.ones(3)}
        state = {"optimizer": optimizer, "epoch": 1, "rng": [{"rank": 0}, {"rank": 1}]}
        with patch("mask_fusion.train._load_training_state", return_value=state), \
             patch("torch.distributed.broadcast_object_list") as broadcast:
            metadata, restored_optimizer = train._load_resume_state(Path("unused"), context())
        self.assertNotIn("optimizer", metadata)
        self.assertIs(restored_optimizer, optimizer)
        self.assertNotIn("optimizer", broadcast.call_args.args[0][1])


if __name__ == "__main__":
    unittest.main()
