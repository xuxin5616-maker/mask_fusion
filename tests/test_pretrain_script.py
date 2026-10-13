"""Check mask pretraining entry points without GPU training or downloads."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_training_script as helpers

from mask_fusion import pretrain


SCRIPT = helpers.PROJECT / "scripts" / "pretrain_mask_zero3_2gpu.sh"


class PretrainEntryTests(unittest.TestCase):
    def test_defaults_and_explicit_overrides_reach_shared_training_loop(self):
        arguments = ["--base-model", "base", "--data", "train.jsonl", "--output", "out",
                     "--projector-lr", "2e-4", "--dtype", "float32"]
        with patch("mask_fusion.train.main", return_value="shared loop") as train_main:
            self.assertEqual(pretrain.main(arguments), "shared loop")
        self.assertEqual(train_main.call_args.args[0], [
            "--training-stage", "mask_pretrain", "--image-mode", "square",
            "--projector-lr", "1e-4", *arguments,
        ])

    def test_sft_stage_cannot_override_pretraining_entry(self):
        for arguments in (["--training-stage", "sft"], ["--training-stage=sft"],
                          ["--training-stag=sft"],
                          ["--training-stage", "sft", "--training-stage", "mask_pretrain"]):
            with self.subTest(arguments=arguments), patch("mask_fusion.train.main") as train_main:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    pretrain.main(arguments)
                self.assertEqual(caught.exception.code, 2)
                train_main.assert_not_called()

    def test_help_exits_before_distributed_or_model_loading(self):
        with patch("mask_fusion.distributed.initialize_distributed", side_effect=AssertionError("distributed initialized")), \
                patch("mask_fusion.model.load_model", side_effect=AssertionError("model loaded")):
            with contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as caught:
                pretrain.main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--training-stage", output.getvalue())


class PretrainScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = helpers.find_bash()
        if cls.bash is None:
            raise unittest.SkipTest("GNU Bash is unavailable")

    def helper(self):
        helper = helpers.TrainingScriptTests()
        helper.bash = self.bash
        return helper

    def run_script(self, *arguments, **kwargs):
        env = {"SYNTHETIC_DATA_DIR": "", "DEEPSPEED_CONFIG": "", "FUSION_CONFIG": "",
               "LOG_EVERY": "", **kwargs.pop("env", {})}
        return self.helper().run_script(*arguments, script=SCRIPT, env=env, **kwargs)

    def test_syntax_and_default_two_gpu_command_without_existing_data(self):
        self.assertNotIn(b"\r", SCRIPT.read_bytes())
        checked = subprocess.run([self.bash, "-n", SCRIPT.as_posix()], capture_output=True, text=True, timeout=15)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        helper = self.helper()
        with tempfile.TemporaryDirectory() as directory:
            root, command = helper.command(self.run_script(cwd=directory))
        self.assertEqual(command[:9], [
            "CUDA_VISIBLE_DEVICES=0,1", "OMP_NUM_THREADS=1", "/not installed/python with spaces",
            "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "--module",
            "mask_fusion.pretrain",
        ])
        for option, expected in {
            "--distributed": "deepspeed", "--dtype": "bfloat16", "--image-mode": "square",
            "--projector-lr": "1e-4", "--fusion-config": root + "/configs/default.json",
            "--deepspeed-config": root + "/configs/zero3.json",
            "--output": root + "/outputs/mask_pretrain_zero3",
            "--data": root + "/data/synthetic_masks/train.jsonl",
            "--val-data": root + "/data/synthetic_masks/val.jsonl",
        }.items():
            self.assertEqual(helper.argument(command, option), expected, option)
        self.assertNotIn("--require-disjoint-categories", command)

    def test_multiple_cli_manifests_and_precision_override_preserve_spaces(self):
        helper = self.helper()
        arguments = ["--data", "/data/train one.json", "/data/train two.jsonl",
                     "--val-data", "/data/val one.jsonl", "/data/val two.json",
                     "--dtype", "float32", "--projector-lr", "2e-4"]
        _, command = helper.command(self.run_script(*arguments, env={
            "SYNTHETIC_DATA_DIR": "/ignored defaults", "DATA_ROOT": "/shared train root",
            "VAL_DATA_ROOT": "/shared val root", "BASE_MODEL": "/models/base with spaces",
            "OUTPUT_DIR": "/outputs/pretrain with spaces", "CUDA_VISIBLE_DEVICES": "2,3",
            "REQUIRE_DISJOINT_CATEGORIES": "1",
        }))
        self.assertEqual(command[-len(arguments):], arguments)
        self.assertEqual(command.count("--data"), 1)
        self.assertEqual(command.count("--val-data"), 1)
        self.assertFalse(any("/ignored defaults" in value for value in command))
        self.assertEqual(command[0], "CUDA_VISIBLE_DEVICES=2,3")
        self.assertEqual(helper.argument(command, "--data-root"), "/shared train root")
        self.assertEqual(helper.argument(command, "--val-data-root"), "/shared val root")
        self.assertEqual(helper.argument(command, "--output"), "/outputs/pretrain with spaces")
        self.assertNotIn("--require-disjoint-categories", command)

    def test_environment_synthetic_directory_and_equals_alias(self):
        helper = self.helper()
        _, defaults = helper.command(self.run_script(env={"SYNTHETIC_DATA_DIR": "/generated masks/run one"}))
        self.assertEqual(helper.argument(defaults, "--data"), "/generated masks/run one/train.jsonl")
        self.assertEqual(helper.argument(defaults, "--val-data"), "/generated masks/run one/val.jsonl")
        _, command = helper.command(self.run_script("--data=my train.json", "--validation=my val.jsonl"))
        self.assertNotIn("--data", command)
        self.assertNotIn("--val-data", command)
        self.assertEqual(command[-2:], ["--data=my train.json", "--validation=my val.jsonl"])

    def test_missing_default_data_fails_before_starting_python(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_script(env={"DRY_RUN": "0", "SYNTHETIC_DATA_DIR": directory})
        self.assertEqual(result.returncode, 2)
        self.assertIn("Synthetic manifest does not exist", result.stderr)
        self.assertIn("--data and --val-data", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_actual_help_without_manifests_or_cuda_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_script("--help", env={
                "DRY_RUN": "0", "SYNTHETIC_DATA_DIR": directory,
                "PYTHON_BIN": Path(sys.executable).as_posix(),
            })
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--training-stage", result.stdout)
        self.assertNotIn("Synthetic manifest does not exist", result.stderr)


if __name__ == "__main__":
    unittest.main()
