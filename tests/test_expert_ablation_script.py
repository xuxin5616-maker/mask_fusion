"""Check all four ablation commands and sequential execution without a GPU."""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

from test_training_script import PROJECT, find_bash


SCRIPT = PROJECT / "scripts" / "test_projector_expert_ablation.sh"


class ExpertAblationScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = find_bash()
        if cls.bash is None:
            raise unittest.SkipTest("GNU Bash unavailable")

    def run_script(self, **changes):
        env = os.environ.copy()
        env.update({"DRY_RUN": "1", "PYTHON_BIN": "/missing python with spaces", "TEST_GPU": "6",
                    "JET_ADAPTER": "outputs/siglip_jet_v1/sft",
                    "CNN_ADAPTER": "outputs/mask_residual_mlp_sft_bf16_fa2_gpu6_fixroot/sft",
                    "JET_DTYPE": "float16", "CNN_DTYPE": "bfloat16", "SEED": "0", "NOISE_SEED": "0",
                    "RUN_ORIGINAL": "0", "VISUALIZE": "0", "OUTPUT_DIR": "results/ablation"})
        env.update(changes)
        return subprocess.run([self.bash, "--noprofile", "--norc", SCRIPT.as_posix()],
                              cwd=PROJECT, env=env, text=True, capture_output=True, timeout=20)

    def commands(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return [shlex.split(line[line.index("env CUDA_VISIBLE_DEVICES"):])
                for line in result.stdout.splitlines() if "env CUDA_VISIBLE_DEVICES=" in line]

    @staticmethod
    def argument(command, flag):
        return command[command.index(flag) + 1]

    def test_four_cases_keep_same_roi_seed_and_adapter_specific_dtype(self):
        commands = self.commands(self.run_script())
        self.assertEqual(len(commands), 4)
        self.assertEqual([self.argument(c, "--expert-input") for c in commands], ["zero", "random"] * 2)
        self.assertEqual([self.argument(c, "--dtype") for c in commands], ["float16"] * 2 + ["bfloat16"] * 2)
        for command in commands:
            self.assertIn("CUDA_VISIBLE_DEVICES=6", command)
            self.assertIn("mask_fusion.infer_expert_ablation", command)
            self.assertEqual(self.argument(command, "--seed"), "0")
            self.assertEqual(self.argument(command, "--noise-seed"), "0")
            self.assertEqual(self.argument(command, "--anomaly-prompt"), "category-mcq")
            self.assertNotIn("--overwrite", command)
            self.assertNotIn("--visualize", command)
            self.assertNotIn("--image-mode", command)

    def test_original_controls_and_visualizations_are_explicit_options(self):
        commands = self.commands(self.run_script(RUN_ORIGINAL="1", VISUALIZE="1", NOISE_SEED="7"))
        self.assertEqual(len(commands), 6)
        self.assertEqual([self.argument(c, "--expert-input") for c in commands], ["original", "zero", "random"] * 2)
        for command in commands:
            self.assertIn("--visualize", command)
        self.assertIn("random_seed7", self.argument(commands[2], "--output"))

    def test_real_execution_stops_on_failed_case_without_starting_other_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base").mkdir()
            (root / "test.jsonl").write_text("{}\n")
            for name in ("jet", "cnn"):
                (root / name).mkdir()
                (root / name / "adapter_config.json").write_text("{}")
                (root / name / "adapter.safetensors").write_bytes(b"stub")
            fake = root / "fake.sh"
            fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*"\nexit 23\n', encoding="utf-8", newline="\n")
            fake.chmod(0o755)
            result = self.run_script(DRY_RUN="0", PYTHON_BIN=fake.as_posix(), BASE_MODEL=(root / "base").as_posix(),
                                     TEST_DATA=(root / "test.jsonl").as_posix(), JET_ADAPTER=(root / "jet").as_posix(),
                                     CNN_ADAPTER=(root / "cnn").as_posix(), OUTPUT_DIR=(root / "results").as_posix())
            self.assertEqual(result.returncode, 23, result.stderr)
            self.assertTrue((root / "results/logs/jet_zero.log").is_file())
            self.assertFalse((root / "results/logs/jet_random_seed0.log").exists())


if __name__ == "__main__":
    unittest.main()
