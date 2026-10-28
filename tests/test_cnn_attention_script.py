"""Run the fresh CNN attention-only launcher without models or GPUs."""

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest

from test_training_script import PROJECT, find_bash


SCRIPT = PROJECT / "scripts" / "run_cnn_attention_pipeline.sh"


class AttentionScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = find_bash()
        if cls.bash is None:
            raise unittest.SkipTest("GNU Bash unavailable")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project with spaces"
        (self.root / "scripts").mkdir(parents=True)
        self.script = self.root / "scripts" / SCRIPT.name
        source, count = re.subn(r'TRAIN_FILES=\(\n.*?\n\)',
                               'TRAIN_FILES=("train one.json" "train two.jsonl")',
                               SCRIPT.read_text(encoding="utf-8"), count=1, flags=re.S)
        self.assertEqual(count, 1)
        self.script.write_text(source, encoding="utf-8", newline="\n")
        self.env = os.environ.copy()
        for name in re.findall(r'^([A-Z_]+)=', source, re.M):
            self.env.pop(name, None)
        self.env.update({"DRY_RUN": "1", "PYTHON_BIN": "/missing python with spaces",
                         "PRETRAIN_OUTPUT": "/old adapter", "INIT_ADAPTER": "/old adapter",
                         "CUDA_VISIBLE_DEVICES": "0,1"})

    def run_script(self, **env):
        return subprocess.run([self.bash, "--noprofile", "--norc", self.script.as_posix()],
                              cwd=self.temp.name, env={**self.env, **env}, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=20)

    @staticmethod
    def argument(command, flag):
        return command[command.index(flag) + 1]

    def prepare_fake_python(self):
        for name in ("train one.json", "train two.jsonl", "test data.jsonl", "fusion.json", "zero3.json"):
            (self.root / name).write_text("{}", encoding="utf-8")
        (self.root / "base model").mkdir()
        (self.root / "calls").mkdir()
        fake = self.root / "fake python.sh"
        fake.write_text('''#!/usr/bin/env bash
set -eu
case " $* " in
    *" mask_fusion.train "*) stage=3 ;;
    *" mask_fusion.infer "*) stage=4 ;;
    *) exit 98 ;;
esac
printf '%s\\0' "${CUDA_VISIBLE_DEVICES:-}" "$PWD" "$@" > "calls/${stage}.args"
printf '%s\\n' "$stage" >> calls/order.txt
if [[ "${FAIL_STAGE:-}" == "$stage" ]]; then exit 23; fi
''', encoding="utf-8", newline="\n")
        fake.chmod(0o755)
        self.env.update({"DRY_RUN": "0", "PYTHON_BIN": "./fake python.sh", "BASE_MODEL": "base model",
                         "FUSION_CONFIG": "fusion.json", "DEEPSPEED_CONFIG": "zero3.json",
                         "TEST_DATA": "test data.jsonl"})

    def command(self, stage):
        return (self.root / "calls" / f"{stage}.args").read_bytes().decode().split("\0")[:-1]

    def test_dry_run_is_fresh_sft_without_synthetic_generation_or_adapter_initialization(self):
        before = sorted(p.relative_to(self.root) for p in self.root.rglob("*"))
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if " -m " in line]
        self.assertEqual(len(commands), 2)
        training, testing = commands
        self.assertIn("CUDA_VISIBLE_DEVICES=6", training)
        self.assertEqual(self.argument(training, "--distributed"), "none")
        self.assertEqual(self.argument(training, "--device"), "cuda:0")
        self.assertEqual(self.argument(training, "--dtype"), "float16")
        self.assertEqual(self.argument(training, "--attn-implementation"), "flash_attention_2")
        self.assertEqual(self.argument(training, "--fusion-config"), "configs/cnn_attention_only.json")
        self.assertEqual(self.argument(training, "--data-root"), "/home/yz/xxy/data/datasets")
        self.assertEqual(self.argument(training, "--training-stage"), "sft")
        self.assertNotIn("--init-adapter", training)
        self.assertNotIn("--resume", training)
        self.assertEqual(self.argument(training, "--output"), "outputs/cnn_attention_only_v1/sft")
        self.assertEqual(self.argument(testing, "--adapter"), self.argument(training, "--output"))
        self.assertEqual(self.argument(testing, "--visualization-limit"), "0")
        self.assertEqual(before, sorted(p.relative_to(self.root) for p in self.root.rglob("*")))

    def test_single_gpu_real_sequence_logs_and_multiple_manifests(self):
        self.prepare_fake_python()
        (self.root / "zero3.json").unlink()
        result = self.run_script(RUN_NAME="fresh run", TRAIN_GPUS="6", TEST_GPU="6")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "calls/order.txt").read_text().splitlines(), ["3", "4"])
        training = self.command(3)
        self.assertEqual(training[0], "6")
        self.assertEqual(training[2:5], ["-m", "mask_fusion.train", "--distributed"])
        index = training.index("--data")
        self.assertEqual(training[index + 1:index + 3], ["train one.json", "train two.jsonl"])
        self.assertNotIn("--deepspeed-config", training)
        self.assertTrue((self.root / "outputs/fresh run/logs/3_sft.log").exists())
        self.assertTrue((self.root / "outputs/fresh run/logs/4_test.log").exists())

    def test_multiple_gpu_ids_select_matching_zero3_process_count(self):
        self.prepare_fake_python()
        result = self.run_script(TRAIN_GPUS="2,3", TEST_GPU="3", DTYPE="bfloat16")
        self.assertEqual(result.returncode, 0, result.stderr)
        training = self.command(3)
        self.assertEqual(training[0], "2,3")
        self.assertIn("--nproc_per_node=2", training)
        self.assertEqual(self.argument(training, "--distributed"), "deepspeed")
        self.assertEqual(self.argument(training, "--dtype"), "bfloat16")
        self.assertEqual(self.command(4)[0], "3")

    def test_training_failure_stops_test_and_test_only_does_not_start_training(self):
        self.prepare_fake_python()
        failed = self.run_script(FAIL_STAGE="3")
        self.assertEqual(failed.returncode, 23)
        self.assertIn("3_sft failed", failed.stderr)
        self.assertFalse((self.root / "calls/4.args").exists())
        completed = self.run_script(START_STAGE="4", TRAIN_GPUS="", VISUALIZE="0")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual((self.root / "calls/order.txt").read_text().splitlines(), ["3", "4"])
        self.assertNotIn("--visualize", self.command(4))

    def test_old_pretraining_stage_and_invalid_gpu_selections_are_rejected(self):
        for change in ({"START_STAGE": "2"}, {"TRAIN_GPUS": "6,6"}, {"TRAIN_GPUS": "6,,7"},
                       {"TRAIN_GPUS": "6, 7"}, {"TEST_GPU": "6,7"}, {"START_STAGE": "4", "END_STAGE": "3"}):
            with self.subTest(change=change):
                self.assertNotEqual(self.run_script(**change).returncode, 0)


if __name__ == "__main__":
    unittest.main()
