"""Exercise orchestration with a fake Python executable; no models or GPUs."""

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest

import test_training_script as helpers


SCRIPT = helpers.PROJECT / "scripts" / "run_mask_pretrain_pipeline.sh"


class PipelineScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = helpers.find_bash()
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
        # Isolate the launcher configuration from variables on the test host.
        for name in re.findall(r'^([A-Z_]+)=', source, re.M):
            self.env.pop(name, None)
        self.env.update({"DRY_RUN": "1", "PYTHON_BIN": "/missing python with spaces",
                         "DATA_ROOT": "/unrelated inherited root", "CUDA_VISIBLE_DEVICES": "6,7"})

    def run_script(self, **env):
        return subprocess.run(
            [self.bash, "--noprofile", "--norc", self.script.as_posix()],
            cwd=self.temp.name, env={**self.env, **env}, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=20,
        )

    def prepare_fake_python(self):
        for filename in ("train one.json", "train two.jsonl", "test data.jsonl",
                         "fusion.json", "zero3.json", "synthetic.json"):
            (self.root / filename).write_text("{}", encoding="utf-8")
        (self.root / "base model").mkdir()
        (self.root / "calls").mkdir()
        fake = self.root / "fake python.sh"
        fake.write_text('''#!/usr/bin/env bash
set -eu
case " $* " in
    *" mask_fusion.synthetic_masks "*) stage=1 ;;
    *" mask_fusion.pretrain "*) stage=2 ;;
    *" mask_fusion.train "*) stage=3 ;;
    *" mask_fusion.infer "*) stage=4 ;;
    *) exit 98 ;;
esac
printf '%s\\0' "${CUDA_VISIBLE_DEVICES:-}" "$PWD" "$@" > "calls/${stage}.args"
printf '%s\\n' "$stage" >> calls/order.txt
printf 'fake stage %s\\n' "$stage"
if [[ "${FAIL_STAGE:-}" == "$stage" ]]; then exit 23; fi
''', encoding="utf-8", newline="\n")
        fake.chmod(0o755)
        self.env.update({"DRY_RUN": "0", "PYTHON_BIN": "./fake python.sh",
                         "BASE_MODEL": "base model", "TEST_DATA": "test data.jsonl",
                         "SYNTHETIC_CONFIG": "synthetic.json", "FUSION_CONFIG": "fusion.json",
                         "DEEPSPEED_CONFIG": "zero3.json"})

    def command(self, stage):
        return (self.root / "calls" / f"{stage}.args").read_bytes().decode().split("\0")[:-1]

    @staticmethod
    def argument(command, option):
        return command[command.index(option) + 1]

    def test_dry_run_with_spaces_has_no_side_effects_and_links_stages(self):
        before = sorted(p.relative_to(self.root) for p in self.root.rglob("*"))
        result = self.run_script(RUN_NAME="experiment one", TRAIN_GPUS="2,3", TEST_GPU="3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, sorted(p.relative_to(self.root) for p in self.root.rglob("*")))
        commands = [shlex.split(line) for line in result.stdout.splitlines()
                    if " -m " in line]
        self.assertEqual(len(commands), 4, result.stdout)
        generated, pretrain, sft, test = commands
        self.assertEqual(self.argument(pretrain, "--fusion-config"), "configs/residual_mlp.json")
        self.assertEqual(self.argument(sft, "--fusion-config"), "configs/residual_mlp.json")
        self.assertEqual(self.argument(pretrain, "--data-root"), self.argument(generated, "--output"))
        self.assertEqual(self.argument(pretrain, "--val-data-root"), self.argument(generated, "--output"))
        self.assertEqual(self.argument(sft, "--init-adapter"), self.argument(pretrain, "--output"))
        self.assertEqual(self.argument(test, "--adapter"), self.argument(sft, "--output"))
        self.assertIn("CUDA_VISIBLE_DEVICES=2,3", pretrain)
        self.assertIn("--nproc_per_node=2", pretrain)
        self.assertEqual(self.argument(pretrain, "--distributed"), "deepspeed")
        self.assertIn("CUDA_VISIBLE_DEVICES=3", test)
        self.assertNotIn("--image-mode", test)
        self.assertNotIn("--resume", sft)

    def test_actual_sequence_preserves_roots_gpu_arguments_and_logs(self):
        self.prepare_fake_python()
        result = self.run_script(TRAIN_GPUS="2,3", TEST_GPU="3",
                                 TRAIN_DATA_ROOT="real train root", TEST_DATA_ROOT="real test root")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "calls/order.txt").read_text().splitlines(), ["1", "2", "3", "4"])
        pretrain, sft, test = [self.command(i) for i in (2, 3, 4)]
        self.assertEqual([pretrain[0], sft[0], test[0]], ["2,3", "2,3", "3"])
        self.assertEqual(self.argument(sft, "--data-root"), "real train root")
        self.assertEqual(self.argument(test, "--data-root"), "real test root")
        start = sft.index("--data")
        self.assertEqual(sft[start + 1:start + 3], ["train one.json", "train two.jsonl"])
        self.assertEqual(self.argument(test, "--visualization-limit"), "0")
        self.assertEqual(self.argument(test, "--anomaly-prompt"), "category-mcq")
        self.assertEqual(self.argument(pretrain, "--output"), "outputs/mask_residual_mlp_v1/pretrain")
        self.assertEqual(self.argument(sft, "--init-adapter"), "outputs/mask_residual_mlp_v1/pretrain")
        self.assertEqual(self.argument(test, "--adapter"), "outputs/mask_residual_mlp_v1/sft")
        logs = self.root / "outputs/mask_residual_mlp_v1/logs"
        self.assertEqual(len(list(logs.glob("*.log"))), 4)
        self.assertIn("fake stage 2", (logs / "2_pretrain.log").read_text())

    def test_training_failure_stops_before_sft_and_test_and_preserves_exit_code(self):
        self.prepare_fake_python()
        result = self.run_script(FAIL_STAGE="2")
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual((self.root / "calls/order.txt").read_text().splitlines(), ["1", "2"])
        self.assertIn("2_pretrain failed", result.stderr)
        self.assertFalse((self.root / "calls/3.args").exists())

    def test_single_gpu_uses_plain_training_without_deepspeed_config(self):
        self.prepare_fake_python()
        result = self.run_script(START_STAGE="2", TRAIN_GPUS="2", TEST_GPU="2",
                                 DTYPE="bfloat16", DEEPSPEED_CONFIG="missing zero config.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "calls/order.txt").read_text().splitlines(), ["2", "3", "4"])
        for stage in (2, 3):
            command = self.command(stage)
            self.assertEqual(command[0], "2")
            self.assertEqual(command[2:4], ["-m", "mask_fusion.pretrain" if stage == 2 else "mask_fusion.train"])
            self.assertEqual(self.argument(command, "--distributed"), "none")
            self.assertEqual(self.argument(command, "--device"), "cuda:0")
            self.assertEqual(self.argument(command, "--dtype"), "bfloat16")
            self.assertNotIn("torch.distributed.run", command)
            self.assertNotIn("--deepspeed-config", command)
        self.assertEqual(self.command(4)[0], "2")

    def test_process_count_matches_selected_gpus(self):
        for gpu_ids, count in (("3", 1), ("1,3", 2), ("0,2,3", 3)):
            with self.subTest(gpu_ids=gpu_ids):
                result = self.run_script(START_STAGE="2", END_STAGE="3", TRAIN_GPUS=gpu_ids)
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = [shlex.split(line) for line in result.stdout.splitlines() if " -m " in line]
                self.assertEqual(len(commands), 2)
                for command in commands:
                    self.assertIn("CUDA_VISIBLE_DEVICES=" + gpu_ids, command)
                    if count == 1:
                        self.assertNotIn("torch.distributed.run", command)
                    else:
                        self.assertIn("--nproc_per_node=" + str(count), command)
                        self.assertEqual(self.argument(command, "--distributed"), "deepspeed")

    def test_stage_selection_and_visualization_disable(self):
        self.prepare_fake_python()
        result = self.run_script(START_STAGE="3", END_STAGE="4", VISUALIZE="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "calls/order.txt").read_text().splitlines(), ["3", "4"])
        self.assertNotIn("--visualization-dir", self.command(4))
        self.assertNotIn("--visualize", self.command(4))

    def test_invalid_ranges_and_gpu_selection_fail_before_writing(self):
        for env in ({"START_STAGE": "0"}, {"END_STAGE": "5"}, {"START_STAGE": "word"},
                    {"START_STAGE": "4", "END_STAGE": "2"}, {"TRAIN_GPUS": ",0"},
                    {"TRAIN_GPUS": "0,"}, {"TRAIN_GPUS": "0,,1"}, {"TRAIN_GPUS": "0, 1"},
                    {"TRAIN_GPUS": "0,0"}, {"TRAIN_GPUS": "0,2,0"}, {"TEST_GPU": "0,1"}):
            with self.subTest(env=env):
                result = self.run_script(**env)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("[1/4]", result.stdout)
        self.assertFalse((self.root / "outputs").exists())

    def test_syntax_and_lf(self):
        self.assertNotIn(b"\r", SCRIPT.read_bytes())
        result = subprocess.run([self.bash, "-n", SCRIPT.as_posix()], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
