"""Exercise the Linux launcher with Bash, without importing Torch or using GPUs."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "train_fsdp_2gpu.sh"


def find_bash():
    candidates = [os.environ.get("BASH_BIN")]
    if os.name == "nt":
        candidates.extend([
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ])
    candidates.append(shutil.which("bash"))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            result = subprocess.run([candidate, "--version"], capture_output=True,
                                    text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and "GNU bash" in result.stdout:
            return candidate
    return None


class TrainingScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = find_bash()
        if cls.bash is None:
            raise unittest.SkipTest("No runnable GNU Bash; launcher execution has not been verified")

    def run_script(self, *arguments, script=SCRIPT, cwd=None, env=None):
        child_env = os.environ.copy()
        for key in ("BASE_MODEL", "PYTHON_BIN", "OUTPUT_DIR", "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS", "BATCH_SIZE", "GRAD_ACCUM", "EPOCHS", "LR",
                    "PROJECTOR_LR", "IMAGE_MODE", "ATTN_IMPLEMENTATION", "NUM_WORKERS",
                    "DATA_ROOT", "VAL_DATA_ROOT", "REQUIRE_DISJOINT_CATEGORIES"):
            child_env.pop(key, None)
        child_env.update({"DRY_RUN": "1", "PYTHON_BIN": "/not installed/python with spaces"})
        child_env.update(env or {})
        return subprocess.run(
            [self.bash, "--noprofile", "--norc", script.as_posix(), *arguments],
            cwd=cwd, env=child_env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=15,
        )

    def command(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.strip().splitlines()
        self.assertEqual(len(lines), 2, result.stdout)
        directory = shlex.split(lines[0])
        self.assertEqual(directory[0], "cd")
        return directory[1], shlex.split(lines[1])

    @staticmethod
    def argument(command, option):
        return command[command.index(option) + 1]

    def test_syntax_and_lf_line_endings(self):
        self.assertNotIn(b"\r", SCRIPT.read_bytes())
        result = subprocess.run([self.bash, "-n", SCRIPT.as_posix()],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_launch_and_multiple_cli_files_with_spaces_from_other_cwd(self):
        with TemporaryDirectory() as directory:
            root, command = self.command(self.run_script(
                "--data", "/datasets/train one.jsonl", "/datasets/train two.json",
                "--val-data", "/datasets/val one.jsonl", "/datasets/val two.json",
                cwd=directory,
            ))
        self.assertTrue(root.replace("\\", "/").endswith("/llava_mask_fusion"))
        self.assertEqual(command[:8], [
            "CUDA_VISIBLE_DEVICES=0,1", "OMP_NUM_THREADS=1",
            "/not installed/python with spaces", "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node=2", "--module",
        ])
        self.assertEqual(command[8], "mask_fusion.train")
        expected = {
            "--base-model": "/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov",
            "--distributed": "fsdp", "--dtype": "bfloat16", "--batch-size": "1",
            "--grad-accum": "4", "--epochs": "3", "--lr": "1e-4",
            "--projector-lr": "1e-5", "--image-mode": "anyres_max_9",
            "--attn-implementation": "sdpa", "--num-workers": "0",
            "--output": root + "/outputs/fsdp_2gpu",
        }
        for option, value in expected.items():
            self.assertEqual(self.argument(command, option), value, option)
        self.assertEqual(command[-6:], ["--data", "/datasets/train one.jsonl", "/datasets/train two.json",
                                      "--val-data", "/datasets/val one.jsonl", "/datasets/val two.json"])
        self.assertIn("--require-disjoint-categories", command)
        self.assertNotIn("--data-root", command)
        self.assertNotIn("--val-data-root", command)
        self.assertNotIn("--no-gradient-checkpointing", command)

    def test_optional_validation_and_environment_overrides(self):
        _, command = self.command(self.run_script("--data", "train.jsonl", env={
            "BASE_MODEL": "/models/base with spaces", "OUTPUT_DIR": "/outputs/my run",
            "CUDA_VISIBLE_DEVICES": "2,3", "GRAD_ACCUM": "7", "EPOCHS": "2",
            "LR": "2e-4", "PROJECTOR_LR": "2e-5", "DATA_ROOT": "/data/shared root",
            "VAL_DATA_ROOT": "/data/validation root", "OMP_NUM_THREADS": "2",
        }))
        self.assertEqual(command[:2], ["CUDA_VISIBLE_DEVICES=2,3", "OMP_NUM_THREADS=2"])
        for option, value in (("--base-model", "/models/base with spaces"), ("--output", "/outputs/my run"),
                              ("--grad-accum", "7"), ("--epochs", "2"), ("--lr", "2e-4"),
                              ("--projector-lr", "2e-5"), ("--data-root", "/data/shared root"),
                              ("--val-data-root", "/data/validation root")):
            self.assertEqual(self.argument(command, option), value)
        self.assertNotIn("--require-disjoint-categories", command)
        _, command = self.command(self.run_script(
            "--data=train.jsonl", "--validation=val with spaces.jsonl",
            "--no-gradient-checkpointing", env={"REQUIRE_DISJOINT_CATEGORIES": "0"},
        ))
        self.assertNotIn("--require-disjoint-categories", command)
        self.assertEqual(command[-3:], ["--data=train.jsonl", "--validation=val with spaces.jsonl",
                                      "--no-gradient-checkpointing"])

    def test_edited_arrays_and_cli_replacement(self):
        with TemporaryDirectory(prefix="fusion script ") as directory:
            script = Path(directory) / "project with spaces" / "scripts" / SCRIPT.name
            script.parent.mkdir(parents=True)
            source = SCRIPT.read_text(encoding="utf-8")
            source = source.replace("TRAIN_FILES=()", 'TRAIN_FILES=("default train one.jsonl" "default train two.json")')
            source = source.replace("VAL_FILES=()", 'VAL_FILES=("default validation one.jsonl" "default validation two.json")')
            with script.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(source)
            _, defaults = self.command(self.run_script(script=script, cwd=PROJECT))
            start = defaults.index("--data")
            self.assertEqual(defaults[start:start + 3], ["--data", "default train one.jsonl", "default train two.json"])
            start = defaults.index("--val-data")
            self.assertEqual(defaults[start:start + 3], ["--val-data", "default validation one.jsonl", "default validation two.json"])
            _, overridden = self.command(self.run_script(
                "--data", "replacement train.jsonl", "--val-data", "replacement val.jsonl",
                script=script,
            ))
            self.assertFalse(any("default " in argument for argument in overridden))
            self.assertEqual(overridden.count("--data"), 1)
            self.assertEqual(overridden.count("--val-data"), 1)

    def test_empty_train_array_fails_without_launching_python(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 2)
        self.assertIn("TRAIN_FILES", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
