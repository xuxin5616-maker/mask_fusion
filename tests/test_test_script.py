"""Exercise the MVTec inference launcher without loading a model."""

import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory
import unittest

from test_training_script import find_bash


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "test_mvtec.sh"


class MVTecTestScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = find_bash()
        if cls.bash is None:
            raise unittest.SkipTest("No runnable GNU Bash")

    def dry_run(self, env=None, arguments=()):
        child_env = os.environ.copy()
        for key in ("DATA_ROOT", "TEST_DATA", "ADAPTER", "BASE_MODEL", "ANOMALY_PROMPT",
                    "OUTPUT_FILE", "PYTHON_BIN", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                    "DTYPE", "ATTN_IMPLEMENTATION", "SEED", "MAX_NEW_TOKENS", "MAX_TEXT_LENGTH"):
            child_env.pop(key, None)
        child_env.update({"DRY_RUN": "1", "PYTHON_BIN": "/missing/python with spaces"})
        child_env.update(env or {})
        with TemporaryDirectory() as cwd:
            result = subprocess.run([self.bash, "--noprofile", "--norc", SCRIPT.as_posix(), *arguments],
                                    cwd=cwd, env=child_env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.strip().splitlines()
        self.assertEqual(len(lines), 2)
        return shlex.split(lines[0])[1], shlex.split(lines[1])

    def test_default_script_launches_knowledge_inference_and_separate_output(self):
        syntax = subprocess.run([self.bash, "-n", SCRIPT.as_posix()], capture_output=True, text=True, timeout=15)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        self.assertNotIn(b"\r", SCRIPT.read_bytes())
        root, command = self.dry_run()
        self.assertEqual(command[:5], ["CUDA_VISIBLE_DEVICES=0", "OMP_NUM_THREADS=1",
                                       "/missing/python with spaces", "-m", "mask_fusion.infer"])
        expected = {
            "--eval-task": "anomaly", "--anomaly-prompt": "category-knowledge-mcq",
            "--adapter": root + "/outputs/fsdp_2gpu", "--device": "cuda:0", "--dtype": "bfloat16",
            "--output": root + "/results/mvtec_category_knowledge_mcq.jsonl",
            "--data-root": "/home/yz/xxy/data/datasets",
            "--data": "/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc_muscv2_classnorm_group_score.jsonl",
        }
        for key, value in expected.items():
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertNotIn("--overwrite", command)

    def test_environment_overrides_paths_with_spaces_and_baseline_output(self):
        root, command = self.dry_run({
            "DATA_ROOT": "/data/my dataset", "ADAPTER": "/models/my adapter", "CUDA_VISIBLE_DEVICES": "1",
            "ANOMALY_PROMPT": "category-mcq",
        }, arguments=("--max-new-tokens", "128"))
        self.assertEqual(command[0], "CUDA_VISIBLE_DEVICES=1")
        self.assertEqual(command[command.index("--adapter") + 1], "/models/my adapter")
        self.assertEqual(command[command.index("--data-root") + 1], "/data/my dataset")
        self.assertTrue(command[command.index("--data") + 1].startswith("/data/my dataset/"))
        self.assertEqual(command[command.index("--anomaly-prompt") + 1], "category-mcq")
        self.assertEqual(command[command.index("--output") + 1], root + "/results/mvtec_category_mcq.jsonl")
        self.assertEqual(command[-2:], ["--max-new-tokens", "128"])


if __name__ == "__main__":
    unittest.main()
