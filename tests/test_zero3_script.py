"""Execute only Bash dry runs; no CUDA or DeepSpeed installation is needed."""

from pathlib import Path
import subprocess
import tempfile
import unittest

import test_training_script as helpers


SCRIPT = helpers.PROJECT / "scripts" / "train_zero3_2gpu.sh"


class ZeroScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = helpers.find_bash()
        if cls.bash is None:
            raise unittest.SkipTest("GNU Bash is unavailable")

    def helper(self):
        helper = helpers.TrainingScriptTests()
        helper.bash = self.bash
        return helper

    def test_syntax_defaults_and_multiple_manifests(self):
        self.assertNotIn(b"\r", SCRIPT.read_bytes())
        checked = subprocess.run([self.bash, "-n", SCRIPT.as_posix()], capture_output=True, text=True, timeout=15)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        helper = self.helper()
        with tempfile.TemporaryDirectory() as cwd:
            root, command = helper.command(helper.run_script(
                "--data", "train one.json", "train two.jsonl", "--val-data", "val.jsonl",
                script=SCRIPT, cwd=cwd, env={"DATA_ROOT": "/data/with spaces"},
            ))
        self.assertEqual(helper.argument(command, "--distributed"), "deepspeed")
        self.assertEqual(helper.argument(command, "--deepspeed-config"), root + "/configs/zero3.json")
        self.assertEqual(helper.argument(command, "--fusion-config"), root + "/configs/no_hidden_norm.json")
        self.assertEqual(helper.argument(command, "--output"), root + "/outputs/zero3_2gpu_no_hidden_norm")
        self.assertEqual(helper.argument(command, "--data-root"), "/data/with spaces")
        self.assertIn("--nproc_per_node=2", command)
        self.assertIn("--require-disjoint-categories", command)
        self.assertEqual(command[-5:], ["--data", "train one.json", "train two.jsonl", "--val-data", "val.jsonl"])

    def test_edited_arrays_and_cli_overrides(self):
        helper = self.helper()
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "project with spaces" / "scripts" / SCRIPT.name
            script.parent.mkdir(parents=True)
            source = SCRIPT.read_text(encoding="utf-8").replace("TRAIN_FILES=()", 'TRAIN_FILES=("default a.json" "default b.json")')
            script.write_text(source, encoding="utf-8", newline="\n")
            _, command = helper.command(helper.run_script(script=script))
            self.assertEqual(command[-3:], ["--data", "default a.json", "default b.json"])
            _, overridden = helper.command(helper.run_script(
                "--data", "replacement.jsonl", "--fusion-config", "my config.json",
                script=script, env={"LOG_EVERY": "1"},
            ))
            self.assertFalse(any("default " in value for value in overridden))
            self.assertEqual(overridden[-2:], ["--fusion-config", "my config.json"])
            self.assertEqual(helper.argument(overridden, "--log-every"), "1")

    def test_no_manifests_reports_error_without_starting_python(self):
        result = self.helper().run_script(script=SCRIPT)
        self.assertEqual(result.returncode, 2)
        self.assertIn("TRAIN_FILES", result.stderr)


if __name__ == "__main__":
    unittest.main()
