import unittest

from mask_fusion.progress import ProgressTimer, format_duration


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0

    def timer(self, total, completed=0):
        return ProgressTimer(total, completed, clock=lambda: self.now)

    def test_resume_rate_counts_only_new_work(self):
        timer = self.timer(100, completed=80)
        self.assertIsNone(timer.snapshot()["eta_seconds"])
        self.now += 20
        progress = timer.snapshot(85)
        self.assertEqual(progress["eta"], "00:01:00")
        self.assertEqual(progress["units_per_second"], 0.25)
        self.now += 60
        self.assertEqual(timer.snapshot(100)["eta_seconds"], 0)

    def test_validation_and_checkpoint_pause_do_not_inflate_train_eta(self):
        timer = self.timer(10)
        self.now += 10
        before = timer.snapshot(2)
        with timer.paused():
            self.now += 100
            self.assertEqual(timer.snapshot(), before)
            with timer.paused():
                self.now += 20
        self.now += 10
        after = timer.snapshot(4)
        self.assertEqual(after["elapsed_seconds"], 20)
        self.assertEqual(after["eta_seconds"], 30)

    def test_duration_over_one_day_and_zero_work(self):
        self.assertEqual(format_duration(90061.1), "25:01:02")
        self.assertEqual(self.timer(0).snapshot()["eta"], "00:00:00")
        timer = self.timer(2)
        self.assertIsNone(timer.snapshot(1)["eta_seconds"])
        with self.assertRaises(ValueError):
            timer.snapshot(0)
        with self.assertRaises(ValueError):
            timer.snapshot(3)


if __name__ == "__main__":
    unittest.main()
