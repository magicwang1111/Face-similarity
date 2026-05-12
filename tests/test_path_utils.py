from __future__ import annotations

import unittest

from face_lora_eval.path_utils import normalize_platform_path_text


class PathUtilsTests(unittest.TestCase):
    def test_windows_converts_wsl_mount_path(self) -> None:
        self.assertEqual(
            normalize_platform_path_text("/mnt/e/Face-similarity", platform_name="nt"),
            "E:\\Face-similarity",
        )

    def test_posix_converts_windows_drive_path(self) -> None:
        self.assertEqual(
            normalize_platform_path_text(r"D:\测试素材\foo", platform_name="posix"),
            "/mnt/d/测试素材/foo",
        )


if __name__ == "__main__":
    unittest.main()
