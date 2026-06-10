# dcase22_adapter의 파일명 파싱·분할·payload 생성을 합성 WAV fixture로 검증하는 단위 테스트
import os
import sys
import tempfile
import unittest
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dcase22_adapter as ad


def _write_wav(path, seconds=0.3, sr=16000, freq=440.0):
    t = np.arange(int(sr * seconds)) / sr
    y = (0.2 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())


def _make_fixture(root):
    """ToyCar section 00 source: train normal 12 (car_A1/car_A2), test normal 4, anomaly 4."""
    train = os.path.join(root, "ToyCar", "train")
    test = os.path.join(root, "ToyCar", "test")
    os.makedirs(train)
    os.makedirs(test)
    n = 0
    for car in ("A1", "A2"):
        for i in range(6):
            _write_wav(os.path.join(
                train, f"section_00_source_train_normal_{n:04d}_car_{car}_spd_28V.wav"),
                freq=300 + 40 * n)
            n += 1
    for i in range(2):
        for car in ("A1", "A2"):
            _write_wav(os.path.join(
                test, f"section_00_source_test_normal_{i:04d}_car_{car}_spd_28V.wav"))
            _write_wav(os.path.join(
                test, f"section_00_source_test_anomaly_{i:04d}_car_{car}_spd_28V.wav"))
    # 무시되어야 하는 파일들: target 도메인, 다른 섹션
    _write_wav(os.path.join(train, "section_00_target_train_normal_0000_car_B1_spd_28V.wav"))
    _write_wav(os.path.join(test, "section_01_source_test_normal_0000_car_C1_spd_28V.wav"))


class Dcase22AdapterTest(unittest.TestCase):
    def test_parse_filename_and_pseudo_id(self):
        t = ad.parse_filename("section_02_target_test_anomaly_0013_car_B7_spd_31V.wav")
        self.assertEqual(t["section"], "02")
        self.assertEqual(t["domain"], "target")
        self.assertEqual(t["label"], "anomaly")
        self.assertEqual(ad.pseudo_machine_id(t), "car_B7")
        self.assertIsNone(ad.parse_filename("not_a_dcase_file.wav"))

    def test_load_filters_section_and_domain(self):
        with tempfile.TemporaryDirectory() as root:
            _make_fixture(root)
            feats, labels, tags, meta = ad.load_dcase22_features(root, "ToyCar", "00")
            self.assertEqual(int((labels == 0).sum()), 16)  # 12 train + 4 test normals
            self.assertEqual(int((labels == 1).sum()), 4)
            self.assertEqual(int((tags == "train").sum()), 12)
            self.assertTrue(all(m["domain"] == "source" for m in meta))
            self.assertEqual(sorted({m["machine_id"] for m in meta}), ["car_A1", "car_A2"])

    def test_make_site_data_payload_shapes(self):
        with tempfile.TemporaryDirectory() as root:
            _make_fixture(root)
            config = {"dataset": "dcase2022", "data_root": root,
                      "machine_type": "ToyCar", "section": "00"}
            payloads = ad.make_dcase22_site_data(
                config, num_sites=3, n_mels=32, n_frames=8, alpha=100.0,
                seed=0, device="cpu")
            self.assertEqual(len(payloads), 3)
            total_train = sum(p["normal_train"].shape[0] for p in payloads)
            total_test = sum(p["normal_test"].shape[0] for p in payloads)
            total_anom = sum(p["anomaly_test"].shape[0] for p in payloads)
            self.assertEqual(total_train, 12)
            self.assertEqual(total_test, 4)
            self.assertEqual(total_anom, 4)
            for p in payloads:
                if p["normal_train"].shape[0]:
                    self.assertEqual(tuple(p["normal_train"].shape[1:]), (1, 32, 8))


if __name__ == "__main__":
    unittest.main()
