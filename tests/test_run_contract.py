import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_contract import build_contract, check_contract


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunContractTest(unittest.TestCase):
    def _workspace(self, root: Path, *, ready: bool) -> tuple[Path, Path, Path]:
        (root / "configs").mkdir()
        (root / "scripts").mkdir()
        (root / "data").mkdir()
        (root / "configs" / "official_sources.env.example").write_text("FLEXHOOK_COMMIT=abc\n", encoding="utf-8")
        (root / "scripts" / "preflight_baseline_assets.py").write_text("source\n", encoding="utf-8")
        (root / "scripts" / "run_contract.py").write_text("contract\n", encoding="utf-8")
        dataset = root / "data" / "manifest.json"
        dataset.write_text('{"dataset":"SportsMOT"}\n', encoding="utf-8")
        layout = root / "runs" / "layout"
        layout.mkdir(parents=True)
        (layout / "manifest.json").write_text(
            json.dumps({"inference_data_policy": "NO_GT_AS_TRACKER_INPUT"}), encoding="utf-8"
        )
        preflight = root / "preflight.json"
        names = ["flexhook_layout", "yestrack_layout", "roberta_base", "rope_swin_tiny", "flexhook_task_checkpoint", "yestrack_lora_checkpoint", "yestrack_base_model", "common_sports_tracker"]
        checks = []
        for name in names:
            status = "READY" if ready else ("READY" if name == "flexhook_layout" else "MISSING")
            checks.append({"name": name, "status": status, "kind": "directory", "path": str(layout)})
        preflight.write_text(
            json.dumps({"schema_version": 1, "overall_status": "READY" if ready else "PENDING", "pending_checks": [] if ready else names[2:], "checks": checks}),
            encoding="utf-8",
        )
        return preflight, dataset, layout

    def test_pending_contract_records_every_gate(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            preflight, dataset, layout = self._workspace(root, ready=False)
            contract = build_contract(
                workspace_root=root,
                method="flexhook",
                run_id="pending",
                preflight_path=preflight,
                dataset_manifest_path=dataset,
                layout_path=layout,
                protocol={
                    "seed": 42,
                    "train_split": "train",
                    "evaluation_split": "val",
                    "tracker_policy": "common",
                    "thresholds": {"score": 0.5},
                    "hyperparameters": {"epochs": 1},
                    "launch_command": ["echo", "future"],
                },
            )
            self.assertEqual(contract["status"], "PENDING")
            self.assertFalse(contract["ready_to_launch"])
            self.assertIn("preflight_pending:roberta_base", contract["blocking_reasons"])
            self.assertEqual(contract["policy"]["gt_or_labels_as_tracker_input"], "REJECTED")

    def test_ready_contract_hash_check_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            preflight, dataset, layout = self._workspace(root, ready=True)
            contract = build_contract(
                workspace_root=root,
                method="flexhook",
                run_id="ready",
                preflight_path=preflight,
                dataset_manifest_path=dataset,
                layout_path=layout,
                protocol={
                    "seed": 42,
                    "train_split": "train",
                    "evaluation_split": "val",
                    "tracker_policy": "common",
                    "thresholds": {"score": 0.5},
                    "hyperparameters": {"epochs": 1},
                    "launch_command": ["echo", "future"],
                },
            )
            self.assertTrue(contract["ready_to_launch"])
            output = root / "contract.json"
            output.write_text(json.dumps(contract), encoding="utf-8")
            _, errors = check_contract(output)
            self.assertEqual(errors, [])
            dataset.write_text("tampered\n", encoding="utf-8")
            _, errors = check_contract(output)
            self.assertTrue(any("hash_mismatch" in error for error in errors))

    def test_placeholder_protocol_cannot_be_launch_ready(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            preflight, dataset, layout = self._workspace(root, ready=True)
            contract = build_contract(
                workspace_root=root,
                method="flexhook",
                run_id="placeholder",
                preflight_path=preflight,
                dataset_manifest_path=dataset,
                layout_path=layout,
                protocol={
                    "seed": 42,
                    "train_split": "train",
                    "evaluation_split": "val",
                    "tracker_policy": "common",
                    "thresholds": {"score": "PENDING_UPSTREAM_DEFAULT"},
                    "hyperparameters": {"epochs": "TODO"},
                    "launch_command": ["TO_BE_FILLED"],
                },
            )
            self.assertFalse(contract["ready_to_launch"])
            self.assertIn("protocol_thresholds_missing", contract["blocking_reasons"])

    def test_sports_only_variant_ignores_optional_task_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            preflight, dataset, layout = self._workspace(root, ready=False)
            report = json.loads(preflight.read_text(encoding="utf-8"))
            for item in report["checks"]:
                if item["name"] in {"roberta_base", "rope_swin_tiny", "common_sports_tracker"}:
                    item["status"] = "READY"
            report["pending_checks"] = [
                name for name in report["pending_checks"]
                if name not in {"roberta_base", "rope_swin_tiny", "common_sports_tracker"}
            ]
            preflight.write_text(json.dumps(report), encoding="utf-8")
            contract = build_contract(
                workspace_root=root,
                method="flexhook",
                variant="f1_sports_trained",
                run_id="f1",
                preflight_path=preflight,
                dataset_manifest_path=dataset,
                layout_path=layout,
                protocol={
                    "seed": 42,
                    "train_split": "train",
                    "evaluation_split": "val",
                    "tracker_policy": "common",
                    "thresholds": {"score": 0.5},
                    "hyperparameters": {"epochs": 1},
                    "launch_command": ["echo", "future"],
                },
            )
            self.assertTrue(contract["ready_to_launch"])
            self.assertEqual(contract["variant"], "f1_sports_trained")
            self.assertNotIn("flexhook_task_checkpoint", contract["inputs"]["required_assets"])


if __name__ == "__main__":
    unittest.main()
