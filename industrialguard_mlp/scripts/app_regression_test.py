"""Exercise batch upload, product routing, OOD fallback, and manual override."""
from __future__ import annotations

import io
import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "app_regression_test.json"


def upload_tuple(name: str, data: bytes, mime: str) -> tuple[str, bytes, str]:
    return name, data, mime


def file_tuple(path: Path) -> tuple[str, bytes, str]:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return upload_tuple(path.name, path.read_bytes(), mime)


def analyze(app: AppTest, files: list[tuple[str, bytes, str]]) -> list[dict]:
    app.file_uploader[0].set_value(files).run()
    # Analysis is explicitly user-triggered; uploading alone must not produce results.
    assert "analysis_records" not in app.session_state
    assert not any("Identified as" in item.value for item in app.success)
    app.button[0].click().run()
    assert not app.exception, [item.message for item in app.exception]
    return app.session_state["analysis_records"]


def main() -> None:
    person = Path(r"C:\Users\princ\Downloads\me.jpg")
    noise = np.random.default_rng(73).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(noise).save(buffer, format="PNG")
    noise_tuple = upload_tuple("random_noise.png", buffer.getvalue(), "image/png")

    test_cases = [
        ("normal_cable", ROOT / "datasets/mvtec_ad/cable/test/good/000.png", "cable"),
        ("defective_cable", ROOT / "datasets/mvtec_ad/cable/test/bent_wire/000.png", "cable"),
        ("capsule", ROOT / "datasets/mvtec_ad/capsule/test/good/000.png", "capsule"),
    ]
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    assert not app.exception, [item.message for item in app.exception]
    results: list[dict] = []

    for name, path, expected_category in test_cases:
        record = analyze(app, [file_tuple(path)])[0]
        actual = record.get("result", {})
        passed = "error" not in record and actual.get("category") == expected_category
        results.append({
            "case": name,
            "status": "PASS" if passed else "FAIL",
            "expected_category": expected_category,
            "observed_category": actual.get("category"),
            "route": actual.get("route"),
            "category_score": actual.get("confidence"),
            "anomaly_score": actual.get("anomaly_score"),
            "anomaly_flag": actual.get("anomaly_flag"),
            "damage_model_loaded": actual.get("defect_screen") is not None,
            "damage_screen": ({"score": actual["defect_screen"]["image_score"],
                               "threshold": actual["defect_screen"]["image_threshold"],
                               "flagged": actual["defect_screen"]["flagged"],
                               "highlighted_area_fraction": actual["defect_screen"]["area_fraction"]}
                              if actual.get("defect_screen") is not None else None),
            "error": record.get("error"),
        })

    if person.exists():
        record = analyze(app, [file_tuple(person)])[0]
        actual = record.get("result", {})
        passed = "error" not in record and actual.get("category") is None
        results.append({"case": "person_photo", "status": "PASS" if passed else "FAIL",
                        "observed_category": actual.get("category"),
                        "general_label": actual.get("general_label"),
                        "error": record.get("error")})
    else:
        results.append({"case": "person_photo", "status": "SKIP", "reason": f"missing image: {person}"})

    record = analyze(app, [noise_tuple])[0]
    actual = record.get("result", {})
    passed = "error" not in record and actual.get("category") is None
    results.append({"case": "random_noise", "status": "PASS" if passed else "FAIL",
                    "observed_category": actual.get("category"),
                    "general_label": actual.get("general_label"),
                    "error": record.get("error")})

    consumer = Path(r"D:\pic2.jpg")
    if consumer.exists():
        record = analyze(app, [file_tuple(consumer)])[0]
        actual = record.get("result", {})
        passed = "error" not in record and actual.get("category") == "cable"
        results.append({"case": "consumer_charging_cable", "status": "PASS" if passed else "FAIL",
                        "expected_category": "cable", "observed_category": actual.get("category"),
                        "route": actual.get("route"), "label_match_score": actual.get("confidence"),
                        "mlp_nearest_category": actual.get("mlp_category"),
                        "damage_model_loaded": actual.get("defect_screen") is not None,
                        "damage_screen": ({"score": actual["defect_screen"]["image_score"],
                                           "threshold": actual["defect_screen"]["image_threshold"],
                                           "flagged": actual["defect_screen"]["flagged"],
                                           "highlighted_area_fraction": actual["defect_screen"]["area_fraction"],
                                           "candidate_region_count": len(actual["defect_screen"]["regions"]),
                                           "exploratory": not actual.get("domain_validated", False)}
                                          if actual.get("defect_screen") is not None else None),
                        "damage_type_clues": actual.get("damage_type_clues"),
                        "error": record.get("error")})

        # Verify that the manual category control overrides automatic routing
        # and that the persisted report record uses the manual result.
        # Industry focus is the first selectbox; the per-image category override is next.
        app.selectbox[1].set_value("cable").run()
        manual = app.session_state["analysis_records"][0]["result"]
        passed = manual.get("route") == "Manual selection" and manual.get("category") == "cable"
        results.append({"case": "manual_category_override", "status": "PASS" if passed else "FAIL",
                        "observed_category": manual.get("category"), "route": manual.get("route")})
        app.selectbox[1].set_value("other_product").run()
        other = app.session_state["analysis_records"][0]["result"]
        passed = (other.get("category") == "other_product"
                  and other.get("route") == "Manual selection"
                  and other.get("defect_screen") is not None)
        results.append({"case": "other_product_manual_damage_screen", "status": "PASS" if passed else "FAIL",
                        "category": other.get("category"), "route": other.get("route"),
                        "damage_model_ran": other.get("defect_screen") is not None})
        app.text_input[0].set_value("consumer charging cable").run()
        named_other = app.session_state["analysis_records"][0]["result"]
        passed = (named_other.get("category") == "consumer charging cable"
                  and named_other.get("route") == "Manual selection"
                  and not named_other.get("reference_available"))
        results.append({"case": "other_product_custom_name", "status": "PASS" if passed else "FAIL",
                        "category": named_other.get("category"),
                        "reference_available": named_other.get("reference_available")})
        app.selectbox[1].set_value("Automatic detection").run()
        automatic = app.session_state["analysis_records"][0]["result"]
        passed = (automatic.get("category") == "cable"
                  and automatic.get("route") in {"Industrial MLP", "Open-vocabulary match",
                                                  "Open-vocabulary product match"})
        results.append({"case": "automatic_selection_restores_auto_result",
                        "status": "PASS" if passed else "FAIL",
                        "observed_category": automatic.get("category"), "route": automatic.get("route")})

    etid_cable = sorted((ROOT / "datasets/additional/etid/cable").glob("*.jpg"))
    if etid_cable:
        manifest = {row["path"] for row in __import__("csv").DictReader(
            (ROOT / "datasets/additional/etid/manifest.csv").open(encoding="utf-8"))
                    if row["category"] == "cable" and row["split"] == "test"}
        test_image = next((ROOT / p for p in sorted(manifest)), None)
        if test_image and test_image.is_file():
            record = analyze(app, [file_tuple(test_image)])[0]
            actual = record.get("result", {})
            passed = ("error" not in record and actual.get("category") == "cable"
                      and actual.get("route") in {"Industrial MLP", "Open-vocabulary match"})
            results.append({"case": "ETID_group_held_out_cable_routing", "status": "PASS" if passed else "FAIL",
                            "observed_category": actual.get("category"), "route": actual.get("route"),
                            "image": str(test_image.relative_to(ROOT))})

    batch_paths = [ROOT / "datasets/mvtec_ad/cable/test/good/000.png",
                   ROOT / "datasets/mvtec_ad/capsule/test/good/000.png"]
    batch = analyze(app, [file_tuple(path) for path in batch_paths])
    batch_categories = [entry.get("result", {}).get("category") for entry in batch]
    passed = batch_categories == ["cable", "capsule"]
    results.append({"case": "multi_image_batch", "status": "PASS" if passed else "FAIL",
                    "expected_categories": ["cable", "capsule"],
                    "observed_categories": batch_categories,
                    "analyzed_count": len(batch)})

    bottle_test_dir = ROOT / "datasets/additional/Plastic Shampoo Bottle Dataset/test/images"
    bottle_samples = sorted(bottle_test_dir.glob("*.jpg"))
    if bottle_samples:
        record = analyze(app, [file_tuple(bottle_samples[0])])[0]
        automatic_bottle = record.get("result", {})
        auto_pass = ("error" not in record
                     and automatic_bottle.get("category") == "plastic_shampoo_bottle"
                     and automatic_bottle.get("route") == "Industrial MLP")
        results.append({"case": "plastic_shampoo_bottle_automatic_category",
                        "status": "PASS" if auto_pass else "FAIL",
                        "expected_category": "plastic_shampoo_bottle",
                        "observed_category": automatic_bottle.get("category"),
                        "route": automatic_bottle.get("route"),
                        "category_confidence": automatic_bottle.get("confidence")})
        app.selectbox[1].set_value("plastic_shampoo_bottle").run()
        manual_bottle = app.session_state["analysis_records"][0]["result"]
        passed = (manual_bottle.get("route") == "Manual selection"
                  and manual_bottle.get("defect_screen") is not None
                  and manual_bottle.get("bottle_damage") is not None
                  and manual_bottle["bottle_damage"].get("source") == "Bottle-specific YOLO defect detector")
        results.append({"case": "plastic_shampoo_bottle_yolo_integration",
                        "status": "PASS" if passed else "FAIL",
                        "category": manual_bottle.get("category"),
                        "localization_model_loaded": manual_bottle.get("defect_screen") is not None,
                        "bottle_yolo_loaded": (manual_bottle.get("bottle_damage") or {}).get("source") == "Bottle-specific YOLO defect detector",
                        "predicted_type_grades": (manual_bottle.get("bottle_damage") or {}).get("predictions", [])[:3]})
        class_line = next(line for line in (ROOT / "datasets/additional/Plastic Shampoo Bottle Dataset/data.yaml")
                          .read_text(encoding="utf-8").splitlines() if line.strip().startswith("names:"))
        class_names = ast.literal_eval(class_line.split(":", 1)[1].strip())
        label_file = ROOT / "datasets/additional/Plastic Shampoo Bottle Dataset/test/labels" / f"{bottle_samples[0].stem}.txt"
        actual_labels = sorted({class_names[int(line.split()[0])]
                                for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()})
        predicted_labels = [item["label"] for item in (manual_bottle.get("bottle_damage") or {}).get("predictions", [])]
        matched_labels = sorted(set(actual_labels) & set(predicted_labels))
        results.append({"case": "bottle_yolo_test_sample_label_match", "status": "PASS" if matched_labels else "FAIL",
                        "ground_truth_labels": actual_labels, "predicted_labels": predicted_labels,
                        "matched_labels": matched_labels})
        app.selectbox[1].set_value("bottle").run()
        generic_bottle = app.session_state["analysis_records"][0]["result"]
        generic_pass = (generic_bottle.get("category") == "bottle"
                        and generic_bottle.get("bottle_damage") is None)
        results.append({"case": "generic_bottle_not_routed_to_shampoo_yolo",
                        "status": "PASS" if generic_pass else "FAIL",
                        "category": generic_bottle.get("category"),
                        "yolo_detector_ran": generic_bottle.get("bottle_damage") is not None})

    report = {"app": "IndustrialGuard AI", "test_method": "Streamlit AppTest against app.py",
              "model_2_loaded": bool(results and results[0].get("damage_model_loaded")),
              "cases": results}
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any(item["status"] == "FAIL" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
