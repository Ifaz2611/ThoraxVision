import json

from thoraxvision.class_info import (
    CANONICAL_CLASSES,
    canonical_class_name,
    class_metadata,
    describe_class,
    display_label,
    load_class_names,
)


def test_long_dataset_folder_names_map_to_canonical_classes():
    assert (
        canonical_class_name("adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib")
        == "adenocarcinoma"
    )
    assert (
        canonical_class_name("large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa")
        == "large.cell.carcinoma"
    )
    assert (
        canonical_class_name("squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa")
        == "squamous.cell.carcinoma"
    )
    assert canonical_class_name("normal") == "normal"


def test_unknown_folder_name_is_preserved():
    assert canonical_class_name("custom_class") == "custom_class"
    assert display_label("custom_class") == "Custom Class"
    assert describe_class("custom_class").startswith("Custom Class:")


def test_class_metadata_covers_every_class():
    metadata = class_metadata(CANONICAL_CLASSES)
    assert metadata["class_names"] == CANONICAL_CLASSES
    for name in CANONICAL_CLASSES:
        assert metadata["class_labels"][name]
        assert metadata["class_descriptions"][name]


def test_load_class_names_prefers_classes_file(tmp_path):
    classes_file = tmp_path / "model_classes.json"
    info_file = tmp_path / "model_info.json"
    classes_file.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    info_file.write_text(json.dumps({"classes": ["c", "d"]}), encoding="utf-8")
    assert load_class_names(classes_file, info_file) == ["a", "b"]


def test_load_class_names_falls_back_to_model_info(tmp_path):
    info_file = tmp_path / "model_info.json"
    info_file.write_text(json.dumps({"classes": ["c", "d"]}), encoding="utf-8")
    assert load_class_names(tmp_path / "missing.json", info_file) == ["c", "d"]


def test_load_class_names_uses_builtin_fallback(tmp_path):
    assert (
        load_class_names(tmp_path / "missing.json", tmp_path / "missing2.json")
        == CANONICAL_CLASSES
    )
