from tools.export_tflite_header import bytes_to_c_array, write_header


def test_bytes_to_c_array_format():
    header = bytes_to_c_array(bytes([0x00, 0x0F, 0xFF, 0x10]), "model_data",
                              bytes_per_line=2)
    assert "alignas(8) const unsigned char model_data[]" in header
    assert "0x00, 0x0f," in header
    assert "0xff, 0x10," in header
    assert "model_data_len = 4;" in header


def test_write_header_roundtrip(tmp_path):
    data = bytes(range(256)) * 3
    source = tmp_path / "m.tflite"
    source.write_bytes(data)
    out = tmp_path / "model_data.h"
    size = write_header(source, out, "model_data")
    assert size == len(data)
    text = out.read_text()
    assert f"model_data_len = {len(data)};" in text
