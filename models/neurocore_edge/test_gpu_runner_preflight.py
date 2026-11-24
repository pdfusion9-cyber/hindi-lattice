from gpu_runner_preflight import assess_profile, parse_gpu_rows


def test_parses_nvidia_smi_csv():
    rows = parse_gpu_rows("0, Tesla T4, 15360\n")
    assert rows == [{"index": "0", "name": "Tesla T4", "memory_total_mib": 15360}]


def test_accepts_single_qualifying_gpu_runner():
    report = assess_profile(
        runner_id=7,
        cpu_count=4,
        ram_bytes=28 * 1024**3,
        disk_total_bytes=176 * 10**9,
        disk_free_bytes=140 * 10**9,
        gpus=[{"index": "0", "name": "Tesla T4", "memory_total_mib": 15360}],
    )
    assert report["status"] == "GPU_RUNNER_PREFLIGHT_PASS"
    assert report["training_started"] is False


def test_blocks_standard_cpu_runner():
    report = assess_profile(
        runner_id=0,
        cpu_count=4,
        ram_bytes=16 * 1024**3,
        disk_total_bytes=14 * 10**9,
        disk_free_bytes=8 * 10**9,
        gpus=[],
    )
    assert report["status"] == "GPU_RUNNER_PREFLIGHT_BLOCKED"
    assert len(report["blockers"]) >= 3


def test_blocks_gpu_with_insufficient_vram():
    report = assess_profile(
        runner_id=19,
        cpu_count=8,
        ram_bytes=32 * 1024**3,
        disk_total_bytes=300 * 10**9,
        disk_free_bytes=200 * 10**9,
        gpus=[{"index": "0", "name": "small GPU", "memory_total_mib": 8192}],
    )
    assert report["status"] == "GPU_RUNNER_PREFLIGHT_BLOCKED"
    assert "VRAM" in report["blockers"][-1]
