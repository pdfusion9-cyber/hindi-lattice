#!/usr/bin/env python3
"""Local/public-runner probe only; this does not start or train the model."""
import json, os, shutil
def main():
    ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    disk = shutil.disk_usage(".").free
    report = {"cpu_count": os.cpu_count(), "ram_gib": round(ram / 2**30, 2),
              "free_disk_gib": round(disk / 2**30, 2), "device": "cpu",
              "training_started": False}
    try:
        import torch
        report["torch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
    except Exception as exc:
        report["torch_import_error"] = str(exc)
        print(json.dumps(report, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    return 0 if (os.cpu_count() or 0) >= 4 and ram >= 12 * 2**30 and disk >= 6 * 2**30 else 2
if __name__ == "__main__":
    raise SystemExit(main())
