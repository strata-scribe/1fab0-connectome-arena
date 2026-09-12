"""
Janelia MaleCNS v1.0 Dataset Verification & Intake Module (Grant 1fab0)
Connects to public Google Cloud Storage Feather repositories published by Janelia & Google Research.
"""

import urllib.request
import json

GCS_BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

DATASET_MANIFEST = {
    "annotations": {
        "filename": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "url": f"{GCS_BASE_URL}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "expected_bytes": 14483314,
        "md5": "UKdxh3DFciDxYLpPQxq4ng==",
        "license": "CC-BY 4.0 (Creative Commons Attribution 4.0 International)",
        "citation": "Janelia MaleCNS Connectome v1.0 (Takemura et al., Cell 2026)"
    },
    "neurotransmitters": {
        "filename": "body-neurotransmitters-male-cns-v1.0.feather",
        "url": f"{GCS_BASE_URL}/body-neurotransmitters-male-cns-v1.0.feather",
        "expected_bytes": 43282834,
        "md5": "PYQrEv5cSe763lKNfdJKHw==",
        "license": "CC-BY 4.0",
        "citation": "Eckstein et al., Cell 2024 / Janelia MaleCNS neurotransmitter predictions"
    },
    "weights": {
        "filename": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "url": f"{GCS_BASE_URL}/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "expected_bytes": 1051241946,
        "license": "CC-BY 4.0",
        "citation": "MaleCNS 151.8M directed edges (pre, post, weight)"
    }
}

def verify_dataset_headers():
    """
    Performs HTTP HEAD requests against all GCS bulk artifacts to verify availability,
    byte lengths, and ETag checksums without transferring the multi-gigabyte files.
    """
    results = {}
    for key, info in DATASET_MANIFEST.items():
        req = urllib.request.Request(info["url"], method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
                cl = int(resp.headers.get("content-length", 0))
                etag = resp.headers.get("etag", "").strip('"')
                md5_hdr = resp.headers.get("x-goog-hash", "")
                results[key] = {
                    "status": status,
                    "verified": status == 200 and cl == info["expected_bytes"],
                    "content_length": cl,
                    "expected_bytes": info["expected_bytes"],
                    "etag": etag,
                    "license": info["license"]
                }
        except Exception as e:
            results[key] = {
                "status": "error",
                "verified": False,
                "error": str(e)
            }
    return results

if __name__ == "__main__":
    print("Verifying Janelia MaleCNS GCS Dataset Manifest...")
    res = verify_dataset_headers()
    print(json.dumps(res, indent=2))
