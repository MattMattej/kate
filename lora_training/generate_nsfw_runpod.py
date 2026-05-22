#!/usr/bin/env python3
"""
Genera imágenes vía RunPod. El worker carga LoRAs desde env (LORA1_URL, LORA2_URL).
No envíes lora URLs en el payload salvo que cambien.
"""

import os
import sys
import time
import base64
import argparse
import requests

def load_env():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        p = os.path.join(d, ".env")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip()
            break
        d = os.path.dirname(d)

load_env()

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT = os.environ.get("RUNPOD_ENDPOINT", "")


def call_runpod(payload: dict, timeout_sec: int = 900) -> dict:
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT:
        print("❌ Falta RUNPOD_API_KEY y RUNPOD_ENDPOINT en .env")
        sys.exit(1)

    base = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

    print(f"\n📤 Job → {RUNPOD_ENDPOINT} (/run async)")
    try:
        r = requests.post(f"{base}/run", json={"input": payload}, headers=headers, timeout=60)
    except requests.RequestException as e:
        print(f"❌ Network error posting to RunPod: {e}")
        sys.exit(1)
    
    if r.status_code != 200:
        print(f"❌ HTTP {r.status_code}: {r.text[:400]}")
        sys.exit(1)

    data = r.json()
    job_id = data.get("id")
    status = data.get("status", "IN_QUEUE")
    print(f"   Job ID: {job_id}")

    t_start = time.time()
    last = status
    poll_count = 0
    
    while status in ("IN_QUEUE", "IN_PROGRESS"):
        if time.time() - t_start > timeout_sec:
            print(f"\n❌ Timeout {timeout_sec}s después de {poll_count} polls")
            print(f"   Revisa logs RunPod — busca BUILD_ID=pixel-kate-v7")
            sys.exit(1)
        
        time.sleep(8)
        poll_count += 1
        
        try:
            sr = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=30)
        except requests.RequestException as e:
            print(f"   ⚠️  Poll {poll_count}: network error: {e}")
            continue
        
        if sr.status_code != 200:
            print(f"   ⚠️  Poll {poll_count}: HTTP {sr.status_code}")
            continue
        
        data = sr.json()
        status = data.get("status", status)
        elapsed = int(time.time() - t_start)
        
        if status != last:
            print(f"   [{elapsed}s] {status}")
            last = status

    if status == "FAILED":
        error_detail = data.get("error", data)
        print(f"\n❌ FAILED: {error_detail}")
        sys.exit(1)

    out = data.get("output") or {}
    if out.get("error"):
        print(f"\n❌ Worker error: {out['error']}")
        sys.exit(1)

    bid = out.get("build_id", "?")
    gen_time = out.get("generation_time_seconds", "?")
    print(f"\n✅ OK (build_id={bid}, gen_time={gen_time}s)")
    return out


def save_image(output: dict, out_dir: str = "./test_outputs") -> str:
    b64 = output.get("image_base64")
    if not b64:
        print(f"❌ Sin imagen: {output}")
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"nsfw_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
    with open(fn, "wb") as f:
        f.write(base64.b64decode(b64))
    return fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="KATESTYLE, portrait, soft lighting, masterpiece")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--output-dir", default="./test_outputs")
    args = p.parse_args()

    # Payload mínimo: LoRAs vienen del env del worker en RunPod
    payload = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance": args.guidance,
    }
    if args.seed is not None:
        payload["seed"] = args.seed

    print("=" * 50)
    print("Pixel Kate — RunPod")
    print(f"  {args.width}x{args.height} | {args.steps} steps")
    print("=" * 50)

    t0 = time.time()
    out = call_runpod(payload, args.timeout)
    print(f"   GPU gen: {out.get('generation_time_seconds')}s | total: {time.time()-t0:.0f}s")
    print(f"   Guardada: {save_image(out, args.output_dir)}")


if __name__ == "__main__":
    main()
