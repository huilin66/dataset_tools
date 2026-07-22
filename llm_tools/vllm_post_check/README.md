# VLLM YOLO Post Check

A small standalone toolkit for using Qwen/VLLM to generate or refine YOLO txt labels.

## Install

```powershell
pip install pillow openai tqdm
```

`openai` is only needed when calling a VLLM/OpenAI-compatible endpoint.


## Smoke-test the VLLM model

If processing is suspiciously fast or all output txt files are empty, first test whether the model endpoint can really see one image:

```powershell
python test_vllm.py ^
  --image E:\data\images\one_image.jpg ^
  --base-url http://127.0.0.1:18001/v1 ^
  --model Qwen/Qwen3-VL-8B-Instruct
```

To test the defect-detection prompt and JSON parser:

```powershell
python test_vllm.py ^
  --image E:\data\images\one_image.jpg ^
  --classes E:\data\classes.txt ^
  --mode detect ^
  --base-url http://127.0.0.1:18001/v1 ^
  --model Qwen/Qwen3-VL-8B-Instruct ^
  --raw-output E:\data\vllm_test_raw.json
```

Also inspect `manifest.csv`; many `no_pred`, `skipped`, or `error` rows mean the pipeline did not actually refine detections.

## Mode 1: full-image prediction

Read images and `classes.txt`, ask Qwen/VLLM to detect defects, then write YOLO txt.

```powershell
python cli.py full-image ^
  --images E:\data\images ^
  --classes E:\data\classes.txt ^
  --output E:\data\vllm_full ^
  --base-url http://127.0.0.1:8000/v1 ^
  --model Qwen/Qwen3-VL-8B-Instruct
```

Output:

```text
output/
├─ labels/
├─ raw/
├─ results/       # one structured JSON per image
└─ manifest.csv
```

## Mode 2: crop-refine existing YOLO predictions

Read images, existing YOLO prediction txt, and `classes.txt`. Each YOLO box is cropped with padding, checked by Qwen/VLLM, then written back as YOLO txt.

```powershell
python cli.py crop-refine ^
  --images E:\data\images ^
  --pred-labels E:\data\yolo_preds ^
  --classes E:\data\classes.txt ^
  --output E:\data\vllm_refined ^
  --base-url http://127.0.0.1:8000/v1 ^
  --model Qwen/Qwen3-VL-8B-Instruct ^
  --crop-padding 0.15 ^
  --mode classification
```

Modes:

- `classification`: Qwen keeps/rejects/corrects class; original YOLO box is preserved.
- `detect`: Qwen returns a bbox inside the crop; the bbox is mapped back to the original image.

## YOLO txt format

Default output includes confidence:

```text
class_id x_center y_center width height confidence
```

Use `--no-conf` to write training-label style:

```text
class_id x_center y_center width height
```


## Port forwarding to a remote VLLM server

This script mirrors `E:\repository\bdd_project\demo\link\link.sh` by default:

```text
local 127.0.0.1:18001 -> remote 127.0.0.1:8001
local 127.0.0.1:18002 -> remote 127.0.0.1:8002
ssh port: 10022
remote user/host: 23039356r@10.21.17.29
```

Start the tunnel:

```powershell
.\start_port_forward.ps1
```

Then call the post-check tool with one of these base URLs:

```powershell
--base-url http://127.0.0.1:18001/v1
--base-url http://127.0.0.1:18002/v1
```

Keep reconnecting after disconnect:

```powershell
.\start_port_forward.ps1 -Reconnect
```

Run in background:

```powershell
.\start_port_forward.ps1 -Background
```

Custom remote:

```powershell
.\start_port_forward.ps1 -RemoteHost 192.168.1.10 -RemoteUser user -SshPort 22
```

Use only the first mapping:

```powershell
.\start_port_forward.ps1 -Single -LocalPort1 18001 -RemotePort1 8001
```

## Environment variables

```text
VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

## Notes

- The tool is intentionally independent from the BDD workflow/config system.
- Empty detections still create an empty txt file.
- Raw model responses are saved for audit and parser debugging.
- For large datasets, run with `--limit` first as a smoke test.
- A tqdm progress bar is shown by default; use `--no-progress` to disable it.
- Each image writes a structured JSON summary to `output/results/{image_stem}.json`.





