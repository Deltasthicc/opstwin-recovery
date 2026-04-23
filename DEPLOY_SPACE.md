# Deploying OpsTwin to a HuggingFace Space

This is the step-by-step procedure to publish the environment as a Docker-SDK
Space, which is a minimum requirement for the Meta PyTorch OpenEnv Hackathon.

The Space serves the OpenEnv HTTP API (`/reset`, `/step`, `/health`) that any
client can hit, including the `client.py` in this repo and any OpenEnv-compliant
training loop.

## Prerequisites

1. A HuggingFace account. Your team uses the `Deltasthic` org.
2. A HuggingFace write token: https://huggingface.co/settings/tokens
3. The `huggingface_hub` Python package. Install with `pip install "huggingface_hub>=0.24"`.
4. This repo checked out locally.

## One-time setup

```bash
huggingface-cli login
```

Paste your write token when prompted. Do NOT commit the token to any repo.

## Step 1: Create the Space (one-time)

```bash
huggingface-cli repo create opstwin-recovery \
    --type space \
    --space_sdk docker \
    --organization Deltasthic
```

This creates the empty repo at `https://huggingface.co/spaces/Deltasthic/opstwin-recovery`.
If you are deploying to your personal account instead, omit the `--organization` flag.

## Step 2: Push the code

```bash
# From the repo root
git remote add hfspace https://huggingface.co/spaces/Deltasthic/opstwin-recovery
git push hfspace main:main
```

HF Spaces will detect the `sdk: docker` line in `README.md`, build the Dockerfile,
and start the container. Build takes 3 to 5 minutes. Visit the Space URL, the
**App** tab, and wait for status = Running.

If you prefer token-in-URL (not recommended; rotates easily):

```bash
HF_TOKEN=hf_xxx
git push https://user:${HF_TOKEN}@huggingface.co/spaces/Deltasthic/opstwin-recovery main:main
```

## Step 3: Smoke-test the live Space

Once the Space is Running, its URL is `https://Deltasthic-opstwin-recovery.hf.space`
(HF rewrites `/` to `-` in the org-plus-name pair).

```bash
python tests/smoke_http.py --base-url https://Deltasthic-opstwin-recovery.hf.space
```

Expected output: all 5 HTTP checks pass. If `/reset` or `/step` return HTTP 500,
the container likely crashed at import time. Check the Space **Logs** tab for
the traceback and patch either the Dockerfile or `server/`.

After confirming, save the URL to `demo-assets/hf-space-link.txt` so the blog
and video deliverables can reference it.

## Updating the Space

Any `git push hfspace` triggers a rebuild automatically. No manual redeploy step.

## Known constraints for HF Spaces Docker builds

- **Ephemeral filesystem.** Writes to paths other than `/tmp` or `/data` are lost
  on restart. The environment itself does not persist state, so this is fine.
- **Base image size.** The current `python:3.12-slim` + deps footprint is about
  450 MB. Well under the 50 GB free tier cap.
- **No GPU.** The serving container is CPU-only. Inference and training happen
  on the client side (Colab, RTX 5090, whatever the user supplies).
- **`app_port` must match Dockerfile `EXPOSE`.** Both are 8000 here. Do not
  change one without the other.

## Rollback

To take the Space offline without deleting it, click **Settings → Pause this Space**
in the HF web UI. To re-enable, click **Restart this Space**. No git action needed.
