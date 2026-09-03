# Local LLM Foundry Quick Start

This guide uses the Local LLM Foundry 2.0 identity. The `llama-monitor` binary
and legacy root remain valid compatibility aliases through 2.x; see the
[upgrade guide](upgrade-2-0.md).

Get your first local AI model up and running with Local LLM Foundry.

## 1. Launch Local LLM Foundry

Open Local LLM Foundry. By default it listens at:

- http://127.0.0.1:7778

On first launch, Local LLM Foundry auto-generates an API token and stores it in
`~/.config/llama-monitor`. All API endpoints (including local ones) require this token
via the `Authorization: Bearer <token>` header. The token is shown in Settings and is
used internally by the UI.

You’ll land on the dashboard / setup view, where you can browse available models, see your
hardware memory, and get started.

## 2. Use the Setup wizard (recommended)

The setup wizard is the easiest way to get started. It will:

- Detect your platform and recommend a backend:
  - llama.cpp on most platforms.
  - Rapid-MLX on Apple Silicon macOS (uses a managed runtime maintained by
    Llama Monitor).
- Guide you through model and quant selection.
- Use the VRAM estimator to keep your system stable.

Or, if you already have a server running, you can:

- Attach directly to an existing llama.cpp or Rapid-MLX server by selecting its
  engine and entering its URL.

Click **Open setup wizard** on the dashboard to begin.

The Setup wizard walks you through:

- Choosing a model from the list (curated picks and Hugging Face search).
- Picking a quant (e.g., Q4_K_M for a good balance of quality and speed).
- Setting hardware and memory (VRAM estimator) to keep your system stable.

## 3. Adjust hardware / memory (VRAM estimator)

On the Hardware & memory step:

- Use the suggested GPU layers (usually "Auto (recommended)").
- Check the VRAM estimator bar: it shows:
  - How much memory your model will use.
  - Whether you’re under budget (green), tight (yellow), or at risk of OOM (red).
- If it says "At risk", reduce GPU layers or context size until it shows "Safe".

## 4. Start server

At the end of the wizard:

- Review your preset settings.
- Click **Start server**.

Llama Monitor will:

- Start the selected backend with your chosen preset:
  - llama.cpp: launches llama-server.
  - Rapid-MLX: ensures its managed runtime is installed and launches Rapid-MLX.
- Switch you to the Performance & metrics dashboard.

## 5. Open a new conversation

- In the sidebar, click the **Chat** tab.
- Use **New conversation** to start a chat with your model.
- Send a message; you’ll see live tokens/sec, context usage, and response quality.

## 6. Use Performance & metrics to check health

On the **Server** page (Performance & metrics), watch:

- **Speed (throughput)** – tokens/sec (both prompt and generation).
- **Active sessions** – current parallel chat sessions.
- **Requests** – how many requests you’ve sent.
- **Model info** – current model, quant, and speculative decoding status.
- **GPU / System** – memory and load if you have a remote agent (optional but recommended).
- **Memory pressure** (macOS) – shows free memory, compressed memory, and swap activity. A warning or critical indicator means macOS is struggling; reduce context, stop heavy downloads, or disable mlock.

If you see:

- Steady tokens/sec and no red warnings: your setup is healthy.
- Very low tokens/sec or memory warnings: return to the preset and reduce GPU layers or context size.
- Memory pressure at warning or critical: macOS is under pressure; lower context size, reduce GPU layers, or disable mlock.

Note: mlock is a llama-server launch parameter (enforced via the wizard or preset), not
a simple runtime toggle: when enabled, model memory is pinned and cannot be paged out.
On tight systems, turning it off can prevent swap storms and freezes.

That’s it—you’re running your local AI model with full visibility.

## 7. Use sleep modes when the server is running

The monitoring chip in the top nav lets you cycle through three modes:

- **Monitoring** – full telemetry, GPU reads, system metrics, and live logs.
- **Logs only** – only the live log stream is active; GPU, system, and sparkline updates are paused to save resources.
- **Paused** – all telemetry and logs are paused; the backend keeps running.

Click the chip to cycle modes, or use it when you want the server running but need lower overhead on your system.

## 8. Network, GPU, and config (short)

- Default URL: http://127.0.0.1:7778 (loopback only; not reachable from other devices).
- LAN exposure: use `--host 0.0.0.0` (or 0.0.0.0) to allow access from other devices on your
  network. If you do this:
  - Set a strong server API key in the wizard or preset editor for your backend.
  - Be aware that exposing llama-monitor over the internet without TLS or an API gateway
    is not recommended.
- GPU / backend setup:
  - macOS on Apple Silicon:
  - Rapid-MLX: automatic (Metal); managed runtime handled by Local LLM Foundry.
    - llama.cpp: Metal is automatic.
  - Linux with NVIDIA GPU: use the Vulkan backend when the Vulkan driver is
    available, or install a custom CUDA-enabled llama.cpp build. The
    downloadable llama.cpp release archives currently publish CUDA binaries
    for Windows, not Linux.
  - AMD GPU: select ROCm for a supported GPU with the ROCm runtime. Vulkan is
    another option for many AMD discrete or integrated GPUs when the Vulkan
    driver is available; otherwise use the generic CPU backend.
  - Intel Linux or Windows: use OpenVINO for an Intel CPU/GPU/NPU runtime, or
    SYCL/oneAPI for supported Intel GPU/iGPU acceleration. GPU/NPU acceleration
    requires the device and matching host/container drivers to be available.
  - Windows: choose the backend that matches your GPU (CUDA, Vulkan, SYCL, OpenVINO) in
    the llama.cpp version modal or setup wizard.
- Config directory: most settings and files live under `~/.config/llama-monitor`:
  - API token, HF token, models directory, llama.cpp binary, and Rapid-MLX
    managed runtime are stored there (configurable in Settings).
