# Local LLM Foundry Documentation

Documentation index for Local LLM Foundry — local LLM inference, GPU
monitoring, chat, and managed runtime tooling. Historical evidence keeps its
original Llama Monitor names intentionally.

## Upgrade and brand

- [Upgrade from 1.x to 2.0](reference/upgrade-2-0.md)
- [Brand usage](reference/branding.md)

The reference guides below describe Foundry's local LLM inference and GPU
monitoring surfaces. Technical backend names and historical compatibility paths
are called out where they are part of the contract.

## Reference Guides

| Document | Description |
| ---------- | ------------- |
| [chat.md](reference/chat.md) | Chat features, guided generation, explicit mode, persona system |
| [dashboard.md](reference/dashboard.md) | Dashboard layout, settings, server tab, GPU metrics |
| [api.md](reference/api.md) | REST API endpoints, WebSocket schema |
| [model-library.md](reference/model-library.md) | Structured GGUF, MLX, safetensors, cache, and migration behavior |
| [rapid-mlx-runtime.md](reference/rapid-mlx-runtime.md) | Managed Rapid-MLX install, upgrade, repair, rollback, and isolation contracts |
| [remote-agent.md](reference/remote-agent.md) | Remote agent connection, multi-server setup |
| [inference-tuning.md](reference/inference-tuning.md) | Model/quant/KV/flag tuning for Apple Silicon & discrete GPUs (dense vs MoE, `--n-cpu-moe`) |
| [capabilities.md](reference/capabilities.md) | System capabilities, hardware support |
| [cli-flags.md](reference/cli-flags.md) | Command-line options |
| [llama-runtime.md](reference/llama-runtime.md) | llama.cpp runtime: source builds, prerequisites, rollback, K/V support, troubleshooting |
| [realtime-communication.md](reference/realtime-communication.md) | WebSocket protocol details |
| [cross-compilation.md](reference/cross-compilation.md) | Building for different platforms |
| [windows-sensor-bridge-implementation.md](reference/windows-sensor-bridge-implementation.md) | Windows sensor bridge |

## Other Directories

- **plans/** — Temporary planning documents for active feature development
- **archive/** — Old build docs and historical notes
- **screenshots/** — All project screenshots and GIFs
