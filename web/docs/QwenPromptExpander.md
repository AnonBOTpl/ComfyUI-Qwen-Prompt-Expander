# Qwen Prompt Expander

Local AI prompt generator for ComfyUI using Qwen/SmolLM2 models.

## Quick Start

1. Add the **Qwen Prompt Expander** node to your workflow
2. Choose a model (SmolLM2-1.7B recommended for beginners)
3. Enter your subject (e.g., "cat in forest")
4. Click Generate!

## First Time Setup

**Important:** Run the **Qwen Diagnostics** node first!

This will:
- Test your internet connection
- Verify folder permissions
- Download a small test model
- Confirm everything works

## Parameters

### Model Selection
- **Model Source**: Choose between preset models, custom HuggingFace ID, or local path
- **Preset Model**: Pre-configured models with size information
- **Precision**: 
  - 4-bit (Ultra Light): ~1.2GB VRAM, best for 6GB GPUs
  - 8-bit (Fast): ~1.8GB VRAM, balanced
  - fp16 (Standard): ~3.5GB VRAM, best quality

### Prompting
- **Subject**: Your main idea (supports Polish with translation enabled)
- **Max Tokens**: Length of generated prompt (60-80 for SD 1.5, 100-150 for SDXL)

### Style Controls
- **Style/Lighting/Quality**: Optional presets to guide the AI
- Set to "Disabled" for full LLM creativity

### System Settings
- **Device Mode**: GPU (fast) or CPU (safe, slower)
- **Unload Model**: 
  - False (Cache): 10x faster subsequent generations
  - True (Save VRAM): Frees memory after each generation

## Examples

**Input:** "cyberpunk girl"  
**Settings:** SmolLM2-1.7B, 8-bit, Cyberpunk + Dark & Moody  
**Output:** "cyberpunk girl in high-tech environment with sleek lines, sharp angles, digital textures, deep shadows, cinematic lighting effects, ultra-detailed facial features, 8k"

**Input:** "cat in forest"  
**Settings:** Qwen-0.5B, 4-bit, Photorealistic + Golden Hour  
**Output:** "fluffy orange cat, green forest background, dappled sunlight, autumn leaves, shallow depth of field, natural lighting, professional photography"

## Custom Models

You can use any instruction-tuned model from HuggingFace:

1. Set **Model Source** to "Custom HuggingFace ID"
2. Enter model ID (e.g., `mistralai/Mistral-7B-Instruct-v0.2`)
3. Model downloads automatically on first use

## Troubleshooting

**Models don't download:**
- Run Diagnostics node first
- Check internet connection
- Ensure 10GB+ free disk space

**Out of Memory:**
- Use 4-bit precision
- Enable "Unload Model"
- Try smaller model (Qwen-0.5B)
- Switch to CPU mode

**Slow generations:**
- First run always slower (model loading)
- Set Unload Model = False for instant subsequent generations
- Check console for "⚡ Using cached model" message

For more help, see the full README on GitHub.
