# 🔐 ComfyUI Qwen Prompt Expander

A privacy-focused, local AI prompt generator for ComfyUI.  
Turn short concepts (e.g., "woman in red dress") into professional, detailed Stable Diffusion prompts using advanced local LLMs like **Qwen 2.5** or **SmolLM2**.

**100% Offline. Zero API keys. Maximum Privacy.**

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)
![GPU: 6GB+](https://img.shields.io/badge/GPU-6GB%2B-orange.svg)

---

## ✨ Key Features

- **🧠 Local LLM Power:** Uses powerful models (Qwen, SmolLM2, Dolphin) running locally on your machine
- **🛡️ 100% Private:** No data leaves your computer. Works without internet (after initial model download)
- **📉 VRAM Friendly:** Supports **4-bit and 8-bit quantization**. Runs perfectly on **6GB VRAM cards** alongside SD 1.5!
- **🌍 Auto-Translation:** Built-in **Polish-to-English** translator (offline MarianMT) with GPU/CPU support
- **🧩 Embeddings Support:** Automatically detects and inserts embeddings from your ComfyUI models folder
- **🧹 Smart Memory Management:** Configurable model unloading and intelligent caching to prevent Out-Of-Memory (OOM) errors
- **✂️ Intelligent Token Management:** Automatic prompt trimming that preserves your subject and important details
- **♻️ Translation Cache:** Avoids re-translating the same text multiple times for faster workflows

---

## 🚀 Installation

### Method 1: ComfyUI Manager (Recommended)
1. Search for "Qwen Prompt Expander" in ComfyUI Manager
2. Click **Install**
3. Restart ComfyUI

---

### Method 2: Manual Installation
1. Navigate to your ComfyUI `custom_nodes` directory
2. Clone this repository:
```bash
   git clone https://github.com/AnonBOTpl/ComfyUI-Qwen-Prompt-Expander.git
```
3. Install dependencies:

   > **Note:** ComfyUI already includes `torch`, `transformers`, and other core packages.  
   > This will only install the additional packages needed for this node.

   **Portable Version (Windows):**
```cmd
   .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-Qwen-Prompt-Expander\requirements.txt
```
---

## 🎛️ Node Parameters Guide

### 1. Model Settings

**model_name:** Choose the brain of the generator
- **SmolLM2-1.7B (Recommended)**: Best balance of intelligence and size. Excellent at understanding context
- **Dolphin-2.9-Qwen (Uncensored)**: Great for creative/unrestricted prompting
- **Qwen-0.5B (Ultra Light)**: Ultra-fast, very lightweight, good for extremely low specs
- **Qwen-1.5B (Balanced)**: More capable than 0.5B, still runs on modest hardware

**precision:** Controls model size in VRAM
- **4-bit (Ultra Light)**: ~1.2GB VRAM. Minimal quality loss. **Best for 6GB GPUs**
- **8-bit (Fast)**: ~1.8GB VRAM. Good balance of speed and quality
- **fp16 (Standard)**: ~3.5GB VRAM. Full precision, best quality, high VRAM usage

### 2. Prompting

- **subject**: Your main idea (e.g., "knight eating pizza"). Can be in Polish if translation is enabled
- **use_translator_PL_EN**: Translates input from Polish to English using offline CPU/GPU model
- **translator_device**: Choose CPU (stable, zero VRAM) or GPU (faster) for translation
- **max_tokens**: Limits the length of generated prompt. Lower (60-80) for SD 1.5, higher (100-150) for SDXL/Flux

### 3. Style & Quality

- **style / lighting / quality**: Presets to guide the AI. Set to "Disabled" if you want the LLM to invent everything itself
- Combinations create unique atmospheres (e.g., "Cyberpunk + Neon Lights + 8k Masterpiece")

### 4. Embeddings & Negative

- **emb_positive / emb_negative**: Select embeddings from your `ComfyUI/models/embeddings` folder. Automatically added to prompt
- **add_negative**: If enabled, outputs the negative prompt text
- **negative_prompt_text**: Edit your negative prompt (defaults include common quality issues)

### 5. System Settings

**device_mode:**
- **GPU (Fast)**: Uses graphics card. Fast generation (2-5 seconds)
- **CPU (Safe Mode)**: Runs entirely on RAM/CPU. Slower (10-30s), but uses **0 VRAM**. Use if GPU crashes

**unload_model:** CRITICAL VRAM SETTING
- **True (Save VRAM)**: Loads LLM → Generates → Deletes from VRAM. Essential for low VRAM setups
- **False (Cache - Fast)**: Keeps LLM in VRAM. Instant subsequent generations

**seed:** Set to 0 for random, or specific number for reproducible prompts (with slight variation due to temperature)

---

## 🔋 VRAM Optimization Guide

Choose settings based on your GPU memory:

### 🟢 Scenario A: Low VRAM (4GB - 6GB)
**Goal:** Run alongside SD 1.5 without crashing

- **Model:** SmolLM2-1.7B or Qwen-0.5B
- **Precision:** 4-bit (Ultra Light)
- **Unload Model:** True *(Must unload to make room for SD generation)*
- **Device:** GPU
- **Expected:** 2-4s generation, stable workflow

### 🟡 Scenario B: Medium VRAM (8GB - 12GB)
**Goal:** Speed and quality balance

- **Model:** SmolLM2-1.7B or Qwen-1.5B
- **Precision:** 8-bit or 4-bit
- **Unload Model:** False *(Keep in cache for instant prompting)*
- **Device:** GPU
- **Expected:** 1-3s generation, excellent quality

### 🔴 Scenario C: High VRAM (16GB - 24GB)
**Goal:** Maximum quality, no compromises

- **Model:** Qwen-1.5B or larger
- **Precision:** fp16 (Standard)
- **Unload Model:** False
- **Device:** GPU
- **Expected:** <1s generation, best possible quality

### 🥔 Scenario D: Integrated Graphics / Very Old GPU
**Goal:** Just make it work

- **Device Mode:** CPU (Safe Mode)
- **Model:** Qwen-0.5B
- **Unload Model:** True
- **Expected:** 15-30s generation, but will not crash

---

## 🎭 Prompt Philosophy

This node generates **conceptual prompts** rather than **literal descriptions**, giving Stable Diffusion creative freedom while maintaining your intended direction.

### Example Generation:

**Input:** `"cyberpunk girl"`  
**Settings:** SmolLM2-1.7B, 8-bit, 80 tokens, Cyberpunk + Dark & Moody

**Generated Output:**
```
cyberpunk girl in high-tech environment with sleek lines, sharp angles, 
digital textures, deep shadows, cinematic lighting effects, ultra-detailed 
facial features, masterful composition, 8k resolution
```

**Tokens Used:** 35/80 ✅

### Why Conceptual Prompts Work Better:

| Traditional (Literal) | Our Approach (Conceptual) |
|----------------------|---------------------------|
| "girl with black leather jacket, purple hair, neon city background, standing pose..." | "cyberpunk girl in high-tech environment with sleek lines, sharp angles, digital textures..." |
| ❌ Too rigid, limits SD creativity | ✅ Focuses on mood and atmosphere |
| ❌ Often 100+ tokens | ✅ Concise 30-50 tokens |
| ❌ Same output every time | ✅ Natural variation between generations |

---

## 🛠️ Troubleshooting

### Error: `ModuleNotFoundError: No module named 'bitsandbytes'`

Windows users often face issues with bitsandbytes (required for 4-bit/8-bit quantization).

**Solution:**
1. Go to `ComfyUI_windows_portable` folder
2. Open CMD/Terminal there
3. Run:
   ```cmd
   .\python_embeded\python.exe -m pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
   ```

**Note:** Without bitsandbytes, the node will automatically fall back to fp16 precision.

---

### Error: OOM (Out of Memory)

If you see "CUDA out of memory" or your GPU driver crashes:

1. **Set `precision` to 4-bit (Ultra Light)**
2. **Set `unload_model` to True (Save VRAM)**
3. **Lower `max_tokens` to 60** (generates shorter prompts)
4. If still crashing: **Switch `device_mode` to CPU (Safe Mode)**

The node includes intelligent OOM protection that will attempt to recover and return a fallback prompt instead of crashing your entire workflow.

---

### Slow Generation Times

**On GPU:**
- First generation: 2-5s (model loading)
- Subsequent: <1s (if `unload_model` is False)
- If always slow: Check if model is loading each time (set `unload_model` to False)

**On CPU:**
- Expected: 10-30s per generation
- This is normal for CPU inference
- Consider upgrading to a GPU, or use Qwen-0.5B for faster CPU generation

---

### Translation Issues

**Translation cache:** The node remembers up to 100 translations to speed up repeated use

**Polish text not translating:**
1. Verify `use_translator_PL_EN` is enabled
2. First use downloads the model (~300MB) - check console for "Loading Translator"
3. Check console for translation errors

**To force fresh translation:** Restart ComfyUI (clears cache)

---

## 🧪 Advanced Tips

### Token Management

The node includes intelligent token trimming:
- Generates 20% more tokens than requested
- Intelligently trims while preserving your subject
- Prioritizes important keywords over generic quality tags

**Console Output Example:**
```
🎨 Generating prompt for: cyberpunk girl...
✂️ [LLM] Response too long (95 tokens), trimming to 80...
📊 Kept 15/22 tags (78 tokens)
```

### Memory Optimization

**Cache Strategy:**
- LLM stays in memory if `unload_model` = False
- Translator stays in memory across generations
- Translation cache holds 100 entries
- Everything cleared when `unload_model` = True

**Best for Speed:** Set `unload_model` = False  
**Best for VRAM:** Set `unload_model` = True

### Seed Behavior

Seeds provide *approximate* reproducibility:
- Same seed + same input ≈ similar output
- Temperature (0.8) and sampling introduce variation
- Useful for iterating on a concept while keeping the general style

---

## ❤️ Credits

- **Qwen Models**: [Qwen Team at Alibaba Cloud](https://github.com/QwenLM/Qwen)
- **SmolLM2**: [HuggingFaceTB](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- **MarianMT Translation**: [Helsinki-NLP](https://huggingface.co/Helsinki-NLP)
- **Dolphin Uncensored Models**: [Cognitive Computations](https://huggingface.co/cognitivecomputations)
- **ComfyUI**: [comfyanonymous](https://github.com/comfyanonymous/ComfyUI)
- **Transformers & BitsAndBytes**: [HuggingFace](https://huggingface.co/)

---

## 📄 License

MIT License - Feel free to use and modify!

See [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
git clone https://github.com/AnonBOTpl/ComfyUI-Qwen-Prompt-Expander.git
cd ComfyUI-Qwen-Prompt-Expander
pip install -r requirements.txt
```

### Areas for Contribution
- 🌍 Additional language support (Spanish, German, French, etc.)
- 🎨 New style presets
- 🔧 Performance optimizations
- 📝 Documentation improvements
- 🐛 Bug reports and fixes

---

## 🌟 Community Showcase

Have you created amazing images using this node? We'd love to see them!

Open an issue with the tag `showcase` and include:
- Your input prompt
- Model settings used
- Final generated image (optional)

The best submissions will be featured in this README!

---

## 🔮 Roadmap

- [ ] Support for more translation languages
- [ ] Preset save/load system
- [ ] Advanced prompt weighting options
- [ ] Integration with more LLM models
- [ ] Batch processing support
- [ ] Style transfer from reference images

---

## 💬 Support

- **Issues:** [GitHub Issues](https://github.com/AnonBOTpl/ComfyUI-Qwen-Prompt-Expander/issues)
- **Discussions:** [GitHub Discussions](https://github.com/AnonBOTpl/ComfyUI-Qwen-Prompt-Expander/discussions)

---

## ⭐ Star History

If this project helped you, please consider giving it a star! It helps others discover the project.

---

**Made with ❤️ for the ComfyUI community**
