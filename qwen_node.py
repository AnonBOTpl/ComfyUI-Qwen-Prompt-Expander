import torch
import gc
import random
import os
import folder_paths
import comfy.model_management

# Obsługa importów
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel, MarianTokenizer, BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    print("\n\033[91m[Offline Node] BRAK 'bitsandbytes'! Tryb 4-bit/8-bit nie zadziala. Uruchom: pip install bitsandbytes\033[0m")
    HAS_BNB = False
    if "AutoModelForCausalLM" not in locals():
        from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel, MarianTokenizer

class QwenOfflinePrompt:
    def __init__(self):
        self.comfy_base = folder_paths.base_path
        self.model_dir = os.path.join(self.comfy_base, "models", "LLM")
        
        # CACHE DLA LLM (GPU)
        self.llm_model = None
        self.llm_tokenizer = None
        self.current_model_name = ""
        self.current_precision = ""

        # CACHE DLA TRANSLATORA
        self.trans_model = None
        self.trans_tokenizer = None
        self.trans_device = None
        self.translation_cache = {}
        self.max_cache_size = 100
        
        if not os.path.exists(self.model_dir):
            try: os.makedirs(self.model_dir)
            except: pass

    @classmethod
    def INPUT_TYPES(s):
        # Pobieranie listy embeddings
        try:
            embeddings_list = folder_paths.get_filename_list("embeddings")
        except:
            embeddings_list = []
        
        embeddings_list = ["None"] + embeddings_list

        return {
            "required": {
                # MODEL SELECTION - Nowy system z rozmiarami!
                "model_source": (
                    ["Preset Models", "Custom HuggingFace ID", "Local Path"],
                    {"default": "Preset Models"}
                ),
                "preset_model": (
                    [
                        "HuggingFaceTB/SmolLM2-1.7B-Instruct (~1.7B - Recommended)",
                        "Qwen/Qwen2.5-0.5B-Instruct (~0.5B - Ultra Light)",
                        "Qwen/Qwen2.5-1.5B-Instruct (~1.5B - Balanced)",
                        "cognitivecomputations/dolphin-2.9.4-qwen2.5-1.5b (~1.5B - Uncensored)",
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~1.1B - Fast)",
                        "microsoft/Phi-3-mini-4k-instruct (~3.8B - Quality, needs 12GB VRAM)",
                        "meta-llama/Llama-3.2-1B-Instruct (~1B - Meta)"
                    ], 
                    {"default": "HuggingFaceTB/SmolLM2-1.7B-Instruct (~1.7B - Recommended)"}
                ),
                "custom_model_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "e.g., mistralai/Mistral-7B-Instruct-v0.2 or /path/to/local/model"
                }),
                
                "precision": (
                    ["fp16 (Standard)", "8-bit (Fast)", "4-bit (Ultra Light)"],
                    {"default": "8-bit (Fast)"}
                ),
                "subject": ("STRING", {
                    "multiline": True, 
                    "default": "beautiful scenery nature glass bottle landscape, purple galaxy bottle", 
                    "placeholder": "Type your concept here..."
                }),
                "negative_prompt_text": ("STRING", {
                    "multiline": True,
                    "default": "ugly, deformed, noisy, blurry, low quality, watermark, text, bad anatomy, bad hands, missing fingers, extra limbs",
                    "placeholder": "Edit your negative prompt here..."
                }),
                "style": (["Disabled", "Photorealistic", "Cyberpunk", "Dark Fantasy", "Anime", "Oil Painting", "Cinematic", "Vintage Photo"],),
                "lighting": (["Disabled", "Cinematic Lighting", "Studio Softbox", "Golden Hour", "Neon Lights", "Dark & Moody"],),
                "quality": (["Disabled", "8k, Masterpiece", "RAW Photo", "Unreal Engine 5", "Sharp Focus"],),
                
                "max_tokens": ("INT", {"default": 80, "min": 20, "max": 500, "step": 5, "display": "slider"}),
                "emb_positive": (embeddings_list, {"default": "None"}),
                "emb_negative": (embeddings_list, {"default": "None"}),

                "device_mode": (["GPU (Fast)", "CPU (Safe Mode)"], {"default": "GPU (Fast)"}),
                "translator_device": (["CPU (Stable)", "GPU (Faster)"], {"default": "CPU (Stable)"}),
                "use_translator_PL_EN": ("BOOLEAN", {"default": False}),
                "unload_model": ("BOOLEAN", {"default": False, "label_on": "True (Save VRAM)", "label_off": "False (Cache - Fast)"}),
                "add_negative": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "generate"
    CATEGORY = "🔐 Private AI"

    def translate(self, text, translator_device="cpu"):
        """Tłumaczy używając CACHE (nie ładuje modelu od nowa)"""
        if not text or len(text) < 2: 
            return text
        
        # Sprawdź cache
        if text in self.translation_cache:
            print("♻️ [Translator] Using cached translation")
            return self.translation_cache[text]
        
        try:
            # Określ device dla translatora
            trans_device = "cpu"
            if translator_device == "GPU (Faster)" and torch.cuda.is_available():
                trans_device = "cuda"
            
            # Ładowanie lub przenoszenie modelu
            needs_load = self.trans_model is None
            needs_move = (self.trans_model is not None and 
                         str(next(self.trans_model.parameters()).device) != trans_device)
            
            if needs_load:
                print(f"🌍 [Translator] Loading to {trans_device.upper()}...")
                self.trans_tokenizer = MarianTokenizer.from_pretrained(
                    "Helsinki-NLP/opus-mt-pl-en", 
                    cache_dir=self.model_dir
                )
                self.trans_model = MarianMTModel.from_pretrained(
                    "Helsinki-NLP/opus-mt-pl-en", 
                    cache_dir=self.model_dir
                ).to(trans_device)
                self.trans_device = trans_device
            elif needs_move:
                print(f"🔄 [Translator] Moving to {trans_device.upper()}...")
                self.trans_model = self.trans_model.to(trans_device)
                self.trans_device = trans_device
            
            # Tłumaczenie
            inputs = self.trans_tokenizer(text, return_tensors="pt", padding=True).to(trans_device)
            translated = self.trans_model.generate(**inputs, max_new_tokens=512)
            res = self.trans_tokenizer.decode(translated[0], skip_special_tokens=True)
            
            # Zarządzanie cache'em
            if len(self.translation_cache) >= self.max_cache_size:
                self.translation_cache.pop(next(iter(self.translation_cache)))
            
            self.translation_cache[text] = res
            return res
            
        except Exception as e:
            print(f"❌ [Translator] Error: {e}")
            return text

    def generate(self, model_source, preset_model, custom_model_id, precision, subject, 
                 negative_prompt_text, style, lighting, quality, max_tokens, emb_positive, 
                 emb_negative, device_mode, translator_device, use_translator_PL_EN, 
                 unload_model, add_negative, seed):
        
        # Determine which model to use based on source
        if model_source == "Preset Models":
            # Extract model ID from preset (before " (~")
            actual_model_id = preset_model.split(" (~")[0]
            print(f"📦 [Model] Using preset: {actual_model_id}")
        elif model_source == "Custom HuggingFace ID":
            if not custom_model_id or custom_model_id.strip() == "":
                print("❌ [Model] Custom HuggingFace ID is empty! Using default preset.")
                actual_model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
            else:
                actual_model_id = custom_model_id.strip()
                print(f"🌐 [Model] Using custom HuggingFace model: {actual_model_id}")
        else:  # Local Path
            if not custom_model_id or custom_model_id.strip() == "":
                print("❌ [Model] Local path is empty! Using default preset.")
                actual_model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
            else:
                actual_model_id = custom_model_id.strip()
                if os.path.exists(actual_model_id):
                    print(f"💾 [Model] Using local model: {actual_model_id}")
                else:
                    print(f"⚠️ [Model] Local path does not exist: {actual_model_id}")
                    print(f"💡 [Model] Treating as HuggingFace ID instead...")
        
        target_device = "cpu"
        if "GPU" in device_mode and comfy.model_management.get_torch_device().type == "cuda":
            target_device = "cuda"

        # 1. Tłumaczenie
        final_subject = subject
        if use_translator_PL_EN:
            final_subject = self.translate(subject, translator_device)

        # 2. VRAM Management - POPRAWKA: tylko gdy zmiana modelu, nie przy pierwszym load
        model_changed = (self.llm_model is None) or \
                        (self.current_model_name != actual_model_id) or \
                        (self.current_precision != precision)

        # Only clear VRAM when switching models (not on first load)
        if target_device == "cuda" and model_changed and self.llm_model is not None:
            print("🧹 [LLM] Clearing VRAM for model change...")
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

        # 3. Load Model
        if model_changed:
            print(f"🔐 [LLM] Loading {actual_model_id} ({precision})...")
            
            # Sprawdź czy folder istnieje i jest writable
            if self.model_dir and not os.path.exists(self.model_dir):
                print(f"📁 [LLM] Creating model directory: {self.model_dir}")
                try:
                    os.makedirs(self.model_dir, exist_ok=True)
                except Exception as e:
                    print(f"❌ [LLM] Cannot create model directory: {e}")
                    print(f"💡 [LLM] Using default HuggingFace cache instead...")
                    self.model_dir = None
            
            # Test write permissions
            if self.model_dir:
                test_file = os.path.join(self.model_dir, ".write_test")
                try:
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                except Exception as e:
                    print(f"⚠️ [LLM] No write permission: {e}")
                    print(f"💡 [LLM] Using default HuggingFace cache...")
                    self.model_dir = None
            
            # Usuń stary jeśli istnieje
            if self.llm_model is not None:
                del self.llm_model
                del self.llm_tokenizer
                gc.collect()

            try:
                print(f"📥 [LLM] Downloading/Loading model (first use: 5-10 min)...")
                
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    actual_model_id, 
                    cache_dir=self.model_dir,
                    trust_remote_code=True
                )
                
                quantization_config = None
                load_dtype = torch.float16
                
                # Obsługa quantization tylko gdy bitsandbytes jest dostępne
                if target_device == "cuda" and HAS_BNB:
                    if "4-bit" in precision:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                    elif "8-bit" in precision:
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                elif not HAS_BNB and precision != "fp16 (Standard)":
                    print("⚠️ [LLM] bitsandbytes not available, falling back to fp16")
                    precision = "fp16 (Standard)"
                
                if target_device == "cpu":
                    quantization_config = None
                    load_dtype = torch.float32

                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    actual_model_id, 
                    torch_dtype=load_dtype,
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True, 
                    cache_dir=self.model_dir,
                    trust_remote_code=True
                )
                
                if quantization_config is None and target_device == "cuda":
                    self.llm_model = self.llm_model.to(target_device)

                self.current_model_name = actual_model_id
                self.current_precision = precision
                
                # Info o modelu
                try:
                    num_params = self.llm_model.num_parameters() / 1e9
                    vram_estimate = {
                        "4-bit": num_params * 0.5,
                        "8-bit": num_params * 1.0,
                        "fp16": num_params * 2.0
                    }
                    precision_key = "4-bit" if "4-bit" in precision else ("8-bit" if "8-bit" in precision else "fp16")
                    estimated_vram = vram_estimate.get(precision_key, num_params * 2.0)
                    
                    print(f"✅ [LLM] Model loaded successfully!")
                    print(f"📊 [LLM] Parameters: ~{num_params:.1f}B")
                    print(f"💾 [LLM] Est. VRAM usage: ~{estimated_vram:.1f}GB ({precision})")
                except:
                    print("✅ [LLM] Model loaded successfully")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"💥 [LLM] OUT OF MEMORY!")
                    print(f"💡 Try: 4-bit precision or smaller model")
                    print(f"💡 Or: Enable 'unload_model' option")
                    # Cleanup i zwróć fallback
                    gc.collect()
                    if target_device == "cuda":
                        torch.cuda.empty_cache()
                    return (
                        f"{final_subject}, high quality, detailed",
                        negative_prompt_text if add_negative else ""
                    )
                else:
                    print(f"❌ [LLM] Runtime Error: {e}")
                    return (final_subject, negative_prompt_text if add_negative else "")
            except Exception as e:
                error_msg = str(e)
                print(f"❌ [LLM] Load Error: {error_msg}")
                
                # Detailed diagnostics
                if "ConnectTimeout" in error_msg or "ConnectionError" in error_msg:
                    print(f"🌐 Internet connection issue!")
                    print(f"💡 Check connection or try again later")
                elif "rate limit" in error_msg.lower():
                    print(f"⏱️ HuggingFace rate limit!")
                    print(f"💡 Wait a few minutes and retry")
                elif "permission" in error_msg.lower() or "denied" in error_msg.lower():
                    print(f"🔒 Permission error!")
                    print(f"💡 Check folder permissions: {self.model_dir}")
                elif "disk" in error_msg.lower() or "space" in error_msg.lower():
                    print(f"💾 Disk space issue!")
                    print(f"💡 Free up ~10GB and retry")
                else:
                    print(f"💡 Check model ID or try Diagnostics node")
                
                return (final_subject, negative_prompt_text if add_negative else "")
        else:
            # Model w cache - instant generation!
            print("⚡ [LLM] Using cached model (instant generation)")

        # 4. Prompt Construction
        random.seed(seed)
        if seed != 0:
            print(f"🎲 Using seed: {seed} (Note: Results may vary slightly due to temperature)")
        
        modifiers = []
        if style != "Disabled": modifiers.append(style)
        if lighting != "Disabled": modifiers.append(lighting)
        if quality != "Disabled": modifiers.append(quality)
        style_string = ", ".join(modifiers) if modifiers else "High Quality"

        # Inteligentny bufor - dajemy modelowi 20% więcej przestrzeni
        generation_buffer = int(max_tokens * 1.2)

        system_prompt = f"""You are a Creative Art Director for Stable Diffusion.
TASK: Create a detailed visual prompt using approximately {max_tokens} tokens.
RULES:
1. Be concise and selective - quality over quantity.
2. FORMAT: Comma-separated descriptive tags only.
3. NO CHAT. NO JSON. NO CODE. NO explanations.
4. Focus on the most impactful visual elements.
"""

        user_message = f"""
Input: cat, Cyberpunk
Output: fluffy black cat, glowing green eyes, metallic collar, neon roof, futuristic city skyline, pink and blue neon lights, wet texture, 8k

Input: {final_subject}
Style: {style_string}
Output:"""

        messages = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_message}
        ]
        
        try:
            # Try apply_chat_template (works for most models)
            try:
                text_input = self.llm_tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except:
                # Fallback for models without chat template
                print("⚠️ [LLM] Model doesn't support chat template, using simple format")
                text_input = f"{system_prompt}\n\n{user_message}"
            
            model_inputs = self.llm_tokenizer([text_input], return_tensors="pt").to(self.llm_model.device)

            # 5. Generowanie
            print(f"🎨 Generating prompt for: {final_subject[:50]}...")
            with torch.no_grad():
                generated_ids = self.llm_model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=generation_buffer,
                    do_sample=True,
                    temperature=0.8,
                    repetition_penalty=1.2,
                    pad_token_id=self.llm_tokenizer.eos_token_id if self.llm_tokenizer.eos_token_id else 0
                )

            new_tokens = generated_ids[0][len(model_inputs.input_ids[0]):]
            raw_response = self.llm_tokenizer.decode(new_tokens, skip_special_tokens=True)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"💥 [LLM] OOM during generation! Returning fallback...")
                gc.collect()
                if target_device == "cuda":
                    torch.cuda.empty_cache()
                return (
                    f"{final_subject}, {style_string}",
                    negative_prompt_text if add_negative else ""
                )
            else:
                raise e
        
        # 6. CLEANING
        clean_response = raw_response.replace("Output:", "").replace("Here is the prompt:", "")
        for char in ['{', '}', '[', ']', '```json', '```']:
            clean_response = clean_response.replace(char, "")

        forbidden_headers = [
            "Clothing:", "Outfit:", "Background:", "Environment:", 
            "Lighting:", "Style:", "Subject:", "Appearance:"
        ]
        for header in forbidden_headers:
            clean_response = clean_response.replace(header, "")
            clean_response = clean_response.replace(header.lower(), "")

        clean_response = clean_response.replace("\n", ", ").replace("  ", " ").strip().strip(",")
        
        # Usuwanie duplikatów
        tags = [t.strip() for t in clean_response.split(",")]
        seen = set()
        unique_tags = []
        for tag in tags:
            t_lower = tag.lower()
            if t_lower and t_lower not in seen:
                seen.add(t_lower)
                unique_tags.append(tag)
        
        clean_response = ", ".join(unique_tags)

        # 7. SMART TOKEN TRIMMING
        token_count = len(self.llm_tokenizer.encode(clean_response))
        
        if token_count > max_tokens:
            print(f"✂️ [LLM] Response too long ({token_count} tokens), trimming to {max_tokens}...")
            
            tags = [t.strip() for t in clean_response.split(",") if t.strip()]
            
            subject_keywords = set(final_subject.lower().split())
            important_tags = []
            other_tags = []
            
            for tag in tags:
                tag_lower = tag.lower()
                if any(keyword in tag_lower for keyword in subject_keywords):
                    important_tags.append(tag)
                else:
                    other_tags.append(tag)
            
            trim_ratio = (max_tokens / token_count) * 0.95
            total_tags = len(tags)
            keep_count = max(1, int(total_tags * trim_ratio))
            
            min_important = min(len(important_tags), max(1, keep_count // 2))
            
            final_tags = important_tags[:min_important]
            remaining_slots = keep_count - len(final_tags)
            
            if remaining_slots > 0:
                final_tags.extend(other_tags[:remaining_slots])
            
            clean_response = ", ".join(final_tags)
            final_token_count = len(self.llm_tokenizer.encode(clean_response))
            print(f"📊 Kept {len(final_tags)}/{total_tags} tags ({final_token_count} tokens)")
        else:
            print(f"📊 Generated {token_count} tokens (within limit)")

        # Fallback
        subject_words = final_subject.lower().split()
        key_word = subject_words[0] if subject_words else ""
        if key_word and key_word not in clean_response.lower():
             clean_response = f"{final_subject}, {clean_response}"

        # 8. EMBEDDINGS
        if emb_positive != "None":
            clean_response = f"embedding:{emb_positive}, {clean_response}"

        neg = ""
        if add_negative:
            neg = negative_prompt_text
            if emb_negative != "None":
                neg = f"embedding:{emb_negative}, {neg}"

        # 9. Unload Logic
        if unload_model:
            print("🧹 [Node] Unloading EVERYTHING (LLM + Translator)...")
            
            if self.llm_model is not None:
                del self.llm_model
                del self.llm_tokenizer
                self.llm_model = None
                self.llm_tokenizer = None
            
            if self.trans_model is not None:
                del self.trans_model
                del self.trans_tokenizer
                self.trans_model = None
                self.trans_tokenizer = None
                self.trans_device = None
            
            gc.collect()
            if target_device == "cuda":
                torch.cuda.empty_cache()
            comfy.model_management.soft_empty_cache()

        return (clean_response, neg)


class QwenDiagnostics:
    """Diagnostic node to test model downloading and system compatibility"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "test_model": (
                    [
                        "Qwen-0.5B (Smallest - ~1GB)",
                        "SmolLM2-1.7B (Recommended - ~3.5GB)"
                    ], 
                    {"default": "Qwen-0.5B (Smallest - ~1GB)"}
                ),
                "run_test": ("BOOLEAN", {"default": False, "label_on": "Run", "label_off": "Ready"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("diagnostic_report",)
    FUNCTION = "diagnose"
    CATEGORY = "🔐 Private AI"
    OUTPUT_NODE = True
    
    def diagnose(self, test_model, run_test):
        if not run_test:
            return ("ℹ️ Set 'Run Test' to True and queue prompt to start diagnostics.\n\nThis will test:\n- Internet connection\n- HuggingFace Hub access\n- Model directory permissions\n- Model download capability\n\nExpected time: 2-5 minutes",)
        
        results = []
        results.append("🔍 COMFYUI QWEN NODE DIAGNOSTICS")
        results.append("=" * 50)
        
        # Test 1: System Info
        results.append("\n📊 SYSTEM INFO:")
        results.append(f"PyTorch: {torch.__version__}")
        results.append(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            results.append(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            results.append(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        try:
            from transformers import __version__ as tf_version
            results.append(f"Transformers: {tf_version}")
        except:
            results.append("Transformers: NOT FOUND!")
        
        # Test 2: Internet
        results.append("\n🌐 TESTING INTERNET CONNECTION:")
        try:
            import socket
            socket.create_connection(("huggingface.co", 443), timeout=5)
            results.append("✅ Internet: Connected")
        except Exception as e:
            results.append(f"❌ Internet: FAILED - {e}")
            results.append("💡 Check your internet connection")
            return ("\n".join(results),)
        
        # Test 3: HuggingFace Hub
        results.append("\n🤗 TESTING HUGGINGFACE HUB:")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.list_models(limit=1)
            results.append("✅ HuggingFace Hub: Accessible")
        except Exception as e:
            results.append(f"⚠️ HuggingFace Hub: {e}")
            results.append("💡 Hub may be slow but should work")
        
        # Test 4: Model Directory
        model_dir = os.path.join(folder_paths.base_path, "models", "LLM")
        results.append(f"\n📁 TESTING MODEL DIRECTORY:")
        results.append(f"Path: {model_dir}")
        
        if not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir, exist_ok=True)
                results.append("✅ Directory created successfully")
            except Exception as e:
                results.append(f"❌ Cannot create directory: {e}")
                results.append("💡 Check permissions or use different location")
                return ("\n".join(results),)
        else:
            results.append("✅ Directory exists")
        
        # Test 5: Write Permissions
        test_file = os.path.join(model_dir, ".diagnostic_test")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            results.append("✅ Write permissions: OK")
        except Exception as e:
            results.append(f"❌ Write permissions: FAILED - {e}")
            results.append("💡 Run ComfyUI with appropriate permissions")
            return ("\n".join(results),)
        
        # Test 6: Disk Space
        try:
            import shutil
            total, used, free = shutil.disk_usage(model_dir)
            free_gb = free / (2**30)
            results.append(f"💾 Free disk space: {free_gb:.1f}GB")
            if free_gb < 10:
                results.append(f"⚠️ Warning: Less than 10GB free!")
                results.append(f"💡 Models need 3-10GB each")
        except:
            results.append("⚠️ Could not check disk space")
        
        # Test 7: Download Model
        model_ids = {
            "Qwen-0.5B (Smallest - ~1GB)": "Qwen/Qwen2.5-0.5B-Instruct",
            "SmolLM2-1.7B (Recommended - ~3.5GB)": "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        }
        
        model_id = model_ids[test_model]
        results.append(f"\n📥 TESTING MODEL DOWNLOAD:")
        results.append(f"Model: {model_id}")
        results.append(f"⏳ Downloading... (this may take 2-5 minutes)")
        results.append(f"")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Download tokenizer first (small)
            results.append("📦 Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=model_dir
            )
            results.append("✅ Tokenizer downloaded")
            
            # Download model (large)
            results.append("📦 Downloading model (this is the big one)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=model_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            results.append("✅ Model downloaded successfully!")
            
            # Quick test
            results.append("\n🧪 TESTING MODEL INFERENCE:")
            test_input = tokenizer("test", return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**test_input, max_new_tokens=5)
            results.append("✅ Model can generate text!")
            
            # Cleanup
            del model
            del tokenizer
            gc.collect()
            
            results.append("\n" + "=" * 50)
            results.append("🎉 ALL TESTS PASSED!")
            results.append("✅ Your system is ready to use Qwen Node!")
            results.append("")
            results.append("Next steps:")
            results.append("1. Use the main Qwen Prompt Expander node")
            results.append("2. Select your preferred model")
            results.append("3. Start generating prompts!")
            
        except Exception as e:
            results.append(f"\n❌ MODEL DOWNLOAD FAILED!")
            results.append(f"Error: {e}")
            results.append("")
            results.append("💡 TROUBLESHOOTING:")
            results.append("1. Check internet connection")
            results.append("2. Try VPN if HuggingFace is blocked")
            results.append("3. Check firewall/antivirus settings")
            results.append("4. Ensure 10GB+ free disk space")
            results.append("5. Try again (sometimes servers are slow)")
            results.append("")
            results.append("For manual download:")
            results.append(f"Visit: https://huggingface.co/{model_id}")
            results.append("Download files to: " + model_dir)
        
        return ("\n".join(results),)


# Rejestracja node'ów
NODE_CLASS_MAPPINGS = {
    "QwenOfflinePrompt": QwenOfflinePrompt,
    "QwenDiagnostics": QwenDiagnostics
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenOfflinePrompt": "🔐 Qwen Prompt Expander",
    "QwenDiagnostics": "🔍 Qwen Diagnostics and Downloader"
}
