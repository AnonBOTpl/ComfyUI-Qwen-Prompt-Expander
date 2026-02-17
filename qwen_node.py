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
                "model_name": (
                    [
                        "HuggingFaceTB/SmolLM2-1.7B-Instruct (New & Smart)",
                        "cognitivecomputations/dolphin-2.9.4-qwen2.5-1.5b (Uncensored)",
                        "Qwen/Qwen2.5-0.5B-Instruct (Ultra Light)",
                        "Qwen/Qwen2.5-1.5B-Instruct (Balanced)"
                    ], 
                    {"default": "HuggingFaceTB/SmolLM2-1.7B-Instruct (New & Smart)"}
                ),
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
            
            # Ładowanie tylko jeśli nie ma w cache
            if self.trans_model is None:
                print(f"🌍 [Translator] Loading to {trans_device.upper()}...")
                self.trans_tokenizer = MarianTokenizer.from_pretrained(
                    "Helsinki-NLP/opus-mt-pl-en", 
                    cache_dir=self.model_dir
                )
                self.trans_model = MarianMTModel.from_pretrained(
                    "Helsinki-NLP/opus-mt-pl-en", 
                    cache_dir=self.model_dir
                ).to(trans_device)
            
            # Jeśli zmienił się device, przenieś model
            if str(next(self.trans_model.parameters()).device) != trans_device:
                print(f"🔄 [Translator] Moving to {trans_device.upper()}...")
                self.trans_model = self.trans_model.to(trans_device)
            
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

    def generate(self, model_name, precision, subject, negative_prompt_text, style, lighting, quality, 
                 max_tokens, emb_positive, emb_negative, device_mode, translator_device, 
                 use_translator_PL_EN, unload_model, add_negative, seed):
        
        actual_model_id = model_name.split(" ")[0]
        
        target_device = "cpu"
        if "GPU" in device_mode and comfy.model_management.get_torch_device().type == "cuda":
            target_device = "cuda"

        # 1. Tłumaczenie
        final_subject = subject
        if use_translator_PL_EN:
            final_subject = self.translate(subject, translator_device)

        # 2. VRAM Management
        model_changed = (self.llm_model is None) or \
                        (self.current_model_name != actual_model_id) or \
                        (self.current_precision != precision)

        if target_device == "cuda" and model_changed:
            print("🧹 [LLM] Clearing VRAM for new model...")
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

        # 3. Load Model
        if model_changed:
            print(f"🔐 [LLM] Loading {actual_model_id} ({precision})...")
            
            # Usuń stary jeśli istnieje
            if self.llm_model is not None:
                del self.llm_model
                del self.llm_tokenizer
                gc.collect()

            try:
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    actual_model_id, 
                    cache_dir=self.model_dir
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
                    cache_dir=self.model_dir
                )
                
                if quantization_config is None and target_device == "cuda":
                    self.llm_model = self.llm_model.to(target_device)

                self.current_model_name = actual_model_id
                self.current_precision = precision
                print("✅ [LLM] Model loaded successfully")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"💥 [LLM] OUT OF MEMORY! Try: 4-bit precision or CPU mode")
                    print(f"💡 Tip: Enable 'unload_model' to free VRAM after generation")
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
                print(f"❌ [LLM] Load Error: {e}")
                return (final_subject, negative_prompt_text if add_negative else "")

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
            text_input = self.llm_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            model_inputs = self.llm_tokenizer([text_input], return_tensors="pt").to(self.llm_model.device)

            # 5. Generowanie
            print(f"🎨 Generating prompt for: {final_subject[:50]}...")
            with torch.no_grad():
                generated_ids = self.llm_model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=generation_buffer,  # 20% więcej przestrzeni
                    do_sample=True,
                    temperature=0.8,
                    repetition_penalty=1.2,
                    pad_token_id=self.llm_tokenizer.eos_token_id
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

        # 7. SMART TOKEN TRIMMING (ultra fast - proporcjonalny cut)
        token_count = len(self.llm_tokenizer.encode(clean_response))
        
        if token_count > max_tokens:
            print(f"✂️ [LLM] Response too long ({token_count} tokens), trimming to {max_tokens}...")
            
            tags = [t.strip() for t in clean_response.split(",") if t.strip()]
            
            # Inteligentne pierwszeństwo tagów
            subject_keywords = set(final_subject.lower().split())
            important_tags = []
            other_tags = []
            
            for tag in tags:
                tag_lower = tag.lower()
                # Jeśli tag zawiera słowo z subject, jest priorytetowy
                if any(keyword in tag_lower for keyword in subject_keywords):
                    important_tags.append(tag)
                else:
                    other_tags.append(tag)
            
            # Oblicz ile tagów możemy zmieścić
            trim_ratio = (max_tokens / token_count) * 0.95  # 5% safety margin
            total_tags = len(tags)
            keep_count = max(1, int(total_tags * trim_ratio))
            
            # Zawsze zachowaj przynajmniej niektóre important_tags
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

        # Fallback - upewnij się że główny subject jest w prompcie
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
            
            # Czyścimy LLM
            if self.llm_model is not None:
                del self.llm_model
                del self.llm_tokenizer
                self.llm_model = None
                self.llm_tokenizer = None
            
            # Czyścimy Translatora
            if self.trans_model is not None:
                del self.trans_model
                del self.trans_tokenizer
                self.trans_model = None
                self.trans_tokenizer = None
            
            gc.collect()
            if target_device == "cuda":
                torch.cuda.empty_cache()
            comfy.model_management.soft_empty_cache()

        return (clean_response, neg)

# Rejestracja node'a
NODE_CLASS_MAPPINGS = {
    "QwenOfflinePrompt": QwenOfflinePrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenOfflinePrompt": "🔐 Qwen Prompt Expander (Offline)"
}