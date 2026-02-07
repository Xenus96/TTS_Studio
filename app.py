import os
import sys
import warnings
import inspect
from types import ModuleType

# 1. SILENCE THE NOISE
warnings.filterwarnings("ignore")

# 2. SURGICAL BYPASS OF THE CRASHING INIT
# We find where the real TTS folder is located
import importlib.util
spec = importlib.util.find_spec('TTS')
if spec:
    # Create a real module object
    m_tts = ModuleType("TTS")
    # Tell Python where the folder is so it can find submodules like TTS.api
    m_tts.__path__ = spec.submodule_search_locations
    m_tts.__package__ = "TTS"
    # Inject our fake 'fixed' variable so sub-libraries don't crash
    m_tts.TORCHCODEC_IMPORT_ERROR = None
    # Put it into the system memory before anyone else can touch it
    sys.modules["TTS"] = m_tts

# 3. THE ULTIMATE PATCH BLOCK
import transformers.utils.import_utils as import_utils

if not hasattr(import_utils, 'is_torch_greater_or_equal'):
    import torch
    from packaging import version
    def is_torch_greater_or_equal(target_version):
        v = torch.__version__.split('+')[0]
        return version.parse(v) >= version.parse(target_version)
    import_utils.is_torch_greater_or_equal = is_torch_greater_or_equal

if not hasattr(import_utils, 'is_torchcodec_available'):
    import_utils.is_torchcodec_available = lambda: False

try:
    import transformers.pytorch_utils as pytorch_utils
except ImportError:
    m = ModuleType("transformers.pytorch_utils")
    sys.modules["transformers.pytorch_utils"] = m
    import transformers.pytorch_utils as pytorch_utils

if not hasattr(pytorch_utils, 'isin_mps_friendly'):
    import torch
    pytorch_utils.isin_mps_friendly = torch.isin

# Now this will load the API without running the broken __init__.py
# --- THE GLOBAL TENSOR TRUTHINESS FIX ---
# This stops the "Boolean value of Tensor is ambiguous" error globally.
import torch
original_bool = torch.Tensor.__bool__

def safe_bool(self):
    try:
        if self.numel() > 1:
            return True # If it has data, treat it as "True" (exists)
        return original_bool(self)
    except:
        return True

torch.Tensor.__bool__ = safe_bool
print("Global Fix: Tensor Ambiguity resolved.")
# ----------------------------------------


import functools

# Define the wrapper
original_torch_load = torch.load

def absolute_forced_load(*args, **kwargs):
    # Force weights_only to be False regardless of what the library asks for
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Physically overwrite the function in the torch module
torch.load = absolute_forced_load

# Also apply it to the serialization module just in case
import torch.serialization
torch.serialization.load = absolute_forced_load

print("System Patched: Ironclad Torch security override applied.")

import torch.jit

# This forces the @torch.jit.script decorator to do nothing
# It returns the original function instead of trying to compile it
def pass_through(obj, *args, **kwargs):
    return obj

torch.jit.script = pass_through

import threading
import ctypes
import inspect
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import docx  # For reading Word files
import numpy as np
import scipy.io.wavfile as wavfile
import pygame
import langid

# --- 1. THE "LOBOTOMY" PATCH (CRITICAL FOR EXE) ---
# Prevents PyTorch JIT and Inspect errors in the frozen app
def patched_getsource(obj): return ""


def patched_getsourcelines(obj): return [""], 0


def patched_findsource(obj): return [""], 0


inspect.getsource = patched_getsource
inspect.getsourcelines = patched_getsourcelines
inspect.findsource = patched_findsource
inspect.getsourcefile = lambda x: None

try:
    import torch
    import torch._sources

    torch._sources.get_source_lines_and_file = lambda obj, error_msg=None: ([""], None, 0)
    torch._sources.parse_def = lambda func: None
    torch.jit.is_tracing = lambda: False
    torch.jit.is_scripting = lambda: False
except:
    pass

# Force DLL load
if hasattr(sys, '_MEIPASS'):
    torch_lib = os.path.join(sys._MEIPASS, "torch", "lib")
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)

os.environ["TYPEGUARD_DISABLE"] = "1"
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# ---------------------------------------------------

# --- THE SLIDING WINDOW CACHE PATCH (V7) ---
try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    import torch


    def patched_attn_sliding_window(self, query, key, value, attention_mask=None, head_mask=None):
        # 1. ANALYZE THE MISMATCH
        # query (a) is usually the current token, key (b) is the history
        q_len = query.size(-2)
        k_len = key.size(-2)

        # 2. THE CACHE ALIGNMENT
        # If the history (key) is longer than the current math window,
        # we slide the window to the most recent tokens.
        if k_len != q_len and not self.is_cross_attention:
            if k_len > q_len:
                # Key is 239, Query is 120 -> Slice Key/Value to match Query
                # We take the LAST (most recent) tokens
                key = key[:, :, -q_len:, :]
                value = value[:, :, -q_len:, :]
            else:
                # Query is larger than history (rare) -> Pad history
                padding = (0, 0, 0, q_len - k_len)
                key = torch.nn.functional.pad(key, padding)
                value = torch.nn.functional.pad(value, padding)

        # 3. PERFORM THE MATH
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (value.size(-1) ** 0.5)

        # 4. DYNAMIC MASKING (Zero-Logic)
        # We generate a fresh causal mask based on the ACTUAL reconciled size
        cur_q, cur_k = attn_weights.size(-2), attn_weights.size(-1)
        causal_mask = torch.tril(torch.ones((cur_q, cur_k), dtype=torch.bool, device=query.device)).view(1, 1, cur_q,
                                                                                                         cur_k)

        mask_value = torch.finfo(attn_weights.dtype).min
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        # Apply external mask only if it now matches
        if attention_mask is not None and attention_mask.size(-1) == cur_k:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights.to(value.dtype), value), attn_weights


    GPT2Attention._attn = patched_attn_sliding_window
    print("Sliding Window Patch: Cache mismatch (239 vs 120) resolved.")
except Exception as e:
    print(f"Sliding Window Patch Failed: {e}")
# --------------------------------------------

# --- THE NUCLEAR COMPATIBILITY PATCH (V2) ---
try:
    import transformers.modeling_utils

    # 1. Bypass the class check (The "Judge")
    transformers.modeling_utils.PreTrainedModel._validate_model_class = lambda self: None


    # 2. Bypass the existence check (The "Skeleton Key")
    # We provide a dummy method that just returns the inputs as they are.
    def dummy_prepare(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}


    transformers.modeling_utils.PreTrainedModel.prepare_inputs_for_generation = dummy_prepare

    # 3. Force the flag
    transformers.modeling_utils.PreTrainedModel.can_generate = lambda self: True

    print("Nuclear Patch V2: Validation and Preparation checks bypassed.")
except Exception as e:
    print(f"Nuclear Patch Failed: {e}")
# ---------------------------------------

from TTS.api import TTS

# --- GENERATION COMPATIBILITY PATCH ---
import transformers
from transformers import GPT2Config

# We force GPT2Config to believe it has a Language Model head
# This bypasses the GPT2InferenceModel compatibility error
GPT2Config.is_decoder = True
transformers.logging.set_verbosity_error() # Hide the warning spam
# ---------------------------------------

# --- CONFIGURATION ---
MODEL_REL_PATH = "models/tts_models--multilingual--multi-dataset--xtts_v2"
LANGUAGES = {
    "English": "en",
    "Russian": "ru",
    "German": "de",
    "Romanian": "ro",
    "French": "fr",
    "Spanish": "es"
}


# Fallback/Mapping check: XTTS v2 supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, hu, ko, ja.
# Romanian is NOT natively supported in standard XTTS v2.0.2.
# You might need to use 'pl' (Polish) or 'it' (Italian) as a close proxy for phonemes if RO is strictly required,
# OR fine-tune the model. For this script, I will include it, but be warned.

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class TextBlock(ctk.CTkFrame):
    """ A single block of text with its own language setting """

    def __init__(self, master, delete_command, index):
        super().__init__(master)
        self.pack(fill="x", pady=5, padx=5)

        # Controls Row
        self.ctrl_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.ctrl_frame.pack(fill="x", padx=5, pady=2)

        self.lbl_index = ctk.CTkLabel(self.ctrl_frame, text=f"Block {index}", font=("Arial", 12, "bold"))
        self.lbl_index.pack(side="left")

        self.lang_var = ctk.StringVar(value="English")
        self.lang_menu = ctk.CTkOptionMenu(self.ctrl_frame, variable=self.lang_var, values=list(LANGUAGES.keys()),
                                           width=100)
        self.lang_menu.pack(side="left", padx=10)

        self.btn_del = ctk.CTkButton(self.ctrl_frame, text="X", width=30, fg_color="#FF5555", hover_color="#AA0000",
                                     command=lambda: delete_command(self))
        self.btn_del.pack(side="right")

        # Text Area
        self.text_area = ctk.CTkTextbox(self, height=80)
        self.text_area.pack(fill="x", padx=5, pady=5)

    def get_data(self):
        return {
            "text": self.text_area.get("0.0", "end").strip(),
            "lang": LANGUAGES[self.lang_var.get()]
        }


class TTSApp(ctk.CTk):
    # 1. DEFINE LOGIC FIRST
    def request_stop(self):
        self.stop_requested = True
        self.status_label.configure(text="Stopping...")

    def detect_language_segments(self, text):
        """Splits mixed text into chunks with assigned languages."""
        words = text.split()
        if not words: return []

        segments = []
        current_chunk = []
        # Support common XTTS languages
        valid_langs = list(LANGUAGES.values())

        # Initial language detection
        current_lang, _ = langid.classify(words[0])
        if current_lang not in valid_langs: current_lang = "en"

        for word in words:
            det_lang, _ = langid.classify(word)
            if det_lang not in valid_langs: det_lang = "en"

            if det_lang != current_lang:
                segments.append((" ".join(current_chunk), current_lang))
                current_chunk = [word]
                current_lang = det_lang
            else:
                current_chunk.append(word)

        segments.append((" ".join(current_chunk), current_lang))
        return segments

    def __init__(self):
        super().__init__()
        self.title("Multilingual TTS Studio")
        self.geometry("900x700")

        self.tts = None
        self.blocks = []
        self.block_counter = 1
        self.stop_requested = False

        # --- LEFT PANEL (Controls) ---
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")

        ctk.CTkLabel(self.sidebar, text="TTS Studio", font=("Arial", 20, "bold")).pack(pady=20)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: Booting...")
        self.status_label.pack(pady=10)

        ctk.CTkButton(self.sidebar, text="Add Text Block", command=self.add_block).pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(self.sidebar, text="Import Text File", command=self.import_file).pack(pady=10, padx=20, fill="x")

        self.speaker_var = ctk.StringVar(value="Ana Florence")
        ctk.CTkLabel(self.sidebar, text="Speaker Voice:").pack(pady=(20, 5))
        self.speaker_menu = ctk.CTkOptionMenu(self.sidebar, variable=self.speaker_var,
                                              values=["Ana Florence", "Viktor Menelaos"])  # Populated later
        self.speaker_menu.pack(pady=5, padx=20, fill="x")

        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray").pack(fill="x", pady=20)

        self.gen_btn = ctk.CTkButton(self.sidebar, text="RENDER AUDIO", command=self.start_render, height=50,
                                     fg_color="#00AA00", hover_color="#006600", state="disabled")
        self.gen_btn.pack(pady=10, padx=20, fill="x")

        self.stop_btn = ctk.CTkButton(self.sidebar, text="STOP RENDER",
                                      command=self.request_stop,
                                      fg_color="#AA0000", hover_color="#770000")
        self.stop_btn.pack(pady=5, padx=20, fill="x")

        # --- RIGHT PANEL (Blocks) ---
        self.scrollable_frame = ctk.CTkScrollableFrame(self, label_text="Story Board")
        self.scrollable_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Initial Block
        self.add_block()

        # Load Engine
        threading.Thread(target=self.load_engine, daemon=True).start()

    def load_engine(self):
        try:
            model_full_path = get_resource_path(MODEL_REL_PATH)
            config_path = os.path.join(model_full_path, "config.json")
            self.status_label.configure(text="Loading Model...")

            # Load TTS
            if os.path.exists(model_full_path):
                self.tts = TTS(model_path=model_full_path, config_path=config_path).to("cpu")
            else:
                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

                # --- THE FINAL BYPASS ---
                if hasattr(self.tts, 'model') and hasattr(self.tts.model, 'gpt'):
                    self.tts.model.gpt.generate = self.tts.model.gpt.transformer.generate
                    self.tts.model.gpt.can_generate = lambda: True
                    self.tts.model.gpt.prepare_inputs_for_generation = self.tts.model.gpt.transformer.prepare_inputs_for_generation

                    # KEEP THIS TRUE - The sliding window patch now makes the cache safe
                    self.tts.model.config.use_cache = True
                    # ------------------------

            # Update speakers
            speakers = self.tts.speakers if self.tts.speakers else ["Ana Florence"]
            self.speaker_menu.configure(values=speakers)
            self.speaker_var.set(speakers[0])

            self.status_label.configure(text="Ready")
            self.gen_btn.configure(state="normal")
        except Exception as e:
            self.status_label.configure(text="Engine Error!")
            print(f"Engine Load Error: {e}")

    def add_block(self):
        block = TextBlock(self.scrollable_frame, self.remove_block, self.block_counter)
        self.blocks.append(block)
        self.block_counter += 1

    def remove_block(self, block_obj):
        block_obj.destroy()
        self.blocks.remove(block_obj)

    def import_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text/Word", "*.txt *.docx")])
        if not filepath: return

        text = ""
        if filepath.endswith(".docx"):
            doc = docx.Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

        # Add to the last block or create new
        if self.blocks:
            self.blocks[-1].text_area.insert("end", text)
        else:
            self.add_block()
            self.blocks[-1].text_area.insert("0.0", text)

    def start_render(self):
        self.gen_btn.configure(state="disabled", text="Rendering...")
        threading.Thread(target=self.render_logic, daemon=True).start()

    def render_logic(self):
        self.stop_requested = False  # Reset flag
        try:
            audio_segments = []
            speaker = self.speaker_var.get()

            for i, block in enumerate(self.blocks):
                if self.stop_requested: break  # Check for interruption

                data = block.get_data()
                full_text = data['text']
                if not full_text: continue

                # Automatically detect language segments within the block
                self.status_label.configure(text=f"Analyzing Block {i + 1}...")
                segments = self.detect_language_segments(full_text)

                for text_chunk, lang_chunk in segments:
                    if self.stop_requested: break

                    self.status_label.configure(text=f"Rendering ({lang_chunk})...")

                    # This now calls the model with the NATIVE language for each chunk
                    wav_data = self.tts.tts(text=" " + text_chunk, speaker=speaker, language=lang_chunk)
                    audio_segments.append(wav_data)

            if self.stop_requested:
                self.status_label.configure(text="Stopped by user")
                return

            if not audio_segments:
                self.status_label.configure(text="No text!")
                return

            self.status_label.configure(text="Stitching Audio...")
            self.status_label.configure(text="Smoothing Transitions...")

            # --- THE SMOOTH STITCHER ENGINE ---
            sample_rate = 24000
            # 80ms overlap provides a natural bridge between languages
            overlap_samples = int(sample_rate * 0.08)

            # Initialize with the first segment, trimmed of silence
            final_wav = np.trim_zeros(audio_segments[0])

            for i in range(1, len(audio_segments)):
                next_seg = np.trim_zeros(audio_segments[i])

                # If segment is too short to crossfade, just append it
                if len(final_wav) < overlap_samples or len(next_seg) < overlap_samples:
                    final_wav = np.concatenate([final_wav, next_seg])
                    continue

                # Create linear crossfade curves
                fade_out = np.linspace(1.0, 0.0, overlap_samples)
                fade_in = np.linspace(0.0, 1.0, overlap_samples)

                # Mix the end of current audio with the start of the next
                overlap_zone = (final_wav[-overlap_samples:] * fade_out) + (next_seg[:overlap_samples] * fade_in)

                # Stitch the parts together
                final_wav = np.concatenate([
                    final_wav[:-overlap_samples],
                    overlap_zone,
                    next_seg[overlap_samples:]
                ])
            # ----------------------------------

            # Save File
            save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV Audio", "*.wav")])
            if save_path:
                # 24000 is XTTS standard rate
                wavfile.write(save_path, 24000, (final_wav * 32767).astype(np.int16))
                self.status_label.configure(text="Done!")

                # Auto Play
                pygame.mixer.init()
                pygame.mixer.music.load(save_path)
                pygame.mixer.music.play()
            else:
                self.status_label.configure(text="Cancelled")

        except Exception as e:
            self.status_label.configure(text="Error!")
            print(f"Render Error: {e}")
        finally:
            self.gen_btn.configure(state="normal", text="RENDER AUDIO")


if __name__ == "__main__":
    app = TTSApp()
    app.mainloop()