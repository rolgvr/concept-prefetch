"""Segmented KV store: one KV segment per fact sheet at a fixed RoPE slot, CPU tier + GPU budget."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

SYS_PREFIX = (
    "<|im_start|>system\nYou are a precise assistant. Answer questions using only the fact sheets below.\n"
)
QUERY_TMPL = "<|im_end|>\n<|im_start|>user\n{q} Answer with just the value.<|im_end|>\n<|im_start|>assistant\n"


def _sync():
    torch.cuda.synchronize()


class Engine:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", device="cuda"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device).eval()
        self.stop_ids = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id}
        sys_ids = self._ids(SYS_PREFIX)
        self.S = sys_ids.shape[1]
        with torch.no_grad():
            out = self.model(sys_ids, use_cache=True)
        self.sys_kv = [(l.keys, l.values) for l in out.past_key_values.layers]

    def _ids(self, text):
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

    # ---- segment construction -------------------------------------------------------------------
    def build_segments(self, sheets):
        """Prefill every sheet at its own slot; keep the KV in pinned CPU memory as one contiguous tensor."""
        self.names = list(sheets)
        self.seg_ids = {n: self._ids(sheets[n]) for n in self.names}
        self.slot = max(t.shape[1] for t in self.seg_ids.values()) + 8
        self.slot_start = {n: self.S + i * self.slot for i, n in enumerate(self.names)}
        self.query_start = self.S + len(self.names) * self.slot
        self.cpu = {}
        for n in self.names:
            kv = self._prefill_segment(n)  # [L, 2, B, H, T, D] on GPU
            self.cpu[n] = kv.cpu().pin_memory()
        self.seg_bytes = {n: self.cpu[n].numel() * self.cpu[n].element_size() for n in self.names}
        self.seg_tokens = {n: self.seg_ids[n].shape[1] for n in self.names}

    @torch.no_grad()
    def _prefill_segment(self, n):
        ids = self.seg_ids[n]
        cache = DynamicCache(ddp_cache_data=[(k.clone(), v.clone()) for k, v in self.sys_kv])
        pos = torch.arange(ids.shape[1], device=self.device).unsqueeze(0) + self.slot_start[n]
        out = self.model(ids, position_ids=pos, past_key_values=cache, use_cache=True)
        return torch.stack(
            [torch.stack([l.keys[:, :, self.S:], l.values[:, :, self.S:]]) for l in out.past_key_values.layers]
        )

    # ---- miss handling (both tiers are measured on the critical path) ------------------------
    def reload(self, n):
        _sync()
        t0 = time.perf_counter()
        kv = self.cpu[n].to(self.device, non_blocking=True)
        _sync()
        return kv, (time.perf_counter() - t0) * 1e3

    def recompute(self, n):
        _sync()
        t0 = time.perf_counter()
        kv = self._prefill_segment(n)
        _sync()
        return kv, (time.perf_counter() - t0) * 1e3

    # ---- answering ----------------------------------------------------------------------------
    def _assemble(self, gpu_segs):
        layers = []
        for li, (k, v) in enumerate(self.sys_kv):
            ks = [k] + [gpu_segs[n][li, 0] for n in gpu_segs]
            vs = [v] + [gpu_segs[n][li, 1] for n in gpu_segs]
            layers.append((torch.cat(ks, dim=2), torch.cat(vs, dim=2)))
        return DynamicCache(ddp_cache_data=layers)

    @torch.no_grad()
    def answer(self, gpu_segs, question, max_new=16):
        """Return (query_ttft_ms, text). TTFT covers the query prefill and the first token; cache assembly
        is excluded (identical across methods at a given k, and a paged-attention server would skip it)."""
        cache = self._assemble(gpu_segs)
        ids = self._ids(QUERY_TMPL.format(q=question))
        pos = torch.arange(ids.shape[1], device=self.device).unsqueeze(0) + self.query_start
        _sync()
        t0 = time.perf_counter()
        out = self.model(ids, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = out.logits[:, -1].argmax(-1)
        _sync()
        ttft = (time.perf_counter() - t0) * 1e3
        toks, p = [], pos[0, -1].item()
        for _ in range(max_new):
            t = nxt.item()
            if t in self.stop_ids:
                break
            toks.append(t)
            p += 1
            out = self.model(nxt.view(1, 1), position_ids=torch.tensor([[p]], device=self.device),
                             past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
            nxt = out.logits[:, -1].argmax(-1)
        return ttft, self.tok.decode(toks).strip()
