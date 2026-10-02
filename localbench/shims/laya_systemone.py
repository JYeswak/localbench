"""A loopback POST /v1/systemone for one Laya-MLX checkpoint, so `localbench decision run laya:<repo>[@<subfolder>]`
screens it with the same client, validation and metrics as Ollama's System One.

Run by the laya venv's python3 (localbench/decision.py LayaShim starts and stops it); localbench never imports this
file, and it lives outside localbench/*.py because it imports laya_mlx, mlx and numpy from that venv. The checkpoint is
loaded once, from the local Hugging Face cache only (the caller sets HF_HUB_OFFLINE=1: no download, no cloud), before
the port is bound, so a GET /health that answers means the model is ready.

Wire shape (decision.build_request / validate_response): request {model, state, questions}; reply {model, answers,
usage}. `model` must name what this shim serves (the spec after `laya:`) and is echoed. Answers follow Laya's own
Agent.system_one (prepare -> collate -> forward -> per-bucket temperature -> softmax) with two deliberate differences:
probabilities stay unrounded (Laya rounds to 4 decimals, which breaks System One's sum-to-1 and score = sum(j * p_j)
checks at 1e-6) and the softmax runs in float64 on the model's logits (Laya's runs in the logits' float16). Each answer
also carries `laya`: {truncated: the state was cut to fit max_len, head_truncated: instructions or option text were
cut to fit head_max_len / 48 tokens per option, tokens: the untruncated sequence length, max_len}. The client's
answer validation ignores that field; decision.py counts it."""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import laya_mlx
import mlx.core as mx
import numpy as np
from laya_mlx.agent import collate_items, load
from laya_mlx.common import build_prefix, confidence_from_probs, render_options, serialize_state, temp_bucket

MAX_BODY = 1 << 20


def measure(agent, state_ids: list, q: dict) -> dict:
    """Truncation of one question's sequence as laya_mlx.common.build_sequence builds it: [CLS] head [SEP] ([MASK]
    option)* [SEP] state [SEP], cut to max_len, with head and options first cut to head_max_len."""
    tok, max_len = agent.tok, agent.cfg.get("max_len", 512)
    head_max = agent.cfg.get("head_max_len", 192)
    prefix, _ = build_prefix(tok, q, head_max)
    mask = tok.mask_token
    head = tok("%s question: %s" % (q["t"], str(q["ins"]).replace(mask, " ")))["input_ids"]
    options = [tok(" " + o.replace(mask, " "))["input_ids"] for o in render_options(q)]
    full_prefix = 1 + len(head) + 1 + sum(1 + len(o) for o in options) + 1
    room = max(0, max_len - len(prefix) - 1)
    return {"truncated": len(state_ids) > room, "head_truncated": len(prefix) < full_prefix,
            "tokens": full_prefix + len(state_ids) + 1, "max_len": max_len}


def answer_all(agent, state, questions: dict) -> tuple[dict, int]:
    items, internal = agent.prepare(state, questions)
    tok = agent.tok
    state_ids = tok(serialize_state(state).replace(tok.mask_token, " "))["input_ids"]
    names, answers = list(questions), {}
    for start in range(0, len(items), agent.batch_size):
        chunk = items[start:start + agent.batch_size]
        batch = collate_items(chunk, tok.pad_token_id, pad_to_multiple=agent.pad_to_multiple,
                              max_length=agent.cfg.get("max_len", 512))
        logits, _act = agent.forward(batch)
        logits = np.asarray(logits).astype(np.float64)
        if not np.isfinite(logits).all():
            raise FloatingPointError("non-finite model outputs")
        for row, item in enumerate(chunk):
            name, q = names[start + row], internal[start + row]
            k, qt = len(item["markers"]), item["qtype"]
            scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
            z = logits[row, :k] / max(1e-3, float(scale))
            p = np.exp(z - z.max())
            p /= p.sum()
            ps = [float(v) for v in p]
            if q["t"] == "noul":
                a = {"type": "noul", "noul": ps[1], "confidence": max(ps[1], 1.0 - ps[1])}
            elif q["t"] == "choice":
                labels = list(q["crit"])
                a = {"type": "choice", "choice": labels[int(np.argmax(p))],
                     "probabilities": dict(zip(labels, ps, strict=True)),
                     "confidence": confidence_from_probs(p, k)}
            else:
                keys = [str(i) for i in range(k)]
                a = {"type": "score", "score": sum(j * ps[j] for j in range(k)),
                     "legend": {key: c if isinstance(c, str) else json.dumps(c) for key, c in zip(keys, q["crit"],
                                                                                              strict=True)},
                     "probabilities": dict(zip(keys, ps, strict=True)), "confidence": confidence_from_probs(p, k)}
            a["laya"] = measure(agent, state_ids, q)
            answers[name] = a
    return answers, sum(len(item["ids"]) for item in items)


def serve(agent, name: str, port: int, health: dict) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status: int, doc: dict) -> None:
            data = json.dumps(doc, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, health)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            size = int(self.headers.get("Content-Length") or 0)
            if not 0 < size <= MAX_BODY:
                self._send(400, {"error": f"body of {size} bytes; 1-{MAX_BODY} accepted"})
                return
            try:
                body = json.loads(self.rfile.read(size))
            except ValueError as exc:
                self._send(400, {"error": f"body is not JSON: {exc}"})
                return
            if not isinstance(body, dict) or body.get("model") != name:
                self._send(404, {"error": f"model {body.get('model') if isinstance(body, dict) else None!r} not "
                                          f"found; this shim serves {name!r}"})
                return
            questions = body.get("questions")
            if not isinstance(questions, dict) or not questions or body.get("state") in (None, "", {}, []):
                self._send(400, {"error": "state and a non-empty questions object are required"})
                return
            try:
                answers, tokens = answer_all(agent, body["state"], questions)
            except ValueError as exc:   # laya's own refusals: unknown type, bad criteria, options over the budget
                self._send(400, {"error": str(exc)})
                return
            except Exception as exc:   # noqa: BLE001 - any other failure is this request's 500, the shim stays up
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._send(200, {"model": name, "answers": answers,
                             "usage": {"input_tokens": tokens, "output_tokens": 0}})

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(json.dumps({"event": "ready", **health}), flush=True)
    server.serve_forever()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="Hugging Face repo id, e.g. aac6fef/laya-mlx")
    ap.add_argument("--subfolder")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args(argv)
    name = args.model + (f"@{args.subfolder}" if args.subfolder else "")
    agent = load(args.model, subfolder=args.subfolder, dtype=args.dtype)
    health = {"ready": True, "model": name, "laya_version": laya_mlx.__version__, "laya_file": laya_mlx.__file__,
              "mlx_version": mx.__version__, "model_dir": str(agent.model_dir), "dtype": args.dtype,
              "batch_size": agent.batch_size, "max_len": agent.cfg.get("max_len", 512),
              "head_max_len": agent.cfg.get("head_max_len", 192)}
    serve(agent, name, args.port, health)
    return 0


if __name__ == "__main__":
    sys.exit(main())
