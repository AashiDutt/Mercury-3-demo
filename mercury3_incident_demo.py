"""Mercury 3 incident-analysis demo.

Modes:
  dry-run    print the exact messages that would be sent
  probe      dump raw API response/chunk structure (diagnostics)
  compare    non-streaming low vs high reasoning effort
  standard   normal streaming
  diffusion  diffusion streaming (diffusing=True), saves every refinement state
"""

import argparse
import json
import os
import time
from pathlib import Path


MODEL = "mercury-3"

# Inception's quick-start uses 8192. Observed (not documented): a cap that is
# too low truncates the visible answer, and at high effort can leave no answer
# text at all -- a 1600 cap returned 1535 completion tokens and content=None.
DEFAULT_MAX_TOKENS = 8192

SYSTEM = """You are an SRE incident-analysis assistant.
Use only the evidence in the supplied incident packet.
Separate observations from hypotheses.
If the evidence is insufficient, say so explicitly."""

TASK = """Analyze this production incident.

Return:
1. The most likely root cause.
2. Three pieces of supporting evidence from the incident packet.
3. Two plausible alternative explanations.
4. The next three debugging or mitigation steps, in priority order.

Keep the answer concise and technical."""


def load_incident(path: str) -> dict:
    return json.loads(Path(path).read_text())


# A packet may carry its own prompts, so one harness can drive more than one
# scenario. These keys are instructions to us, not evidence, so they are
# stripped out before the packet is shown to the model.
PROMPT_KEYS = ("system_prompt", "task_prompt", "packet_label")


def build_messages(incident: dict):
    system = incident.get("system_prompt", SYSTEM)
    task = incident.get("task_prompt", TASK)
    label = incident.get("packet_label", "INCIDENT PACKET")
    evidence = {k: v for k, v in incident.items() if k not in PROMPT_KEYS}
    packet = json.dumps(evidence, indent=2)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{task}\n\n{label}:\n{packet}"},
    ]


def usage_dict(response) -> dict:
    """Return usage as a plain dict, including any reasoning-token fields."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage)


def extract_diffusion_meta(chunk):
    """Return the chunk's diffusion_meta as a plain dict, if present.

    Undocumented, but Mercury attaches it to diffusing=True chunks:
    {"diffusion_content": true, "diffusion_progress": 0.0}
    """
    meta = getattr(chunk, "diffusion_meta", None)
    if meta is None:
        meta = (getattr(chunk, "model_extra", None) or {}).get("diffusion_meta")
    if meta is None:
        return None
    return meta.model_dump() if hasattr(meta, "model_dump") else dict(meta)


def extract_delta_text(chunk) -> str:
    """Pull text out of a streaming chunk.

    delta.content is the documented field, but diffusion chunks may carry the
    current full answer in an extra field the SDK model does not declare
    (the SDK keeps unknown fields because its BaseModel allows extras).
    """
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = choices[0].delta

    content = getattr(delta, "content", None)
    if content:
        return content

    extra = getattr(delta, "model_extra", None) or {}
    for key, value in extra.items():
        if isinstance(value, str) and value:
            return value
    return ""


def run_nonstream(client, messages, effort: str, max_tokens: int):
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        reasoning_effort=effort,
        temperature=0.75,
        max_completion_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - start

    choice = response.choices[0]
    text = choice.message.content or ""
    usage = usage_dict(response)
    completion_tokens = usage.get("completion_tokens")
    effective_tps = (completion_tokens / elapsed) if completion_tokens else None

    return {
        "mode": "nonstream",
        "effort": effort,
        "max_completion_tokens": max_tokens,
        "elapsed_s": elapsed,
        "finish_reason": getattr(choice, "finish_reason", None),
        "usage": usage,
        "completion_tokens": completion_tokens,
        "effective_output_tokens_per_s": effective_tps,
        "text_chars": len(text),
        "text": text,
    }


def run_standard_stream(client, messages, effort: str, max_tokens: int):
    start = time.perf_counter()
    first_output = None
    finish_reason = None
    chunks = []

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        reasoning_effort=effort,
        temperature=0.75,
        max_completion_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if choices and getattr(choices[0], "finish_reason", None):
            finish_reason = choices[0].finish_reason

        content = extract_delta_text(chunk)
        if content:
            now = time.perf_counter()
            if first_output is None:
                first_output = now
            chunks.append(content)
            print(content, end="", flush=True)

    end = time.perf_counter()
    print()
    return {
        "mode": "standard_stream",
        "effort": effort,
        "max_completion_tokens": max_tokens,
        "time_to_first_output_s": (first_output - start) if first_output else None,
        "total_s": end - start,
        "num_chunks": len(chunks),
        "finish_reason": finish_reason,
        "final_text": "".join(chunks),
    }


def run_diffusion_stream(client, messages, effort: str, max_tokens: int,
                         live: bool = True):
    start = time.perf_counter()
    first_output = None
    finish_reason = None
    states = []
    empty_chunks = 0

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        reasoning_effort=effort,
        temperature=0.75,
        max_completion_tokens=max_tokens,
        stream=True,
        diffusing=True,
    )

    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if choices and getattr(choices[0], "finish_reason", None):
            finish_reason = choices[0].finish_reason

        content = extract_delta_text(chunk)
        if not content:
            empty_chunks += 1
            continue

        now = time.perf_counter()
        if first_output is None:
            first_output = now

        states.append({
            "t_s": now - start,
            "text": content,
            "diffusion_meta": extract_diffusion_meta(chunk),
        })

        if live:
            # ANSI clear-screen + cursor-home for a simple live terminal view.
            print("\033[2J\033[H", end="")
            print(content, flush=True)

    end = time.perf_counter()

    # Is each chunk the full current answer (replace) or a suffix (append)?
    semantics = "unknown"
    if len(states) >= 2:
        a, b = states[-2]["text"], states[-1]["text"]
        if b.startswith(a) and len(b) > len(a):
            semantics = "cumulative (each chunk contains the previous one)"
        elif len(b) >= len(a) * 0.5:
            semantics = "full-state replace (each chunk is the whole answer)"
        else:
            semantics = "incremental suffix (append)"

    return {
        "mode": "diffusion_stream",
        "effort": effort,
        "max_completion_tokens": max_tokens,
        "time_to_first_output_s": (first_output - start) if first_output else None,
        "total_s": end - start,
        "num_refinement_states": len(states),
        "num_empty_chunks": empty_chunks,
        "chunk_semantics": semantics,
        "diffusion_progress": [
            (st["diffusion_meta"] or {}).get("diffusion_progress")
            for st in states
        ],
        "finish_reason": finish_reason,
        "final_text": states[-1]["text"] if states else "",
        "states": states,
    }


def run_probe(client, messages, max_tokens: int, out_path: str):
    """Dump raw response/chunk structure so we can see where text actually lives."""
    report = {}

    for effort in ["low", "high"]:
        t = time.perf_counter()
        r = client.chat.completions.create(
            model=MODEL, messages=messages, reasoning_effort=effort,
            temperature=0.75, max_completion_tokens=max_tokens,
        )
        d = r.model_dump()
        msg = d["choices"][0]["message"]
        content = msg.get("content") or ""
        report[f"nonstream_{effort}"] = {
            "elapsed_s": time.perf_counter() - t,
            "finish_reason": d["choices"][0].get("finish_reason"),
            "usage": d.get("usage"),
            "message_keys": sorted(msg.keys()),
            "content_len": len(content),
            "content_head": content[:300],
            "content_tail": content[-300:],
            "non_content_message_fields": {
                k: str(v)[:300] for k, v in msg.items() if k != "content" and v
            },
        }

    raw = []
    t = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL, messages=messages, reasoning_effort="low",
        temperature=0.75, max_completion_tokens=max_tokens,
        stream=True, diffusing=True,
    )
    for i, chunk in enumerate(stream):
        d = chunk.model_dump()
        raw.append({"i": i, "t_s": time.perf_counter() - t, "chunk": d})

    report["diffusing"] = {
        "chunk_count": len(raw),
        "total_s": time.perf_counter() - t,
        "first_3": [{"i": r["i"], "t_s": r["t_s"],
                     "chunk": json.loads(json.dumps(r["chunk"], default=str))}
                    for r in raw[:3]],
        "last_2": [{"i": r["i"], "t_s": r["t_s"],
                    "chunk": json.loads(json.dumps(r["chunk"], default=str))}
                   for r in raw[-2:]],
    }

    Path(out_path).write_text(json.dumps(report, indent=2, default=str))
    print(f"Probe written to {out_path}")
    print(json.dumps({k: v for k, v in report.items() if k != "diffusing"},
                     indent=2, default=str)[:3000])
    print("\ndiffusing chunk_count:", report["diffusing"]["chunk_count"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--incident", default="case_support.json")
    parser.add_argument(
        "--mode",
        choices=["dry-run", "probe", "compare", "standard", "diffusion"],
        default="dry-run",
    )
    parser.add_argument("--effort", default="low",
                        choices=["instant", "low", "medium", "high"])
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--repeat", type=int, default=1,
                        help="trials per configuration (all live modes). "
                             "Single runs vary a lot; use 5 for quotable numbers.")
    parser.add_argument("--save-states", default="diffusion_states_support.json")
    parser.add_argument("--results", default=None,
                        help="default: results_<mode>.json")
    parser.add_argument("--no-live", action="store_true",
                        help="diffusion mode: do not clear the screen between states")
    args = parser.parse_args()

    incident = load_incident(args.incident)
    messages = build_messages(incident)
    results_path = args.results or f"results_{args.mode}.json"

    if args.mode == "dry-run":
        print(json.dumps(messages, indent=2))
        return

    if not os.getenv("INCEPTION_API_KEY"):
        raise RuntimeError(
            "INCEPTION_API_KEY is not set. Export it before running a live demo."
        )

    from inceptionai import Inception
    client = Inception()

    if args.mode == "probe":
        run_probe(client, messages, args.max_tokens, results_path)
        return

    if args.mode == "compare":
        rows = []
        for effort in ["low", "high"]:
          for trial in range(args.repeat):
            label = f"{effort.upper()}" + (f" (trial {trial + 1}/{args.repeat})"
                                           if args.repeat > 1 else "")
            print(f"\n=== {label} ===")
            result = run_nonstream(client, messages, effort, args.max_tokens)
            result["trial"] = trial
            rows.append(result)
            print(result["text"] or "(no content returned)")
            tps = result["effective_output_tokens_per_s"]
            tps_str = f"{tps:.1f}" if tps is not None else "n/a"
            print(
                f"\nLatency: {result['elapsed_s']:.3f}s | "
                f"Output tokens: {result['completion_tokens']} | "
                f"Effective output tok/s: {tps_str} | "
                f"finish_reason: {result['finish_reason']}"
            )
            print(f"Usage: {json.dumps(result['usage'])}")

        if args.repeat > 1:
            print("\n=== MEDIAN ACROSS TRIALS ===")
            for effort in ["low", "high"]:
                lat = sorted(r["elapsed_s"] for r in rows if r["effort"] == effort)
                tok = sorted(r["completion_tokens"] for r in rows
                             if r["effort"] == effort)
                mid = len(lat) // 2
                print(f"{effort:5} median latency {lat[mid]:.3f}s "
                      f"(range {lat[0]:.3f}-{lat[-1]:.3f}) | "
                      f"median tokens {tok[mid]} (range {tok[0]}-{tok[-1]})")

        Path(results_path).write_text(json.dumps(rows, indent=2))
        print(f"\nSaved to {results_path}")
        return

    trials = []
    for trial in range(args.repeat):
        if args.repeat > 1:
            print(f"\n=== {args.mode.upper()} trial {trial + 1}/{args.repeat} ===")

        if args.mode == "standard":
            result = run_standard_stream(client, messages, args.effort,
                                         args.max_tokens)
        else:
            result = run_diffusion_stream(client, messages, args.effort,
                                          args.max_tokens, live=not args.no_live)
        result["trial"] = trial
        trials.append(result)

        print("\nMetrics:")
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ("final_text", "states")}, indent=2))

    if args.mode == "diffusion" and args.save_states:
        # Keep the richest trajectory: it makes the best animation.
        best = max(trials, key=lambda r: len(r["states"]))
        Path(args.save_states).write_text(json.dumps(best["states"], indent=2))
        print(f"\nSaved {len(best['states'])} states from trial "
              f"{best['trial'] + 1} to {args.save_states}")

    if args.repeat > 1:
        print(f"\n=== MEDIAN ACROSS {args.repeat} TRIALS ===")
        keys = ["time_to_first_output_s", "total_s"]
        keys += ["num_chunks"] if args.mode == "standard" else ["num_refinement_states"]
        for key in keys:
            xs = sorted(r[key] for r in trials if r.get(key) is not None)
            if not xs:
                continue
            mid = xs[len(xs) // 2]
            fmt = "{:.3f}" if isinstance(mid, float) else "{}"
            print(f"{key:26} median " + fmt.format(mid) +
                  "  range " + fmt.format(xs[0]) + "-" + fmt.format(xs[-1]))

    payload = [{k: v for k, v in r.items() if k != "states"} for r in trials]
    Path(results_path).write_text(
        json.dumps(payload if args.repeat > 1 else payload[0], indent=2))
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    main()
