---
name: Bug report
about: The kit is not behaving as expected, produces an error, or numbers do not match the README.
title: ""
labels: "bug"
assignees: ""
---

<!-- Thank you for using this model kit!

     If you are looking for support, please check the README first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     If you have found a bug, then fill out the template below.
-->

---

## Environment

<!-- Fill in what applies to your setup. The README's tables list every knob and its default. -->

- Hardware / nodes: <!-- e.g. 2x DGX Spark (GB10 / SM121, 121.69 GiB unified memory each) -->
- Interconnect: <!-- e.g. ConnectX-7 RoCE (NCCL_IB_GID_INDEX=3), 10GbE, ... -->
- Image: <!-- `docker images | grep dsv41` — ./start.sh builds `dsv41-flash-exl3:local` on vllm/vllm-openai:deepseekv41-flash-0909 -->
- Checkpoint: <!-- EXL3 2.9 bpw / mul1 in ./model (39 shards) + the original DeepSeek-V4.1-Flash tree for the Engram shards 47+48; `./download.sh` prints whether both look complete -->
- Served name (`SERVED_MODEL_NAME`): <!-- e.g. DeepSeek-v4.1-Flash-EXL3 -->
- How you started the serve: <!-- e.g. ./start.sh, ./start.sh restart, SKIP_SYNC=1 ./start.sh, custom compose -->
- Relevant settings: <!-- e.g. MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS, LONG_PREFILL_TOKEN_THRESHOLD, KV_CACHE_MEMORY_BYTES, KV_BLOCK_SIZE, SPEC_METHOD, DSPARK_TOKENS, EXL3_FUSED_MOE, EXL3_FAT_GROUPED, EXL3_TEMP_ROWS_FUSED, WEIGHT_SYNC, LANGUAGE_MODEL_ONLY, PYTORCH_CUDA_ALLOC_CONF -->

---

## Steps to Reproduce

<!-- Full steps so that we can reproduce the problem. -->

1. <!-- e.g. `./start.sh` — what it printed up to the failure -->
2. ... <!-- the request or action that shows the bug -->
3. ... <!-- e.g. "curl /v1/models returns a different served name than configured" -->

**Expected results:** <!-- what did you expect to happen? -->

**Actual results:** <!-- what did you actually see happen? -->

---

### Additional context

Add anything else: a minimal failing request, JSON responses, `docker inspect` output, and so on.

<details>
<summary>Minimal reproduction sample</summary>

<!--
      If the bug is about model output or API behavior, attach a minimal reproducible
      request below between the lines with the backticks.
-->

```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-v4.1-Flash-EXL3",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 256
  }'
```

</details>

<details>
  <summary>Logs</summary>

<!--
      Paste the log output below between the lines with the backticks, and mention
      whether it came from start.sh (logs/start-*.log), the container logs on either
      node (`docker logs dsv41-exl3-head`), a memguard log, or a client.

      Host RAM *is* GPU memory on a Spark, so most failures here are memory failures.
      Common culprits worth checking before filing:
        * Boot refused by preflight -> a node is short of MemAvailable; stop every
          other GPU workload (the native 3-Spark serve, an old worker) first.
        * `memguard ... KILLING` in logs/memguard-head.log (or the worker's) -> the
          node ran out during a long prefill, not at boot. Read the low-water line
          after a long prompt, not just after a boot; each +0.5 GiB of
          KV_CACHE_MEMORY_BYTES costs ~1 GiB of head prefill margin.
        * Long prompts far slower than the README table -> check
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; with False the caching
          allocator refragments on every chunk (a 455k prompt went 9.5 min -> 31 min).
        * Boot hangs with both GPUs busy -> two EXL3 kernels sharing exllamav3's
          per-device lock buffer. Keep DSV41_EXL3_SERIAL_STREAMS=1 and
          VLLM_DISABLE_SHARED_EXPERTS_STREAM=1; logs/hang-*-pyspy.txt has the stacks.
        * Fluent garbage / `smoke FAIL` -> bisect with DSV41_ENGRAM_DISABLE=1, then
          EXL3_FUSED_MOE=0 + ENFORCE_EAGER=1, then SPEC_METHOD=none.
        * `EXL3 load shape mismatch` -> a TP=2 packed-linear or K-map problem; attach
          the tensor name from the traceback.
        * Image requests rejected (`At most 0 image(s)`) -> LANGUAGE_MODEL_ONLY=1
          in `.env`. New checkouts default to 0. Set LANGUAGE_MODEL_ONLY=0 and
          MAX_NUM_BATCHED_TOKENS>=1536, then restart. The SM12x 128-wide window
          clamp is what makes vision runnable, not a reason to disable it.
-->

```

```

</details>

<!--
      Consider also attaching screenshots and/or videos to better illustrate the issue.

      You can upload them directly on GitHub.
      Beware that video file size is limited to 10MB.
-->
