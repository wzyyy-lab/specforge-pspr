#!/usr/bin/env bash
# source this before any managed_local run that needs SGLang spec capture.
# Mirrors the env used by the 20260905/20260906 stage-1 runs (see outputs/SLOTDEEP_*_PROVENANCE/COMMAND.txt).
export PYTHONPATH=/home/wangzhuoyu/sglang-patched-0.5.15${PYTHONPATH:+:$PYTHONPATH}
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1
export MOONCAKE_LOCAL_HOSTNAME=${MOONCAKE_LOCAL_HOSTNAME:-127.0.0.1}
export PATH=/home/wangzhuoyu/.local/bin${PATH:+:$PATH}

if [ "${SGLANG_PATCHED_SELFCHECK:-1}" = "1" ]; then
  python3 - <<'PY'
import importlib.util as u, shutil, sys
ok = True
s = u.find_spec("sglang.srt.spec_capture_sink")
print(f"[env] spec_capture_sink: {s.origin if s else 'MISSING'}")
ok &= s is not None
m = u.find_spec("mooncake.store")
print(f"[env] mooncake.store   : {'OK' if m else 'MISSING'}")
ok &= m is not None
mm = shutil.which("mooncake_master")
print(f"[env] mooncake_master  : {mm or 'MISSING'}")
ok &= mm is not None
import sglang
print(f"[env] sglang           : {sglang.__version__}  {sglang.__file__}")
sys.exit(0 if ok else 1)
PY
  if [ $? -ne 0 ]; then
    echo "[env] FAILED: managed_local prerequisites not satisfied" >&2
    return 1 2>/dev/null || exit 1
  fi
  echo "[env] managed_local prerequisites OK"
fi
