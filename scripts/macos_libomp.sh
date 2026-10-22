#!/usr/bin/env bash
# macOS only: LightGBM needs libomp. If Homebrew's libomp is not installed
# (`brew install libomp` is the normal fix), point LightGBM at the copy bundled in scikit-learn.
set -euo pipefail
SP=$(uv run python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
LIB="$SP/lightgbm/lib/lib_lightgbm.dylib"
[ -f "$LIB" ] || exit 0
if otool -l "$LIB" | grep -q "sklearn/.dylibs"; then exit 0; fi
install_name_tool -add_rpath "@loader_path/../../sklearn/.dylibs" "$LIB"
codesign --force --sign - "$LIB" >/dev/null 2>&1 || true
echo "patched $LIB"
