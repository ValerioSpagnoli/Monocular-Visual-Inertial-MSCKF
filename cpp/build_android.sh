#!/usr/bin/env bash
# Cross-compile the MSCKF core (static library, no Python bindings or tests) for Android with the NDK.
#
#   pixi run build-android                 # arm64-v8a, API level 24
#   ABI=x86_64 pixi run build-android      # e.g. for the emulator
#
# The NDK is taken from $ANDROID_NDK_HOME, or the newest one in ~/Library/Android/sdk/ndk (Android Studio:
# Settings > Languages & Frameworks > Android SDK > SDK Tools > NDK (Side by side)).
# Eigen (header-only) comes from the pixi environment.
set -euo pipefail

ABI="${ABI:-arm64-v8a}"
API="${API:-24}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
    NDK_DIR="${ANDROID_HOME:-$HOME/Library/Android/sdk}/ndk"
    if [[ -d "$NDK_DIR" ]] && [[ -n "$(ls -A "$NDK_DIR")" ]]; then
        ANDROID_NDK_HOME="$NDK_DIR/$(ls "$NDK_DIR" | sort -V | tail -1)"
    else
        echo "Android NDK not found: install it from Android Studio (SDK Tools > NDK) or set ANDROID_NDK_HOME" >&2
        exit 1
    fi
fi
echo "NDK: $ANDROID_NDK_HOME  ABI: $ABI  API: $API"

cmake -S "$ROOT/cpp" -B "$ROOT/build/android-$ABI" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$ABI" -DANDROID_PLATFORM="android-$API" -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release \
    -DMSCKF_BUILD_PYTHON=OFF -DMSCKF_BUILD_TESTS=OFF \
    -DEigen3_DIR="$CONDA_PREFIX/share/eigen3/cmake" \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH
cmake --build "$ROOT/build/android-$ABI"
echo "Built $ROOT/build/android-$ABI/libmsckf_core.a"
