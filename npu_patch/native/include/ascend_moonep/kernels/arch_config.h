#pragma once

// Architecture-dependent planning-kernel parameters, centralized here so the
// host/device code selects the right values purely from the ASCEND_MOONEP_ARCH_*
// macros injected by the build (see native/CMakeLists.txt) instead of open-coding
// per-arch constants at each call site.
//
//   A2 / A3 : Ascend910B / Ascend910_93, NPU arch 2201, non-SIMT
//   A5      : Ascend950,           NPU arch 3510, SIMT

#include <cstdint>

namespace ascend_moonep {

#if defined(ASCEND_MOONEP_ARCH_A5)
// A5 (3510 / SIMT)
inline constexpr uint32_t kPlanningUbSizeBytes = 216 * 1024;
#else
// A2 / A3 (2201 / non-SIMT)
inline constexpr uint32_t kPlanningUbSizeBytes = 192 * 1024;
#endif

} // namespace ascend_moonep
