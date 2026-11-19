#pragma once
#include <acl/acl.h>

namespace ascend_moonep {
// Process-wide on-stream policy, independent of the payload transport.
void BarrierAllOnStream(aclrtStream stream);
}
