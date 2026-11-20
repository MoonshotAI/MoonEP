#pragma once

#include <acl/acl.h>
#include "shmem.h"
#include <sstream>
#include <stdexcept>
#include <string>

inline void CheckAclStatus(aclError status, const char* expression, const char* file, int line) {
    if (status == ACL_SUCCESS) {
        return;
    }
    std::ostringstream oss;
    oss << "ACL call failed: " << expression << ", status=" << status
        << ", location=" << file << ':' << line;
    throw std::runtime_error(oss.str());
}

inline void CheckShmemStatus(int status, const char* expression, const char* file, int line) {
    if (status == ACLSHMEM_SUCCESS) {
        return;
    }
    std::ostringstream oss;
    oss << "CANN SHMEM call failed: " << expression << ", status=" << status
        << ", location=" << file << ':' << line;
    throw std::runtime_error(oss.str());
}

#define ASCEND_MOONEP_CHECK_ACL(expr) CheckAclStatus((expr), #expr, __FILE__, __LINE__)
#define ASCEND_MOONEP_CHECK_SHMEM(expr) CheckShmemStatus((expr), #expr, __FILE__, __LINE__)
