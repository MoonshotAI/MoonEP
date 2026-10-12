#pragma once

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <torch/custom_class.h>

class ShmemInstance : public torch::jit::CustomClassHolder {
public:
    ShmemInstance(
        int64_t rank,
        int64_t world_size,
        std::string ip_port);

    ~ShmemInstance();

    ShmemInstance(const ShmemInstance&) = delete;
    ShmemInstance& operator=(const ShmemInstance&) = delete;

    void Init();
    void Init(int64_t local_mem_size);
    void Barrier();
    void WaitHandle(aclrtStream stream) const;
    void Synchronize(aclrtStream stream) const;
    void Destroy();

    uint64_t FftsAddr() const;
    bool Initialized() const;
    bool IsAvailable() const;
    int64_t Rank() const;
    int64_t WorldSize() const;

private:
    void InitRuntime();
    void DestroyNoThrow() noexcept;

    int rank_;
    int world_size_;
    uint64_t local_mem_size_;
    std::string ip_port_;
    bool initialized_;
    bool acquired_global_runtime_;

    uint64_t ffts_addr_;
};
