// moonEP_combine — AscendC kernel (MoonEP ABI).
//
// Args order: buffers → tiling (same as moonep_dispatch).
//   hidden_buf              GM_ADDR  symmetric local shard start; view [N, H] bf16, N=S*K
//                           remote read via aclshmemx_mte_get_nbi / local DataCopy
//   weight_buf              GM_ADDR  route weights [N] fp32 (placeholder if !with_weight)
//   dst                     GM_ADDR  [N] / [S,K] int32 routing table
//   dup_groups              GM_ADDR  [N,3] int32 (primary_loff, dup_start, dup_n)
//   dup_loffs               GM_ADDR  [N] int32 flat dup local offsets
//   dup_counts              GM_ADDR  [2] int32 {n_groups, n_dup_loffs}
//   hidden_sk_out           GM_ADDR  [S, H] bf16  K-sum (fp32 ACC → bf16)
//   route_weights_sk_out    GM_ADDR  [S, AlignUp(K,8)] fp32 (placeholder if !with_weight)
//   tiling                  MoonEPCombineTilingData (by value)
//
// Parallelism: tokens partitioned across AIVs. Full hidden row processed in one pass
// (no H-chunk); UB holds ping/pong bf16 rows + fp32 tmp/acc.
//
// Sync: prologue → aclshmemx_barrier_all_vec → combine
// Hot paths use MTE2↔V HardEvents + mte_get_nbi ping-pong (grad_reduce style).
// Weight (EVENT_ID2/3): same topK walk as dispatch ScatterPhase — primary issues
// hidden+weight together; dup (load_mask=0) issues weight-only in the same pass
// so small gets hide under neighboring primary RMA / Cast·Add.
// Prologue/combine out-stores use dual store rows (dispatch epilogue style):
// MTE3 is not drained per group/token; next accumulate overlaps prior store.

#include <type_traits>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "shmem.h"

#include "ascend_moonep/kernels/launch.h"
#include "ascend_moonep/kernels/types.h"

using namespace AscendC;
using half_t = __fp16;
using bf16_t = bfloat16_t;
using ascend_moonep::kMoonEPAivNum;
using ascend_moonep::MoonEPCombineTilingData;

namespace {

constexpr int32_t kAlignFp32 = 8;
constexpr int32_t kAlignInt32 = 8;
constexpr int32_t kAlignBf16 = 16;  // 32B / sizeof(bf16)
constexpr int32_t kMaxK = 16;
constexpr uint32_t kUbBytes = 192 * 1024;
constexpr int32_t kCombineDupGroupFields = 3;
// Prologue meta staging (fits after load/store H rows + fp32 acc/tmp for H<=7168).
// Both group meta and loffs use 1024-wide windows: typical ~800/AIV at n≈24k is
// one shot each; larger shards only advance windows (monotonic group/loff order).
constexpr int32_t kCombinePrologueGroupsStageMax = 1024;
constexpr int32_t kCombinePrologueLoffStageMax = 1024;
// primary + dups in one group ≤ K (topk); keep stack list off the loff window size.
constexpr int32_t kCombinePrologueMaxSrcsPerGroup = 1 + kMaxK;

__aicore__ inline int32_t AlignUpI(int32_t n, int32_t a) {
    return ((n + a - 1) / a) * a;
}

__aicore__ inline void SplitRange(
    int32_t total, int32_t parts, int32_t idx, int32_t& begin, int32_t& end)
{
    int32_t per = (total + parts - 1) / parts;
    begin = idx * per;
    end = begin + per;
    if (begin > total) begin = total;
    if (end > total) end = total;
}

template <typename T>
__aicore__ inline __ubuf__ T* UbPtr(LocalTensor<T> ub)
{
    return reinterpret_cast<__ubuf__ T*>(
        static_cast<uintptr_t>(ub.address_.bufferAddr));
}

template <typename T>
class MoonEPCombine {
    static_assert(std::is_same_v<T, bf16_t>, "moonEP_combine payload is bf16");

public:
    __aicore__ inline MoonEPCombine() {}

    __aicore__ inline void Init(
        GM_ADDR hidden_buf,
        GM_ADDR weight_buf,
        GM_ADDR dst,
        GM_ADDR dup_groups,
        GM_ADDR dup_loffs,
        GM_ADDR dup_counts,
        GM_ADDR hidden_sk_out,
        GM_ADDR route_weights_sk_out,
        const MoonEPCombineTilingData& tiling,
        int32_t blk_idx,
        int32_t blk_cnt)
    {
        blk_idx_ = blk_idx;
        blk_cnt_ = blk_cnt;
        my_rank_ = static_cast<int32_t>(tiling.rank);
        world_size_ = static_cast<int32_t>(tiling.world_size);
        S_ = static_cast<int32_t>(tiling.num_tokens);
        H_ = static_cast<int32_t>(tiling.hidden);
        K_ = static_cast<int32_t>(tiling.num_topk);
        with_weight_ = static_cast<int32_t>(tiling.with_weight);
        N_ = S_ * K_;

        SplitRange(S_, blk_cnt_, blk_idx_, s_beg_, s_end_);
        n_tokens_ = (s_end_ > s_beg_) ? (s_end_ - s_beg_) : 0;

        hiddenBase_ = reinterpret_cast<__gm__ uint8_t*>(hidden_buf);
        weightBase_ = reinterpret_cast<__gm__ uint8_t*>(weight_buf);
        weight_byte_off_ = static_cast<int64_t>(tiling.reserved);
        weight_packed_ = (with_weight_ && weight_byte_off_ > 0) ? 1 : 0;

        gmHiddenLocal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hidden_buf),
            static_cast<int64_t>(N_) * H_);
        gmDst_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dst),
            static_cast<int64_t>(S_) * K_);
        gmOut_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hidden_sk_out),
            static_cast<int64_t>(S_) * H_);

        gmDupGroups_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dup_groups),
            static_cast<int64_t>(N_) * 3);
        gmDupLoffs_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dup_loffs),
            static_cast<int64_t>(N_));
        gmDupCounts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dup_counts), 8);

        sk_stride_ = AlignUpI(K_ > 0 ? K_ : 1, kAlignInt32);
        if (with_weight_) {
            gmWeightLocal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight_buf),
                static_cast<int64_t>(N_));
            gmOutSk_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(route_weights_sk_out),
                static_cast<int64_t>(S_) * sk_stride_);
            has_weight_ptr_ = 1;
        } else {
            has_weight_ptr_ = 0;
        }

        dst_copy_n_ = AlignUpI(K_ > 0 ? K_ : 1, kAlignInt32);
        h_copy_n_ = AlignUpI(H_, kAlignBf16);
        h_fp32_n_ = AlignUpI(H_, kAlignFp32);
        row_bytes_ = static_cast<uint32_t>(h_copy_n_) * sizeof(T);

        // UB: load ping/pong + store ping/pong + fp32 tmp/acc + meta/weight.
        // H=7168 → 4*14KiB + 2*28KiB ≈ 114KiB (+ staging), fits 192KiB.
        offRow0_ = 0;
        offRow1_ = offRow0_ + row_bytes_;
        offStore0_ = offRow1_ + row_bytes_;
        offStore1_ = offStore0_ + row_bytes_;
        offTmp_ = offStore1_ + row_bytes_;
        offAcc_ = offTmp_ + static_cast<uint32_t>(h_fp32_n_) * sizeof(float);
        uint32_t scratch = offAcc_ + static_cast<uint32_t>(h_fp32_n_) * sizeof(float);
        offGroup_ = scratch;
        offLoff_ = offGroup_ + static_cast<uint32_t>(
            kCombinePrologueGroupsStageMax * kCombineDupGroupFields * sizeof(int32_t));
        offDst_ = offLoff_ + static_cast<uint32_t>(
            kCombinePrologueLoffStageMax * sizeof(int32_t));
        // Two 8×fp32 windows for weight ping-pong (EVENT_ID2 / EVENT_ID3).
        offWLoad0_ = offDst_ + 64;
        offWLoad1_ = offWLoad0_ + static_cast<uint32_t>(kAlignInt32) * sizeof(float);
        offSk_ = offWLoad1_ + static_cast<uint32_t>(kAlignInt32) * sizeof(float);
        offMeta_ = offSk_ + 64;

        allocUB();

        // Same habit as dispatch epilogue: scalar-read dup_counts from GM.
        n_groups_ = gmDupCounts_.GetValue(0);
        n_dup_loffs_ = gmDupCounts_.GetValue(1);
        if (n_groups_ < 0) n_groups_ = 0;
        if (n_groups_ > N_) n_groups_ = N_;
        if (n_dup_loffs_ < 0) n_dup_loffs_ = 0;
        if (n_dup_loffs_ > N_) n_dup_loffs_ = N_;
    }

    __aicore__ inline void Process()
    {
        if (n_groups_ > 0) {
            ProcessPrologue();
        }
        PipeBarrier<PIPE_ALL>();
        aclshmemx_barrier_all_vec();

        if (n_tokens_ > 0) {
            ProcessCombine();
        }
    }

private:
    __aicore__ inline void allocUB()
    {
        TPipe pipe;
        TBuf<TPosition::VECCALC> tBuf;
        pipe.InitBuffer(tBuf, kUbBytes);
        ubBuf_ = tBuf.Get<uint8_t>();
        pipe.Destroy();
    }

    template <class Element>
    __aicore__ inline LocalTensor<Element> UB(uint32_t off)
    {
        return ubBuf_[off].template ReinterpretCast<Element>();
    }

    __aicore__ inline uint32_t RowOff(int32_t buf) const
    {
        return buf == 0 ? offRow0_ : offRow1_;
    }

    // Issue GM/symmetric → UB hidden row; signals MTE2_V(event).
    __aicore__ inline void IssueHiddenLoad(
        LocalTensor<T> row,
        int32_t pe,
        int32_t loff,
        TEventID event_id)
    {
        __gm__ T* sym =
            reinterpret_cast<__gm__ T*>(hiddenBase_) +
            static_cast<int64_t>(loff) * H_;
        if (pe == my_rank_) {
            GlobalTensor<T> src;
            src.SetGlobalBuffer(sym, h_copy_n_);
            DataCopy(row, src, h_copy_n_);
        } else {
            aclshmemx_mte_get_nbi(
                UbPtr(row), sym, static_cast<uint32_t>(h_copy_n_), pe, event_id);
        }
        SetFlag<HardEvent::MTE2_V>(event_id);
    }

    // Weight MTE2 helpers — EVENT_ID2/3 only (AccumulateRow owns ID0/1).
    __aicore__ inline void IssueWeightLoad(
        LocalTensor<float> ubW, int32_t pe, int32_t loff, TEventID event_id)
    {
        int32_t base = (loff / kAlignInt32) * kAlignInt32;
        if (pe == my_rank_) {
            GlobalTensor<float> gm;
            gm.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(weightBase_) + base, kAlignInt32);
            DataCopy(ubW, gm, kAlignInt32);
        } else if (weight_packed_) {
            __gm__ float* sym = reinterpret_cast<__gm__ float*>(
                reinterpret_cast<__gm__ uint8_t*>(hiddenBase_) + weight_byte_off_) + base;
            aclshmemx_mte_get_nbi(
                UbPtr(ubW), sym, static_cast<uint32_t>(kAlignInt32), pe, event_id);
        } else {
            __gm__ float* sym = reinterpret_cast<__gm__ float*>(weightBase_) + base;
            aclshmemx_mte_get_nbi(
                UbPtr(ubW), sym, static_cast<uint32_t>(kAlignInt32), pe, event_id);
        }
        SetFlag<HardEvent::MTE2_S>(event_id);
    }

    __aicore__ inline float TakeWeightValue(
        LocalTensor<float> ubW, int32_t loff, TEventID event_id)
    {
        WaitFlag<HardEvent::MTE2_S>(event_id);
        return ubW.GetValue(loff - (loff / kAlignInt32) * kAlignInt32);
    }

    __aicore__ inline bool WeightRouteValid(int32_t pe, int32_t loff) const
    {
        return pe >= 0 && pe < world_size_ && loff >= 0 && loff < N_;
    }

    // Plan contract: groups are emitted in order with contiguous, non-overlapping
    // dup_loffs slices (dup_start monotonically increasing). An AIV's contiguous
    // group range therefore maps to one contiguous loff span = [first.dup_start,
    // last.dup_start + last.dup_n).
    __aicore__ inline bool PeekAivLoffSpan(
        int32_t group_beg,
        int32_t group_end,
        LocalTensor<int32_t> groupStage,
        int32_t& loff_span_start,
        int32_t& loff_span_end)
    {
        DataCopy(
            groupStage,
            gmDupGroups_[static_cast<int64_t>(group_beg) * kCombineDupGroupFields],
            kAlignInt32);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        int32_t start0 = groupStage.GetValue(1);
        int32_t n0 = groupStage.GetValue(2);

        int32_t start1 = start0;
        int32_t n1 = n0;
        if (group_end - group_beg > 1) {
            DataCopy(
                groupStage,
                gmDupGroups_[static_cast<int64_t>(group_end - 1) *
                             kCombineDupGroupFields],
                kAlignInt32);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            start1 = groupStage.GetValue(1);
            n1 = groupStage.GetValue(2);
        }

        if (start0 < 0 || n0 <= 0 || start0 >= n_dup_loffs_) {
            return false;
        }
        if (start1 < 0 || n1 <= 0 || start1 >= n_dup_loffs_) {
            return false;
        }
        int32_t avail0 = n_dup_loffs_ - start0;
        if (n0 > avail0) n0 = avail0;
        int32_t avail1 = n_dup_loffs_ - start1;
        if (n1 > avail1) n1 = avail1;
        loff_span_start = start0;
        loff_span_end = start1 + n1;
        return loff_span_end > loff_span_start;
    }

    __aicore__ inline void LoadLoffWindow(
        LocalTensor<int32_t> loffStage, int32_t win_start, int32_t win_end)
    {
        int32_t n = win_end - win_start;
        if (n <= 0) {
            return;
        }
        DataCopy(
            loffStage,
            gmDupLoffs_[win_start],
            AlignUpI(n, kAlignInt32));
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    // Advance the staged loff window so [dup_start, dup_start+dup_n) is covered.
    // Groups walk dup_start monotonically → windows only move forward.
    __aicore__ inline bool EnsureLoffWindow(
        int32_t dup_start,
        int32_t dup_n,
        int32_t aiv_loff_end,
        LocalTensor<int32_t> loffStage,
        int32_t& win_start,
        int32_t& win_end)
    {
        const int32_t dup_end = dup_start + dup_n;
        if (dup_n > kCombinePrologueLoffStageMax) {
            return false;
        }
        if (dup_start >= win_start && dup_end <= win_end) {
            return true;
        }
        win_start = dup_start;
        win_end = win_start + kCombinePrologueLoffStageMax;
        if (win_end > aiv_loff_end) {
            win_end = aiv_loff_end;
        }
        if (dup_end > win_end) {
            return false;
        }
        LoadLoffWindow(loffStage, win_start, win_end);
        return true;
    }

    // ── Prologue: AIV group range + windowed meta/loffs + store-pipelined reduce ──
    __aicore__ inline void ProcessPrologue()
    {
        int32_t group_beg = 0;
        int32_t group_end = 0;
        SplitRange(n_groups_, blk_cnt_, blk_idx_, group_beg, group_end);
        if (group_beg >= group_end) {
            return;
        }

        LocalTensor<int32_t> groupStage = UB<int32_t>(offGroup_);
        LocalTensor<int32_t> loffStage = UB<int32_t>(offLoff_);

        // Dual store-stage credits (dispatch PlanEpiloguePhase style).
        // Prologue owns ID2/ID3 for MTE3_MTE2; drained before the barrier.
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        prologue_store_stage_ = 0;

        int32_t aiv_loff_start = 0;
        int32_t aiv_loff_end = 0;
        const bool have_aiv_span = PeekAivLoffSpan(
            group_beg, group_end, groupStage, aiv_loff_start, aiv_loff_end);
        // Empty loff window → first EnsureLoffWindow loads; ~800 ⇒ one shot.
        int32_t loff_win_start = 0;
        int32_t loff_win_end = 0;

        // groupStage is only a UB capacity cap (same idea as loff windows), not a
        // logical "segment" tied to loff span discovery.
        for (int32_t g0 = group_beg; g0 < group_end;) {
            int32_t g_count = group_end - g0;
            if (g_count > kCombinePrologueGroupsStageMax) {
                g_count = kCombinePrologueGroupsStageMax;
            }
            int32_t group_elems = g_count * kCombineDupGroupFields;
            DataCopy(
                groupStage,
                gmDupGroups_[static_cast<int64_t>(g0) * kCombineDupGroupFields],
                AlignUpI(group_elems, kAlignInt32));
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

            for (int32_t li = 0; li < g_count; ++li) {
                int32_t base = li * kCombineDupGroupFields;
                int32_t primary_loff = groupStage.GetValue(base);
                int32_t dup_start = groupStage.GetValue(base + 1);
                int32_t dup_n = groupStage.GetValue(base + 2);
                if (dup_n <= 0 || primary_loff < 0 || primary_loff >= N_) {
                    continue;
                }
                if (dup_start < 0 || dup_start >= n_dup_loffs_) {
                    continue;
                }
                int32_t avail = n_dup_loffs_ - dup_start;
                if (dup_n > avail) dup_n = avail;

                bool stage_loffs = false;
                int32_t loff_span_start = 0;
                if (have_aiv_span) {
                    stage_loffs = EnsureLoffWindow(
                        dup_start,
                        dup_n,
                        aiv_loff_end,
                        loffStage,
                        loff_win_start,
                        loff_win_end);
                    loff_span_start = loff_win_start;
                }
                PrologueAccumulateGroup(
                    primary_loff,
                    dup_start,
                    dup_n,
                    stage_loffs,
                    loffStage,
                    loff_span_start);
            }
            g0 += g_count;
        }

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    }

    __aicore__ inline int32_t DupLoffAt(
        int32_t d_idx,
        bool stage_loffs,
        LocalTensor<int32_t> loffStage,
        int32_t loff_span_start)
    {
        if (stage_loffs) {
            return loffStage.GetValue(d_idx - loff_span_start);
        }
        LocalTensor<int32_t> ubLoff = UB<int32_t>(offMeta_);
        int32_t loff_base = (d_idx / kAlignInt32) * kAlignInt32;
        DataCopy(ubLoff, gmDupLoffs_[loff_base], kAlignInt32);
        // ID0: keep ID2/ID3 free for prologue store MTE3_MTE2 credits.
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        return ubLoff.GetValue(d_idx - loff_base);
    }

    __aicore__ inline void PrologueAccumulateGroup(
        int32_t primary_loff,
        int32_t dup_start,
        int32_t dup_count,
        bool stage_loffs,
        LocalTensor<int32_t> loffStage,
        int32_t loff_span_start)
    {
        LocalTensor<float> acc = UB<float>(offAcc_);
        LocalTensor<float> tmpF = UB<float>(offTmp_);
        LocalTensor<T> row0 = UB<T>(offRow0_);
        LocalTensor<T> row1 = UB<T>(offRow1_);
        int64_t prim_off = static_cast<int64_t>(primary_loff) * H_;

        // primary + valid dups in one list; ping-pong same as AccumulateRow
        // (avoid unconditional V_MTE2 waits that deadlock when a flag was
        // already consumed or never produced).
        int32_t src_loff[kCombinePrologueMaxSrcsPerGroup];
        src_loff[0] = primary_loff;
        int32_t n_src = 1;
        for (int32_t d = 0;
             d < dup_count && n_src < kCombinePrologueMaxSrcsPerGroup;
             ++d) {
            int32_t dup_loff = DupLoffAt(
                dup_start + d, stage_loffs, loffStage, loff_span_start);
            if (dup_loff < 0 || dup_loff >= N_) continue;
            src_loff[n_src++] = dup_loff;
        }

        IssueHiddenLoad(row0, my_rank_, src_loff[0], EVENT_ID0);

        bool ping_pending = false;
        bool pong_pending = false;
        int32_t cur = 0;
        for (int32_t i = 0; i < n_src; ++i) {
            TEventID cur_ev = cur == 0 ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<T> cur_row = cur == 0 ? row0 : row1;
            WaitFlag<HardEvent::MTE2_V>(cur_ev);

            int32_t next_i = i + 1;
            if (next_i < n_src) {
                int32_t next_buf = 1 - cur;
                TEventID next_ev = next_buf == 0 ? EVENT_ID0 : EVENT_ID1;
                LocalTensor<T> next_row = next_buf == 0 ? row0 : row1;
                if (next_buf == 0 && ping_pending) {
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    ping_pending = false;
                } else if (next_buf == 1 && pong_pending) {
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                    pong_pending = false;
                }
                IssueHiddenLoad(next_row, my_rank_, src_loff[next_i], next_ev);
            }

            if (i == 0) {
                Cast(acc, cur_row, RoundMode::CAST_NONE, H_);
            } else {
                Cast(tmpF, cur_row, RoundMode::CAST_NONE, H_);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, tmpF, h_fp32_n_);
            }
            SetFlag<HardEvent::V_MTE2>(cur_ev);
            if (cur == 0) {
                ping_pending = true;
            } else {
                pong_pending = true;
            }
            cur = 1 - cur;
        }

        if (ping_pending) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        if (pong_pending) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        }

        // Store via independent stage — do not drain; next group can accumulate.
        TEventID store_ev =
            prologue_store_stage_ == 0 ? EVENT_ID2 : EVENT_ID3;
        LocalTensor<T> store_row = prologue_store_stage_ == 0
            ? UB<T>(offStore0_)
            : UB<T>(offStore1_);
        WaitFlag<HardEvent::MTE3_MTE2>(store_ev);
        Cast(store_row, acc, RoundMode::CAST_RINT, H_);
        SetFlag<HardEvent::V_MTE3>(store_ev);
        WaitFlag<HardEvent::V_MTE3>(store_ev);
        DataCopy(gmHiddenLocal_[prim_off], store_row, h_copy_n_);
        SetFlag<HardEvent::MTE3_MTE2>(store_ev);
        prologue_store_stage_ ^= 1;
    }

    // ── Combine: topK walk + cross-token out-store pipeline ──
    __aicore__ inline void ProcessCombine()
    {
        // Out-store credits on ID0/ID1 (MTE3_MTE2); physical store* rows.
        // No fake sk credit on ID2 — weight get also uses ID2 (MTE2_S); only
        // Wait MTE3_MTE2(ID2) after a real sk DataCopy SetFlag.
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        combine_store_stage_ = 0;
        sk_mte3_pending_ = 0;

        for (int32_t s = s_beg_; s < s_end_; ++s) {
            ProcessOneToken(s);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        if (has_weight_ptr_ && sk_mte3_pending_) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
            sk_mte3_pending_ = 0;
        }
    }

    __aicore__ inline void ProcessOneToken(int32_t s)
    {
        int32_t sK_base = s * K_;
        int64_t out_row = static_cast<int64_t>(s) * H_;

        LocalTensor<int32_t> ubDst = UB<int32_t>(offDst_);
        DataCopy(ubDst, gmDst_[sK_base], dst_copy_n_);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        int32_t load_mask[kMaxK];
        int32_t drank[kMaxK];
        int32_t loff[kMaxK];
        int32_t raw_dsts[kMaxK];
        for (int32_t k = 0; k < K_; ++k) {
            int32_t dv = ubDst.GetValue(k);
            if (dv >= 0) {
                load_mask[k] = 1;
                raw_dsts[k] = dv;
            } else {
                load_mask[k] = 0;
                raw_dsts[k] = -dv - 1;
            }
            drank[k] = raw_dsts[k] / N_;
            loff[k] = raw_dsts[k] % N_;
            if (drank[k] < 0 || drank[k] >= world_size_ ||
                loff[k] < 0 || loff[k] >= N_) {
                load_mask[k] = 0;
            }
        }

        LocalTensor<float> ubSk = UB<float>(offSk_);
        // Prior token's sk MTE3 must finish before AccumulateRow rewrites ubSk.
        // First token: sk_mte3_pending_==0 → no Wait (avoids bogus ID2 credit
        // colliding with weight MTE2_S on ID2).
        if (has_weight_ptr_ && sk_mte3_pending_) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
            sk_mte3_pending_ = 0;
        }

        AccumulateRow(load_mask, drank, loff, ubSk, has_weight_ptr_);

        // acc is produced under V_MTE2 waits inside AccumulateRow; safe for V.
        LocalTensor<float> acc = UB<float>(offAcc_);
        TEventID store_ev =
            combine_store_stage_ == 0 ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<T> store_row = combine_store_stage_ == 0
            ? UB<T>(offStore0_)
            : UB<T>(offStore1_);
        // Wait only this stage's prior MTE3 (credit or previous out-store).
        WaitFlag<HardEvent::MTE3_MTE2>(store_ev);
        Cast(store_row, acc, RoundMode::CAST_RINT, H_);
        SetFlag<HardEvent::V_MTE3>(store_ev);
        WaitFlag<HardEvent::V_MTE3>(store_ev);
        DataCopy(gmOut_[out_row], store_row, h_copy_n_);
        SetFlag<HardEvent::MTE3_MTE2>(store_ev);
        combine_store_stage_ ^= 1;

        if (has_weight_ptr_) {
            // ubSk fully filled; S_MTE3 after all SetValue in AccumulateRow.
            SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
            DataCopy(
                gmOutSk_[static_cast<int64_t>(s) * sk_stride_],
                ubSk,
                dst_copy_n_);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
            sk_mte3_pending_ = 1;
        }
    }

    // Take weight buf only if this function previously IssueWeightLoad'd it.
    __aicore__ inline void TakeWeightBufIfPending(
        int32_t (&w_pend_k)[2],
        int32_t wb,
        const int32_t (&loff)[kMaxK],
        LocalTensor<float> w0,
        LocalTensor<float> w1,
        LocalTensor<float> ubSk)
    {
        int32_t pk = w_pend_k[wb];
        if (pk < 0) {
            return;
        }
        TEventID wev = wb == 0 ? EVENT_ID2 : EVENT_ID3;
        LocalTensor<float> wbuf = wb == 0 ? w0 : w1;
        float wv = TakeWeightValue(wbuf, loff[pk], wev);
        ubSk.SetValue(pk, wv);
        w_pend_k[wb] = -1;
    }

    // Issue weight on w_next. If the slot is busy, Take it first (W↔W only;
    // never waits on hidden ID0/1). Prefer calling the free-slot variant on the
    // H→Cast critical path so weight completion cannot stall Cast.
    __aicore__ inline void IssueWeightOnNextBuf(
        int32_t (&w_pend_k)[2],
        int32_t& w_next,
        int32_t k,
        const int32_t (&drank)[kMaxK],
        const int32_t (&loff)[kMaxK],
        LocalTensor<float> w0,
        LocalTensor<float> w1,
        LocalTensor<float> ubSk)
    {
        int32_t wb = w_next;
        TakeWeightBufIfPending(w_pend_k, wb, loff, w0, w1, ubSk);
        LocalTensor<float> wbuf = wb == 0 ? w0 : w1;
        TEventID wev = wb == 0 ? EVENT_ID2 : EVENT_ID3;
        IssueWeightLoad(wbuf, drank[k], loff[k], wev);
        w_pend_k[wb] = k;
        w_next ^= 1;
    }

    // Issue only when w_next is free — no Wait. Returns 1 if issued.
    __aicore__ inline int32_t TryIssueWeightOnFreeBuf(
        int32_t (&w_pend_k)[2],
        int32_t& w_next,
        int32_t k,
        const int32_t (&drank)[kMaxK],
        const int32_t (&loff)[kMaxK],
        LocalTensor<float> w0,
        LocalTensor<float> w1)
    {
        int32_t wb = w_next;
        if (w_pend_k[wb] >= 0) {
            return 0;
        }
        LocalTensor<float> wbuf = wb == 0 ? w0 : w1;
        TEventID wev = wb == 0 ? EVENT_ID2 : EVENT_ID3;
        IssueWeightLoad(wbuf, drank[k], loff[k], wev);
        w_pend_k[wb] = k;
        w_next ^= 1;
        return 1;
    }

    // No-weight: classic compact primary ping-pong (Issue-next before Cast).
    // Must stay bit-equivalent in sync shape to the pre-weight-opt path so
    // --no-weights latency does not regress.
    __aicore__ inline void AccumulateRowNoWeight(
        const int32_t (&load_mask)[kMaxK],
        const int32_t (&drank)[kMaxK],
        const int32_t (&loff)[kMaxK])
    {
        LocalTensor<float> acc = UB<float>(offAcc_);
        LocalTensor<float> tmpF = UB<float>(offTmp_);
        LocalTensor<T> row0 = UB<T>(offRow0_);
        LocalTensor<T> row1 = UB<T>(offRow1_);

        int32_t src_pe[kMaxK];
        int32_t src_loff[kMaxK];
        int32_t n_src = 0;
        for (int32_t k = 0; k < K_; ++k) {
            if (!load_mask[k]) {
                continue;
            }
            src_pe[n_src] = drank[k];
            src_loff[n_src] = loff[k];
            ++n_src;
        }

        if (n_src == 0) {
            Duplicate(acc, 0.0f, h_fp32_n_);
            PipeBarrier<PIPE_V>();
            return;
        }

        IssueHiddenLoad(row0, src_pe[0], src_loff[0], EVENT_ID0);

        bool ping_pending = false;
        bool pong_pending = false;
        int32_t cur = 0;

        for (int32_t i = 0; i < n_src; ++i) {
            TEventID cur_ev = cur == 0 ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<T> cur_row = cur == 0 ? row0 : row1;
            WaitFlag<HardEvent::MTE2_V>(cur_ev);

            int32_t next_i = i + 1;
            if (next_i < n_src) {
                int32_t next_buf = 1 - cur;
                TEventID next_ev = next_buf == 0 ? EVENT_ID0 : EVENT_ID1;
                LocalTensor<T> next_row = next_buf == 0 ? row0 : row1;
                if (next_buf == 0 && ping_pending) {
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    ping_pending = false;
                } else if (next_buf == 1 && pong_pending) {
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                    pong_pending = false;
                }
                IssueHiddenLoad(
                    next_row, src_pe[next_i], src_loff[next_i], next_ev);
            }

            if (i == 0) {
                Cast(acc, cur_row, RoundMode::CAST_NONE, H_);
            } else {
                Cast(tmpF, cur_row, RoundMode::CAST_NONE, H_);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, tmpF, h_fp32_n_);
            }
            SetFlag<HardEvent::V_MTE2>(cur_ev);
            if (cur == 0) {
                ping_pending = true;
            } else {
                pong_pending = true;
            }
            cur = 1 - cur;
        }

        if (ping_pending) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        if (pong_pending) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        }
    }

    // With-weight: dispatch-style topK walk.
    // Hidden ID0/1 and weight ID2/3 are independent — no H↔W HardEvent.
    // On the Wait(H)→Issue(next H)→Cast path only TryIssueWeight (no Take);
    // W↔W Take/Issue happens after SetFlag(V_MTE2) or in the final drain so
    // weight completion cannot stall hidden Cast / buffer recycle.
    __aicore__ inline void AccumulateRowWithWeight(
        const int32_t (&load_mask)[kMaxK],
        const int32_t (&drank)[kMaxK],
        const int32_t (&loff)[kMaxK],
        LocalTensor<float> ubSk)
    {
        LocalTensor<float> acc = UB<float>(offAcc_);
        LocalTensor<float> tmpF = UB<float>(offTmp_);
        LocalTensor<T> row0 = UB<T>(offRow0_);
        LocalTensor<T> row1 = UB<T>(offRow1_);
        LocalTensor<float> w0 = UB<float>(offWLoad0_);
        LocalTensor<float> w1 = UB<float>(offWLoad1_);

        int32_t src_pe[kMaxK];
        int32_t src_loff[kMaxK];
        int32_t src_k[kMaxK];
        int32_t n_src = 0;
        int32_t dup_k[kMaxK];
        int32_t n_dup = 0;

        for (int32_t k = 0; k < K_; ++k) {
            if (load_mask[k]) {
                src_pe[n_src] = drank[k];
                src_loff[n_src] = loff[k];
                src_k[n_src] = k;
                ++n_src;
            } else if (WeightRouteValid(drank[k], loff[k])) {
                dup_k[n_dup++] = k;
            } else {
                ubSk.SetValue(k, 0.0f);
            }
        }

        int32_t w_pend_k[2] = {-1, -1};
        int32_t w_next = 0;
        int32_t dup_i = 0;
        // Weights not issued on the free-slot try path; flushed after SetFlag.
        int32_t deferred_w[kMaxK];
        int32_t n_deferred = 0;

        if (n_src == 0) {
            Duplicate(acc, 0.0f, h_fp32_n_);
            PipeBarrier<PIPE_V>();
            for (; dup_i < n_dup; ++dup_i) {
                IssueWeightOnNextBuf(
                    w_pend_k, w_next, dup_k[dup_i], drank, loff, w0, w1, ubSk);
            }
            TakeWeightBufIfPending(w_pend_k, 0, loff, w0, w1, ubSk);
            TakeWeightBufIfPending(w_pend_k, 1, loff, w0, w1, ubSk);
            return;
        }

        // Kick: H0 and W0 back-to-back (no wait between). Then one free-slot dup.
        IssueHiddenLoad(row0, src_pe[0], src_loff[0], EVENT_ID0);
        IssueWeightOnNextBuf(
            w_pend_k, w_next, src_k[0], drank, loff, w0, w1, ubSk);
        if (dup_i < n_dup &&
            TryIssueWeightOnFreeBuf(
                w_pend_k, w_next, dup_k[dup_i], drank, loff, w0, w1)) {
            ++dup_i;
        }

        bool ping_pending = false;
        bool pong_pending = false;
        int32_t cur = 0;

        for (int32_t i = 0; i < n_src; ++i) {
            TEventID cur_ev = cur == 0 ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<T> cur_row = cur == 0 ? row0 : row1;
            WaitFlag<HardEvent::MTE2_V>(cur_ev);

            n_deferred = 0;
            int32_t next_i = i + 1;
            if (next_i < n_src) {
                int32_t next_buf = 1 - cur;
                TEventID next_ev = next_buf == 0 ? EVENT_ID0 : EVENT_ID1;
                LocalTensor<T> next_row = next_buf == 0 ? row0 : row1;
                if (next_buf == 0 && ping_pending) {
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    ping_pending = false;
                } else if (next_buf == 1 && pong_pending) {
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                    pong_pending = false;
                }
                IssueHiddenLoad(
                    next_row, src_pe[next_i], src_loff[next_i], next_ev);
                // Pair W with H only if a W buf is free — never Take here.
                if (!TryIssueWeightOnFreeBuf(
                        w_pend_k, w_next, src_k[next_i], drank, loff, w0, w1)) {
                    deferred_w[n_deferred++] = src_k[next_i];
                }
            }

            if (dup_i < n_dup) {
                if (TryIssueWeightOnFreeBuf(
                        w_pend_k, w_next, dup_k[dup_i], drank, loff, w0, w1)) {
                    ++dup_i;
                } else {
                    deferred_w[n_deferred++] = dup_k[dup_i];
                    ++dup_i;
                }
            }

            if (i == 0) {
                Cast(acc, cur_row, RoundMode::CAST_NONE, H_);
            } else {
                Cast(tmpF, cur_row, RoundMode::CAST_NONE, H_);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, tmpF, h_fp32_n_);
            }

            // Recycle hidden buf — do not Wait weight before this.
            SetFlag<HardEvent::V_MTE2>(cur_ev);
            if (cur == 0) {
                ping_pending = true;
            } else {
                pong_pending = true;
            }
            cur = 1 - cur;

            // W↔W only: reclaim slots and issue deferred weights after H path.
            for (int32_t d = 0; d < n_deferred; ++d) {
                IssueWeightOnNextBuf(
                    w_pend_k, w_next, deferred_w[d], drank, loff, w0, w1, ubSk);
            }
            if (dup_i < n_dup && (w_pend_k[0] < 0 || w_pend_k[1] < 0)) {
                IssueWeightOnNextBuf(
                    w_pend_k, w_next, dup_k[dup_i], drank, loff, w0, w1, ubSk);
                ++dup_i;
            }
        }

        if (ping_pending) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        if (pong_pending) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        }

        for (; dup_i < n_dup; ++dup_i) {
            IssueWeightOnNextBuf(
                w_pend_k, w_next, dup_k[dup_i], drank, loff, w0, w1, ubSk);
        }
        TakeWeightBufIfPending(w_pend_k, 0, loff, w0, w1, ubSk);
        TakeWeightBufIfPending(w_pend_k, 1, loff, w0, w1, ubSk);
    }

    __aicore__ inline void AccumulateRow(
        const int32_t (&load_mask)[kMaxK],
        const int32_t (&drank)[kMaxK],
        const int32_t (&loff)[kMaxK],
        LocalTensor<float> ubSk,
        int32_t do_weight)
    {
        if (!do_weight) {
            AccumulateRowNoWeight(load_mask, drank, loff);
            return;
        }
        AccumulateRowWithWeight(load_mask, drank, loff, ubSk);
    }

    GlobalTensor<T> gmHiddenLocal_;
    GlobalTensor<float> gmWeightLocal_;
    GlobalTensor<int32_t> gmDst_;
    GlobalTensor<T> gmOut_;
    GlobalTensor<float> gmOutSk_;
    GlobalTensor<int32_t> gmDupGroups_;
    GlobalTensor<int32_t> gmDupLoffs_;
    GlobalTensor<int32_t> gmDupCounts_;
    __gm__ uint8_t* hiddenBase_ = nullptr;
    __gm__ uint8_t* weightBase_ = nullptr;
    int64_t weight_byte_off_ = 0;
    int32_t weight_packed_ = 0;
    LocalTensor<uint8_t> ubBuf_;

    uint32_t offRow0_ = 0, offRow1_ = 0, offStore0_ = 0, offStore1_ = 0;
    uint32_t offTmp_ = 0, offAcc_ = 0;
    uint32_t offGroup_ = 0, offLoff_ = 0;
    uint32_t offDst_ = 0, offMeta_ = 0;
    uint32_t offWLoad0_ = 0, offWLoad1_ = 0, offSk_ = 0;
    uint32_t row_bytes_ = 0;

    int32_t my_rank_ = 0, world_size_ = 1;
    int32_t S_ = 0, H_ = 0, K_ = 0, N_ = 0;
    int32_t with_weight_ = 0, has_weight_ptr_ = 0;
    int32_t s_beg_ = 0, s_end_ = 0, n_tokens_ = 0;
    int32_t n_groups_ = 0, n_dup_loffs_ = 0;
    int32_t dst_copy_n_ = 0, sk_stride_ = 0;
    int32_t h_copy_n_ = 0, h_fp32_n_ = 0;
    int32_t blk_idx_ = 0, blk_cnt_ = 0;
    // Dual store-stage index (prologue: ID2/3; combine out: ID0/1).
    int32_t prologue_store_stage_ = 0;
    int32_t combine_store_stage_ = 0;
    int32_t sk_mte3_pending_ = 0;
};

__aicore__ inline void MoonEPCombineEntry(
    GM_ADDR hidden_buf,
    GM_ADDR weight_buf,
    GM_ADDR dst,
    GM_ADDR dup_groups,
    GM_ADDR dup_loffs,
    GM_ADDR dup_counts,
    GM_ADDR hidden_sk_out,
    GM_ADDR route_weights_sk_out,
    MoonEPCombineTilingData tiling)
{
    int32_t blk_idx = get_block_idx() * get_subblockdim() + get_subblockid();
    int32_t blk_cnt = get_block_num() * get_subblockdim();

    MoonEPCombine<bf16_t> op;
    op.Init(hidden_buf, weight_buf, dst, dup_groups, dup_loffs, dup_counts,
            hidden_sk_out, route_weights_sk_out, tiling, blk_idx, blk_cnt);
    op.Process();
}

}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void moonep_combine_kernel(
    GM_ADDR hidden_buf,
    GM_ADDR weight_buf,
    GM_ADDR dst,
    GM_ADDR dup_groups,
    GM_ADDR dup_loffs,
    GM_ADDR dup_counts,
    GM_ADDR hidden_sk_out,
    GM_ADDR route_weights_sk_out,
    MoonEPCombineTilingData tiling)
{
    MoonEPCombineEntry(hidden_buf, weight_buf, dst,
                       dup_groups, dup_loffs, dup_counts,
                       hidden_sk_out, route_weights_sk_out, tiling);
}

extern "C" ASCEND_MOONEP_EXPORT void moonep_combine_kernel_do(
    void* stream,
    uint8_t* hidden_buf,
    uint8_t* weight_buf,
    uint8_t* dst,
    uint8_t* dup_groups,
    uint8_t* dup_loffs,
    uint8_t* dup_counts,
    uint8_t* hidden_sk_out,
    uint8_t* route_weights_sk_out,
    uint8_t* tiling_host)
{
    auto* td = reinterpret_cast<MoonEPCombineTilingData*>(tiling_host);
    moonep_combine_kernel<<<kMoonEPAivNum, nullptr, stream>>>(
        hidden_buf,
        weight_buf,
        dst,
        dup_groups,
        dup_loffs,
        dup_counts,
        hidden_sk_out,
        route_weights_sk_out,
        *td);
}
