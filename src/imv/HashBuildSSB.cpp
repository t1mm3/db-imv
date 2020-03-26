#include "imv/HashBuildSSB.hpp"
#include <iostream>
size_t build_imv_q1x(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, Allocator*allo, int entry_size) {
  size_t found = 0, cur = begin;
  uint8_t valid_size = VECTORSIZE, done = 0, k = 0;
  // --- constants
  const auto relevant_year = types::Integer(1993);

  auto& d = db["date"];
  auto d_year = d["d_year"].data<types::Integer>();
  auto d_datekey = d["d_datekey"].data<types::Integer>();

  int build_key_off = sizeof(runtime::Hashmap::EntryHeader);
  __m512i v_offset, v_base_entry_off, v_key_off = _mm512_set1_epi64(build_key_off), v_build_hash_mask, v_zero = _mm512_set1_epi64(0), v_all_ones = _mm512_set1_epi64(-1),
      v_conflict, v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), v_seed = _mm512_set1_epi64(primitives::seed), v_1993 = _mm512_set1_epi64(relevant_year.value),
      v_offset_upper = _mm512_set1_epi64(end),v_hash_off = _mm512_set1_epi64(8);

  __mmask8 m_no_conflict, m_rest, m_filter;
  __m256i v256_zero = _mm256_set1_epi32(0), v256_build_key,v256_year;
  v_base_entry_off = _mm512_mullo_epi64(v_base_offset, _mm512_set1_epi64(entry_size));
  uint64_t* hash_value = nullptr;
  const int* build_keys = (const int*) d_datekey;

  BuildSIMDState state[stateNumSIMD + 1];
  BuildSIMDState& RVS = state[stateNumSIMD];

  while (true) {
    k = (k >= stateNumSIMD) ? 0 : k;
    if (cur >= end) {
      if (state[k].m_valid == 0 && state[k].stage != 3) {
        ++done;
        state[k].stage = 3;
        ++k;
        continue;
      }
    }
    if (done >= stateNumSIMD) {
      if (RVS.m_valid > 0) {
        k = stateNumSIMD;
      } else {
        break;
      }
    }
    switch (state[k].stage) {
      case 1: {
        /// step 1: gather build keys (using gather to compilate with loading discontinuous values)
        v_offset = _mm512_add_epi64(_mm512_set1_epi64(cur), v_base_offset);
        state[k].m_valid = _mm512_cmpgt_epu64_mask(v_offset_upper, v_offset);
        v256_build_key = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid, v_offset, build_keys, 4);
        v256_year = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid, v_offset, (const int*)d_year, 4);
        state[k].v_build_key = _mm512_cvtepi32_epi64(v256_build_key);
        cur += VECTORSIZE;
        /// filter
        m_filter = _mm512_cmpeq_epi64_mask(_mm512_cvtepi32_epi64(v256_year), v_1993);
        state[k].m_valid = _mm512_kand(state[k].m_valid, m_filter);
        ///compact
         state[k].compact(RVS, done, stateNumSIMD, k, 2, 1);
      }
        break;
      case 2: {
        /// step 2: allocate new entries
        Hashmap::EntryHeader* ptr = (Hashmap::EntryHeader*) allo->allocate(VECTORSIZE * entry_size);

        /// step 3: write build keys to new entries
        state[k].v_entry_addr = _mm512_add_epi64(_mm512_set1_epi64((uint64_t) ptr), v_base_entry_off);
        _mm512_mask_i64scatter_epi64(0, state[k].m_valid, _mm512_add_epi64(state[k].v_entry_addr, v_key_off), state[k].v_build_key, 1);

        /// step 4: hashing the build keys (note the hash value cannot be used to directly fetch buckets)
        state[k].v_hash_value = runtime::MurMurHash()(state[k].v_build_key, v_seed);
        state[k].stage = 0;
        hash_table->prefetchEntry(state[k].v_hash_value);
      }
        break;
      case 0: {
        /// scalar codes due to writhe conflicts among multi-threads
        hash_table->insert_tagged_sel((Vec8u*) (&state[k].v_entry_addr), (Vec8u*) (&state[k].v_hash_value), state[k].m_valid);
        found += _mm_popcnt_u32(state[k].m_valid);
        state[k].stage = 1;
        state[k].m_valid = 0;
      }
        break;
    }
    ++k;
  }

  return found;
}
