#include "rof/AmacBuild.hpp"

size_t amac_build_q11_date(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, Allocator*allo, int entry_size, uint64_t* pos_buff) {
  size_t found = 0, cur = begin;
  auto& d = db["date"];
  auto d_datekey = d["d_datekey"].data<types::Integer>();
  int build_key_off = sizeof(runtime::Hashmap::EntryHeader);
  uint32_t key=0;
  uint8_t done = 0, k = 0;
  BuildState state[stateNum];

  for (int i = 0; i < stateNum; ++i) {
    state[i].stage = 1;
  }

  while (done < stateNum) {
    k = (k >= stateNum) ? 0 : k;
    switch (state[k].stage) {
      case 1: {
        if (cur >= end) {
          ++done;
          state[k].stage = 3;
          break;
        }
        key = d_datekey[pos_buff[cur]].value;
        state[k].ptr = (Hashmap::EntryHeader*) allo->allocate(entry_size);
        *(int*) (((char*) state[k].ptr) + build_key_off) = key;
        state[k].hash_value = hashFun()(key, primitives::seed);
        ++cur;
        state[k].stage = 0;
        hash_table->PrefetchEntry(state[k].hash_value);
      }
        break;
      case 0: {
        hash_table->insert_tagged(state[k].ptr, state[k].hash_value);
        ++found;
        state[k].stage = 1;
      }
        break;
    }
    ++k;
  }
  return found;
}
size_t imv_build_q11_date(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, Allocator*allo, int entry_size, uint64_t* pos_buf) {
  size_t found = 0, cur = begin;
  uint8_t valid_size = VECTORSIZE, done = 0, k = 0;
  auto& ord = db["orders"];
  auto o_orderkey = ord["o_orderkey"].data<types::Integer>();
  int build_key_off = sizeof(runtime::Hashmap::EntryHeader);
  __m512i v_build_key, v_offset, v_base_entry_off, v_key_off = _mm512_set1_epi64(build_key_off), v_build_hash_mask, v_zero = _mm512_set1_epi64(0), v_all_ones = _mm512_set1_epi64(
      -1), v_conflict, v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), v_seed = _mm512_set1_epi64(primitives::seed),v_offset_upper =_mm512_set1_epi64(end) ;

  __mmask8 m_valid_build = -1, m_no_conflict, m_rest;
  __m256i v256_zero = _mm256_set1_epi32(0), v256_build_key;
  v_base_entry_off = _mm512_mullo_epi64(v_base_offset, _mm512_set1_epi64(entry_size));
  uint64_t* hash_value = nullptr;
  BuildSIMDState state[stateNumSIMD];
  while (done < stateNumSIMD) {
    k = (k >= stateNumSIMD) ? 0 : k;
    if (cur >= end) {
      if (state[k].m_valid == 0 && state[k].stage != 3) {
        ++done;
        state[k].stage = 3;
        ++k;
        continue;
      }
    }
    switch (state[k].stage) {
      case 1: {
        /// step 1: gather build keys (using gather to compilate with loading discontinuous values)
        v_offset = _mm512_add_epi64(_mm512_set1_epi64(cur), v_base_offset);
        state[k].m_valid = _mm512_cmpgt_epu64_mask(v_offset_upper, v_offset);
        v256_build_key = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid, v_offset, (const int*)o_orderkey, 4);
        state[k].v_build_key = _mm512_cvtepi32_epi64(v256_build_key);
        cur += VECTORSIZE;

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
