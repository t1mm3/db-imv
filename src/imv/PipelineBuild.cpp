#include "imv/PipelineBuild.hpp"
#include "imv/HashBuild.hpp"
size_t simd_filter_qa_build(size_t& begin, size_t end, Database& db, uint64_t* pos_buff, types::Date constrant) {
  size_t found = 0;
  __mmask8 m_valid = -1, m_eval;
  __m512i v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), v_offset;
  ///////////////////
  // Date => int32_t => 32 bits
  auto& ord = db["orders"];
  auto o_orderdate = ord["o_orderdate"].data<types::Date>();
  __m256i v256_const = _mm256_set1_epi32(constrant.value), v256_col;
  //////////////////////
  for (size_t & cur = begin; cur < end;) {
    v_offset = _mm512_add_epi64(_mm512_set1_epi64(cur), v_base_offset);
    if (cur + VECTORSIZE >= end) {
      m_valid = (m_valid >> (cur + VECTORSIZE - end));
    }
    /////////////////
    v256_col = _mm256_maskz_loadu_epi32(m_valid, (o_orderdate + cur));
    // constrant > col
    m_eval = _mm256_cmpgt_epu32_mask(v256_const, v256_col);
    m_eval = _mm512_kand(m_eval, m_valid);
    //////////////////
    cur += VECTORSIZE;
    _mm512_mask_compressstoreu_epi64(pos_buff + found, m_eval, v_offset);
    found += _mm_popcnt_u32((m_eval));
    if (found + VECTORSIZE >= ROF_VECTOR_SIZE) {
      return found;
    }
  }
  return found;
}
size_t build_gp_qa(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, Allocator*allo, int entry_size, uint64_t* pos_buff) {
  size_t found = 0, cur = begin;
  auto& ord = db["orders"];
  auto o_orderkey = ord["o_orderkey"].data<types::Integer>();
  int build_key_off = sizeof(runtime::Hashmap::EntryHeader);
  uint8_t done = 0, k = 0, stage2_num = 0;
  BuildState state[stateNum];

  while (cur < end) {
    /// step 1: get probe key and compute hashing
    for (k = 0; (k < stateNum) && (cur < end); ++k, ++cur) {
      state[k].ptr = (Hashmap::EntryHeader*) allo->allocate(entry_size);
      *(int*) (((char*) state[k].ptr) + build_key_off) = o_orderkey[pos_buff[cur]].value;
      state[k].hash_value = hashFun()(o_orderkey[pos_buff[cur]], primitives::seed);
      hash_table->PrefetchEntry(state[k].hash_value);
    }
    stage2_num = k;
    for (k = 0; k < stage2_num; ++k) {
      hash_table->insert_tagged(state[k].ptr, state[k].hash_value);
      ++found;
    }

  }
  return found;
}
size_t build_imv_qa(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, Allocator*allo, int entry_size, uint64_t* pos_buff) {
  size_t found = 0, cur = begin;
  uint8_t valid_size = VECTORSIZE, done = 0, k = 0;
  auto& ord = db["orders"];
  auto o_orderkey = ord["o_orderkey"].data<types::Integer>();
  int build_key_off = sizeof(runtime::Hashmap::EntryHeader);
  __m512i v_build_key, v_offset, v_base_entry_off, v_key_off = _mm512_set1_epi64(build_key_off), v_build_hash_mask, v_zero = _mm512_set1_epi64(0), v_all_ones = _mm512_set1_epi64(
      -1), v_conflict, v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), v_seed = _mm512_set1_epi64(primitives::seed), v_offset_upper = _mm512_set1_epi64(end);

  __mmask8 m_no_conflict, m_rest;
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
        if (pos_buff) {
          v_offset = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid, v_offset, (const long long* )pos_buff, 8);
        }
        v256_build_key = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid, v_offset, (const int* )o_orderkey, 4);
        v_build_key = _mm512_cvtepi32_epi64(v256_build_key);
        cur += VECTORSIZE;

        /// step 2: allocate new entries
        Hashmap::EntryHeader* ptr = (Hashmap::EntryHeader*) allo->allocate(VECTORSIZE * entry_size);

        /// step 3: write build keys to new entries
        state[k].v_entry_addr = _mm512_add_epi64(_mm512_set1_epi64((uint64_t) ptr), v_base_entry_off);
        _mm512_i64scatter_epi64(0, _mm512_add_epi64(state[k].v_entry_addr, v_key_off), v_build_key, 1);

        /// step 4: hashing the build keys (note the hash value cannot be used to directly fetch buckets)
        state[k].v_hash_value = runtime::MurMurHash()(v_build_key, v_seed);
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
size_t build_pipeline_imv_qa(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, Allocator*allo, int entry_size, types::Date constrant) {
  size_t found = 0, cur = begin;
  uint8_t valid_size = VECTORSIZE, done = 0, k = 0;
  auto& ord = db["orders"];
  auto o_orderkey = ord["o_orderkey"].data<types::Integer>();
  auto o_orderdate = ord["o_orderdate"].data<types::Date>();
  __m256i v256_const = _mm256_set1_epi32(constrant.value), v256_col;
  int build_key_off = sizeof(runtime::Hashmap::EntryHeader);
  __m512i v_build_key, v_offset, v_base_entry_off, v_key_off = _mm512_set1_epi64(build_key_off), v_build_hash_mask, v_zero = _mm512_set1_epi64(0), v_all_ones = _mm512_set1_epi64(
      -1), v_conflict, v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), v_seed = _mm512_set1_epi64(primitives::seed), v_offset_upper = _mm512_set1_epi64(end),
      v_constrant = _mm512_set1_epi64(constrant.value);

  __mmask8 m_no_conflict, m_rest, m_filter;
  __m256i v256_zero = _mm256_set1_epi32(0), v256_build_key;
  v_base_entry_off = _mm512_mullo_epi64(v_base_offset, _mm512_set1_epi64(entry_size));
  uint64_t* hash_value = nullptr;
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
        v256_build_key = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid, v_offset, (const int* )o_orderkey, 4);
        v256_col = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid, v_offset, (const int* )o_orderdate, 4);

        state[k].v_build_key = _mm512_cvtepi32_epi64(v256_build_key);
        cur += VECTORSIZE;
        /// filter
        m_filter = _mm512_cmpgt_epi64_mask(v_constrant, _mm512_cvtepi32_epi64(v256_col));
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
