#include "imv/HashBuild.hpp"
#include <iostream>
size_t build_raw(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, Allocator* allo, int entry_size) {
  size_t found = 0;
  auto& ord = db["orders"];
  auto o_orderkey = ord["o_orderkey"].data<types::Integer>();
  int build_key_off = sizeof(runtime::Hashmap::EntryHeader);
  for (size_t i = begin; i < end; ++i) {
    Hashmap::EntryHeader* ptr = (Hashmap::EntryHeader*) allo->allocate(entry_size);
    *(int*) (((char*) ptr) + build_key_off) = o_orderkey[i].value;
    hash_t hash_value = hashFun()(o_orderkey[i], primitives::seed);
    //auto head= ht1.entries+ptr->h.hash;
    hash_table->insert_tagged(ptr, hash_value);
    ++found;
  }
  return found;
}
size_t build_gp(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, Allocator*allo, int entry_size) {
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
      *(int*) (((char*) state[k].ptr) + build_key_off) = o_orderkey[cur].value;
      state[k].hash_value = hashFun()(o_orderkey[cur], primitives::seed);
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
size_t build_amac(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, Allocator*allo, int entry_size) {
  size_t found = 0, cur = begin;
  auto& ord = db["orders"];
  auto o_orderkey = ord["o_orderkey"].data<types::Integer>();
  int build_key_off = sizeof(runtime::Hashmap::EntryHeader);
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
        state[k].ptr = (Hashmap::EntryHeader*) allo->allocate(entry_size);
        *(int*) (((char*) state[k].ptr) + build_key_off) = o_orderkey[cur].value;
        state[k].hash_value = hashFun()(o_orderkey[cur], primitives::seed);
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
size_t build_simd(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, Allocator*allo, int entry_size) {
  size_t found = 0, valid_size = VECTORSIZE;
  auto& ord = db["orders"];
  auto o_orderkey = ord["o_orderkey"].data<types::Integer>();
  int build_key_off = sizeof(runtime::Hashmap::EntryHeader);
  __m512i v_build_key, v_offset, v_base_entry_off, v_entry_addr, v_key_off = _mm512_set1_epi64(build_key_off), v_build_hash_mask, v_zero = _mm512_set1_epi64(0), v_all_ones =
      _mm512_set1_epi64(-1), v_conflict, v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), v_seed = _mm512_set1_epi64(primitives::seed), v_build_hash, v_old_next,
      v_old_next_ptr;

  __mmask8 m_valid_build = -1, m_no_conflict, m_rest;
  __m256i v256_zero = _mm256_set1_epi32(0);
  v_base_entry_off = _mm512_mullo_epi64(v_base_offset, _mm512_set1_epi64(entry_size));
  for (size_t i = begin; i < end;) {
    /// step 1: gather build keys (using gather to compilate with loading discontinuous values)
    v_offset = _mm512_add_epi64(_mm512_set1_epi64(i), v_base_offset);
    i += VECTORSIZE;
    if (i >= end) {
      m_valid_build = (m_valid_build >> (i - end));
      valid_size = end + VECTORSIZE - i;
    }
    auto v256_build_key = _mm512_mask_i64gather_epi32(v256_zero, m_valid_build, v_offset, (const int*)o_orderkey, 4);
    v_build_key = _mm512_cvtepi32_epi64(v256_build_key);

    /// step 2: allocate new entries
    Hashmap::EntryHeader* ptr = (Hashmap::EntryHeader*) allo->allocate(valid_size * entry_size);

    /// step 3: write build keys to new entries
    v_entry_addr = _mm512_add_epi64(_mm512_set1_epi64((uint64_t) ptr), v_base_entry_off);
    _mm512_i64scatter_epi64(0, _mm512_add_epi64(v_entry_addr, v_key_off), v_build_key, 1);

    /// step 4: hashing the build keys (note the hash value cannot be used to directly fetch buckets)
    v_build_hash = runtime::MurMurHash()(v_build_key, v_seed);
#if 0
    //// work only for single thread
    v_build_hash_mask = ((Vec8u(v_build_hash) & Vec8u(hash_table->mask)));
    /// step 5: insert new entries into the buckets
    m_rest = m_valid_build;
    while (m_rest) {
      // note the write conflicts
      v_conflict = _mm512_conflict_epi64(v_build_hash_mask);
      m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
      m_no_conflict = _mm512_kand(m_no_conflict, m_rest);
      found += _mm_popcnt_u32(m_no_conflict);
      // fetch old next
      v_old_next = _mm512_mask_i64gather_epi64(v_zero, m_no_conflict, v_build_hash_mask, hash_table->entries, 8);
      v_old_next_ptr = hash_table->ptr(v_old_next);
      // write old next to the next of new entries
      _mm512_mask_i64scatter_epi64(0, m_no_conflict, v_entry_addr, (v_old_next_ptr), 1);
      // overwrite the old next with new entries
      _mm512_mask_i64scatter_epi64(hash_table->entries, m_no_conflict, v_build_hash_mask, hash_table->update(v_old_next, (v_entry_addr), (v_build_hash)), 8);
      // update
      m_rest = _mm512_kandn(m_no_conflict, m_rest);
      v_build_hash_mask = _mm512_mask_blend_epi64(m_rest, v_all_ones, v_build_hash_mask);
    }

#else
    /// scalar codes due to writhe conflicts among multi-threads
    hash_table->insert_tagged((Vec8u*) (&v_entry_addr), (Vec8u*) (&v_build_hash), valid_size);
    found += valid_size;
#endif
  }
  return found;
}
size_t build_imv(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, Allocator*allo, int entry_size) {
  size_t found = 0, cur = begin;
  uint8_t valid_size = VECTORSIZE, done = 0, k = 0;
  auto& ord = db["orders"];
  auto o_orderkey = ord["o_orderkey"].data<types::Integer>();
  int build_key_off = sizeof(runtime::Hashmap::EntryHeader);
  __m512i v_build_key, v_offset, v_base_entry_off, v_key_off = _mm512_set1_epi64(build_key_off), v_build_hash_mask, v_zero = _mm512_set1_epi64(0), v_all_ones = _mm512_set1_epi64(
      -1), v_conflict, v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), v_seed = _mm512_set1_epi64(primitives::seed);

  __mmask8 m_valid_build = -1, m_no_conflict, m_rest;
  __m256i v256_zero = _mm256_set1_epi32(0), v256_build_key;
  v_base_entry_off = _mm512_mullo_epi64(v_base_offset, _mm512_set1_epi64(entry_size));
  uint64_t* hash_value = nullptr;
  BuildSIMDState state[stateNumSIMD];
  while (done < stateNumSIMD) {
    k = (k >= stateNumSIMD) ? 0 : k;
    if (cur >= end) {
      if (state[k].valid_size == 0 && state[k].stage != 3) {
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
        cur += VECTORSIZE;
        if (cur >= end) {
          m_valid_build = (m_valid_build >> (cur - end));
          valid_size = end + VECTORSIZE - cur;
        }
        v256_build_key = _mm512_mask_i64gather_epi32(v256_zero, m_valid_build, v_offset, (const int*)o_orderkey, 4);
        v_build_key = _mm512_cvtepi32_epi64(v256_build_key);

        /// step 2: allocate new entries
        Hashmap::EntryHeader* ptr = (Hashmap::EntryHeader*) allo->allocate(VECTORSIZE * entry_size);

        /// step 3: write build keys to new entries
        state[k].v_entry_addr = _mm512_add_epi64(_mm512_set1_epi64((uint64_t) ptr), v_base_entry_off);
        _mm512_i64scatter_epi64(0, _mm512_add_epi64(state[k].v_entry_addr, v_key_off), v_build_key, 1);

        /// step 4: hashing the build keys (note the hash value cannot be used to directly fetch buckets)
        state[k].v_hash_value = runtime::MurMurHash()(v_build_key, v_seed);
        state[k].stage = 0;
        state[k].valid_size = valid_size;

        hash_table->prefetchEntry(state[k].v_hash_value);
      }
        break;
      case 0: {
        /// scalar codes due to writhe conflicts among multi-threads
        hash_table->insert_tagged((Vec8u*) (&state[k].v_entry_addr), (Vec8u*) (&state[k].v_hash_value), state[k].valid_size);
        found += state[k].valid_size;
        state[k].stage = 1;
        state[k].valid_size = 0;
      }
        break;
    }
    ++k;
  }

  return found;
}

