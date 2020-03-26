#include "imv/HashAgg.hpp"

#include <stdlib.h>
int agg_constrant = 50;
std::map<uint64_t, uint64_t> match_counts, bucket_counts, end_counts;
__m512i v_all_ones = _mm512_set1_epi64(-1), v_zero = _mm512_set1_epi64(0), v_63 = _mm512_set1_epi64(63);

size_t agg_raw(size_t begin, size_t end, Database& db, Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>* hash_table, PartitionedDeque<PARTITION_SIZE>* partition,
               void** entry_addrs, void** results_entry) {
  size_t found = 0;
  auto& li = db["lineitem"];
#if ORDERKEY
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
#else
  auto l_orderkey = li["l_partkey"].data<types::Integer>();
#endif
  auto l_discount = li["l_discount"].data<types::Numeric<12, 2>>();
  using group_t = Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>::Entry;
  hash_t hash_value;
  group_t* entry = nullptr, *old_entry = nullptr;

  for (size_t cur = begin; cur < end; ++cur) {
    if (nullptr == entry_addrs) {
      hash_value = hashFun()(l_orderkey[cur], primitives::seed);
      entry = hash_table->findOneEntry(l_orderkey[cur], hash_value);
      if (!entry) {
        entry = (group_t*) partition->partition_allocate(hash_value);
        entry->h.hash = hash_value;
        entry->h.next = nullptr;
        entry->k = l_orderkey[cur];
        entry->v = types::Numeric<12, 2>();
        hash_table->insert<false>(*entry);
        ++found;
      }
      entry->v = entry->v + l_discount[cur];
    } else {
      old_entry = (group_t*) entry_addrs[cur];
      entry = hash_table->findOneEntry(old_entry->k, old_entry->h.hash);
      if (!entry) {
        old_entry->h.next = nullptr;
        hash_table->insert<false>(*old_entry);
        results_entry[found++] = entry_addrs[cur];
      } else {
        entry->v = entry->v + old_entry->v;
      }
    }
  }
  return found;
}

size_t agg_amac(size_t begin, size_t end, Database& db, Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>* hash_table, PartitionedDeque<PARTITION_SIZE>* partition,
                void** entry_addrs, void** results_entry) {
  size_t found = 0, pos = 0, cur = begin;
  int k = 0, done = 0, buildkey, probeKey;
  AMACState state[stateNum];
  hash_t probeHash;
  auto& li = db["lineitem"];
#if ORDERKEY
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
#else
  auto l_orderkey = li["l_partkey"].data<types::Integer>();
#endif

  auto l_discount = li["l_discount"].data<types::Numeric<12, 2>>();
  using group_t = Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>::Entry;
  group_t* entry = nullptr, *old_entry = nullptr;

  // initialization
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
        if (nullptr == entry_addrs) {
#if SEQ_PREFETCH
          _mm_prefetch((((char* )(l_orderkey+cur))+PDIS), _MM_HINT_T0);
          _mm_prefetch((((char* )(l_orderkey+cur))+PDIS+64), _MM_HINT_T0);
          _mm_prefetch((((char* )(l_discount+cur))+PDIS), _MM_HINT_T0);
          _mm_prefetch((((char* )(l_discount+cur))+PDIS+64), _MM_HINT_T0);
#endif
          probeKey = l_orderkey[cur].value;
          probeHash = (runtime::MurMurHash()(probeKey, primitives::seed));
          state[k].probeValue = l_discount[cur];
          state[k].tuple_id = cur;
          ++cur;
          state[k].probeKey = probeKey;
          state[k].probeHash = probeHash;
        } else {
#if SEQ_PREFETCH
          _mm_prefetch((((char* )(entry_addrs[cur+PDISD]))), _MM_HINT_T0);
          _mm_prefetch((((char* )(entry_addrs[cur+PDISD]))+64), _MM_HINT_T0);
#endif
          old_entry = (group_t*) entry_addrs[cur];
          state[k].probeHash = old_entry->h.hash;
          state[k].probeKey = old_entry->k.value;
          state[k].probeValue = old_entry->v;
          state[k].tuple_id = cur;
          ++cur;
        }
        hash_table->PrefetchEntry(state[k].probeHash);
        state[k].stage = 2;
      }
        break;
      case 2: {
        state[k].buildMatch = hash_table->find_chain(state[k].probeHash);
        if (nullptr == state[k].buildMatch) {
          state[k].stage = 4;
          --k;  // must immediately shift to case 4
        } else {
          _mm_prefetch((char * )(state[k].buildMatch), _MM_HINT_T0);
          _mm_prefetch((char * )(state[k].buildMatch) + 64, _MM_HINT_T0);
          state[k].stage = 0;
        }
      }
        break;
      case 0: {
        entry = (group_t*) state[k].buildMatch;
        buildkey = entry->k.value;
        if ((buildkey == state[k].probeKey)) {
          entry->v += state[k].probeValue;
          state[k].stage = 1;
          --k;
          break;
        }
        auto entryHeader = entry->h.next;
        if (nullptr == entryHeader) {
          state[k].stage = 4;
          --k;  // must immediately shift to case 4
        } else {
          state[k].buildMatch = entryHeader;
          _mm_prefetch((char * )(entryHeader), _MM_HINT_T0);
          _mm_prefetch((char * )(entryHeader) + 64, _MM_HINT_T0);
        }
      }
        break;
      case 4: {
        if (nullptr == entry_addrs) {
          entry = (group_t*) partition->partition_allocate(state[k].probeHash);
          entry->h.hash = state[k].probeHash;
          entry->h.next = nullptr;
          entry->k = types::Integer(state[k].probeKey);
          entry->v = state[k].probeValue;

        } else {
          entry = (group_t*) entry_addrs[state[k].tuple_id];
          entry->h.next = nullptr;
          results_entry[found] = entry_addrs[state[k].tuple_id];
        }
        auto lastEntry = (group_t*) state[k].buildMatch;
        if (lastEntry == nullptr) { /* the bucket is empty*/
          hash_table->insert<false>(*entry);
        } else {
          lastEntry->h.next = (decltype(lastEntry->h.next)) entry;
        }
        state[k].stage = 1;
        ++found;
        --k;
      }
        break;
    }
    ++k;
  }

  return found;

}
size_t agg_gp(size_t begin, size_t end, Database& db, Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>* hash_table, PartitionedDeque<PARTITION_SIZE>* partition,
              void** entry_addrs, void** results_entry) {
  size_t found = 0, pos = 0, cur = begin;
  int k = 0, done = 0, keyOff = sizeof(runtime::Hashmap::EntryHeader), buildkey, probeKey, valid_size;
  AMACState state[stateNum];
  hash_t probeHash;
  auto& li = db["lineitem"];
#if ORDERKEY
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
#else
  auto l_orderkey = li["l_partkey"].data<types::Integer>();
#endif
  auto l_discount = li["l_discount"].data<types::Numeric<12, 2>>();
  using group_t = Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>::Entry;
  group_t* entry = nullptr, *old_entry = nullptr;

  auto insetNewEntry = [&](AMACState& state) {
    if(nullptr == entry_addrs) {
      entry = (group_t*) partition->partition_allocate(state.probeHash);
      entry->h.hash = state.probeHash;
      entry->h.next = nullptr;
      entry->k = types::Integer(state.probeKey);
      entry->v = state.probeValue;
    } else {
      entry = (group_t*) entry_addrs[state.tuple_id];
      entry->h.next = nullptr;
      results_entry[found] = entry_addrs[state.tuple_id];
    }
    auto lastEntry = (group_t*) state.buildMatch;
    if(lastEntry == nullptr) { /* the bucket is empty*/
      hash_table->insert<false>(*entry);
    } else {
      lastEntry->h.next = (decltype(lastEntry->h.next))entry;
    }

    ++found;
  };
  while (cur < end) {
    /// step 1: get the hash key and compute hash value
    for (k = 0; (k < stateNum) && (cur < end); ++k, ++cur) {
      if (nullptr == entry_addrs) {
#if SEQ_PREFETCH
        _mm_prefetch((((char* )(l_orderkey+cur))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(l_orderkey+cur))+PDIS+64), _MM_HINT_T0);
        _mm_prefetch((((char* )(l_discount+cur))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(l_discount+cur))+PDIS+64), _MM_HINT_T0);
#endif
        probeKey = l_orderkey[cur].value;
        probeHash = (runtime::MurMurHash()(probeKey, primitives::seed));
        state[k].probeValue = l_discount[cur];
        state[k].tuple_id = cur;
        state[k].probeKey = probeKey;
        state[k].probeHash = probeHash;
      } else {
#if SEQ_PREFETCH
        _mm_prefetch((((char* )(entry_addrs[cur+PDISD]))), _MM_HINT_T0);
        _mm_prefetch((((char* )(entry_addrs[cur+PDISD]))+64), _MM_HINT_T0);
#endif
        entry = (group_t*) entry_addrs[cur];
        state[k].probeValue = entry->v;
        state[k].probeHash = entry->h.hash;
        state[k].tuple_id = cur;
        state[k].probeKey = entry->k.value;
      }
      state[k].stage = 0;
      hash_table->PrefetchEntry(state[k].probeHash);
    }
    valid_size = k;
    done = 0;
    /// step 2: fetch the first node in the hash table bucket
    for (k = 0; k < valid_size; ++k) {
      state[k].buildMatch = hash_table->find_chain(state[k].probeHash);
      if (nullptr == state[k].buildMatch) {
        //// must immediately write a new entry
        insetNewEntry(state[k]);
        state[k].stage = 4;
        ++done;
      } else {
        _mm_prefetch((char * )(state[k].buildMatch), _MM_HINT_T0);
        _mm_prefetch((char * )(state[k].buildMatch) + 64, _MM_HINT_T0);
      }
    }
    /// step 3: repeating probing the hash buckets
    while (done < valid_size) {
      for (k = 0; k < valid_size; ++k) {
        // done or need to insert
        if (state[k].stage >= 3) {
          continue;
        }
        entry = (group_t*) state[k].buildMatch;
        buildkey = entry->k.value;
        // found, then update the aggregators
        if ((buildkey == state[k].probeKey)) {
          entry->v += state[k].probeValue;
          state[k].stage = 3;
          ++done;
          continue;
        }
        auto entryHeader = entry->h.next;
        // not found, to insert
        if (nullptr == entryHeader) {
          //// must immediately write a new entry
          insetNewEntry(state[k]);
          state[k].stage = 4;
          ++done;
          continue;
        } else {
          // not found, then continue
          state[k].buildMatch = entryHeader;
          _mm_prefetch((char * )(entryHeader), _MM_HINT_T0);
          _mm_prefetch((char * )(entryHeader) + 64, _MM_HINT_T0);
        }
      }
    }
  }
  return found;
}
size_t agg_simd(size_t begin, size_t end, Database& db, Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>* hash_table, PartitionedDeque<PARTITION_SIZE>* partition,
                void** entry_addrs, void** results_entry) {
  size_t found = 0, pos = 0, cur = begin;
  int k = 0, done = 0, buildkey, probeKey, valid_size;
  AggState state;
  hash_t probeHash;
  auto& li = db["lineitem"];
#if ORDERKEY
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
#else
  auto l_orderkey = li["l_partkey"].data<types::Integer>();
#endif
  auto l_discount = li["l_discount"].data<types::Numeric<12, 2>>();
  using group_t = Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>::Entry;
  __m512i v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), v_zero = _mm512_set1_epi64(0);
  __m512i v_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(end - begin), v_seed = _mm512_set1_epi64(vectorwise::primitives::seed), v_all_ones =
      _mm512_set1_epi64(-1), v_conflict, v_ht_keys, v_hash_mask, v_ht_value, v_next;
  Vec8u u_new_addrs(uint64_t(0)), u_offset_hash(offsetof(group_t, h.hash)), u_offset_k(offsetof(group_t, k)), u_offset_v(offsetof(group_t, v));
  __mmask8 m_no_conflict, m_rest, m_match, m_to_insert;
  state.m_valid_probe = -1;
  void* probe_keys = (void*) l_orderkey, *probe_value = (void*) l_discount;
  __m256i v256_zero = _mm256_set1_epi32(0), v256_probe_keys, v256_probe_value, v256_ht_keys;

  auto insertNewEntry = [&]() {
    Vec8u u_probe_hash(state.v_probe_hash);
    for(int i=0;i<VECTORSIZE;++i) {
      u_new_addrs.entry[i] =0;
      if(m_no_conflict & (1<<i)) {
        u_new_addrs.entry[i] = (uint64_t)partition->partition_allocate(u_probe_hash.entry[i]);
      }
    }
    // write entry->next
      _mm512_mask_i64scatter_epi64(0,m_no_conflict,u_new_addrs.reg,v_zero,1);
      // write entry->hash
      _mm512_mask_i64scatter_epi64(0,m_no_conflict,u_new_addrs + u_offset_hash,state.v_probe_hash,1);
      // write entry->k , NOTE it is 32 bits
      _mm512_mask_i64scatter_epi32(0,m_no_conflict,u_new_addrs + u_offset_k,_mm512_cvtepi64_epi32(state.v_probe_keys),1);
      // write entry->v
      _mm512_mask_i64scatter_epi64(0,m_no_conflict,u_new_addrs + u_offset_v,state.v_probe_value,1);
    };
  for (cur = begin; cur < end;) {
    /// step 1: get offsets
    state.v_probe_offset = _mm512_add_epi64(_mm512_set1_epi64(cur), v_base_offset);
    cur += VECTORSIZE;
    state.m_valid_probe = -1;
    if (cur >= end) {
      state.m_valid_probe = (state.m_valid_probe >> (cur - end));
    }
    if (nullptr == entry_addrs) {
      /// step 2: gather probe keys and values
      v256_probe_keys = _mm512_mask_i64gather_epi32(v256_zero, state.m_valid_probe, state.v_probe_offset, (void* )probe_keys, 4);
      state.v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
      state.v_probe_value = _mm512_mask_i64gather_epi64(v_zero, state.m_valid_probe, state.v_probe_offset, (const long long int* )probe_value, 8);
      /// step 3: compute hash values
      state.v_probe_hash = runtime::MurMurHash()((state.v_probe_keys), (v_seed));
    } else {
      // gather the addresses of entries
      state.v_probe_offset = _mm512_mask_i64gather_epi64(v_zero, state.m_valid_probe, state.v_probe_offset, (const long long int* )entry_addrs, 8);
      v256_probe_keys = _mm512_mask_i64gather_epi32(v256_zero, state.m_valid_probe, state.v_probe_offset + u_offset_k, nullptr, 1);
      state.v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
      state.v_probe_value = _mm512_mask_i64gather_epi64(v_zero, state.m_valid_probe, state.v_probe_offset+u_offset_v, nullptr, 1);
      state.v_probe_hash = _mm512_mask_i64gather_epi64(v_zero, state.m_valid_probe, state.v_probe_offset+u_offset_hash, nullptr, 1);
    }
    /// step 4: find the addresses of corresponding buckets for new probes
    Vec8uM v_new_bucket_addrs = hash_table->find_chain(state.v_probe_hash);

    /// insert new nodes in the corresponding hash buckets
    m_to_insert = _mm512_kandn(v_new_bucket_addrs.mask, state.m_valid_probe);
    v_hash_mask = ((Vec8u(state.v_probe_hash) & Vec8u(hash_table->mask)));
    v_conflict = _mm512_conflict_epi64(v_hash_mask);
    m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
    m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
    if (nullptr == entry_addrs) {
      insertNewEntry();
      // insert the new addresses to the hash table
      _mm512_mask_i64scatter_epi64((long long int * )hash_table->entries, m_no_conflict, v_hash_mask, u_new_addrs.reg, 8);
    } else {
      // set the next of entries = 0
      _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state.v_probe_offset, v_zero, 1);
      _mm512_mask_i64scatter_epi64((long long int * )hash_table->entries, m_no_conflict, v_hash_mask, state.v_probe_offset, 8);
      _mm512_mask_compressstoreu_epi64((results_entry + found), m_no_conflict, state.v_probe_offset);
    }
    found += _mm_popcnt_u32(m_no_conflict);
    // get rid of no-conflict elements
    state.m_valid_probe = _mm512_kandn(m_no_conflict, state.m_valid_probe);
    state.v_bucket_addrs = _mm512_mask_i64gather_epi64(v_all_ones, state.m_valid_probe, v_hash_mask, (long long int* )hash_table->entries, 8);

    while (state.m_valid_probe != 0) {
      /// step 5: gather the all new build keys
      v256_ht_keys = _mm512_mask_i64gather_epi32(v256_zero, state.m_valid_probe, _mm512_add_epi64(state.v_bucket_addrs, u_offset_k.reg), nullptr, 1);
      v_ht_keys = _mm512_cvtepi32_epi64(v256_ht_keys);
      /// step 6: compare the probe keys and build keys and write points
      m_match = _mm512_cmpeq_epi64_mask(state.v_probe_keys, v_ht_keys);
      m_match = _mm512_kand(m_match, state.m_valid_probe);
      /// update the aggregators
      v_conflict = _mm512_conflict_epi64(state.v_bucket_addrs);
      m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
      m_no_conflict = _mm512_kand(m_no_conflict, m_match);

      v_ht_value = _mm512_mask_i64gather_epi64(v_zero, m_no_conflict, _mm512_add_epi64(state.v_bucket_addrs, u_offset_v.reg), nullptr, 1);
      _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state.v_bucket_addrs, u_offset_v.reg), _mm512_add_epi64(state.v_probe_value, v_ht_value), 1);

      state.m_valid_probe = _mm512_kandn(m_no_conflict, state.m_valid_probe);
      // the remaining matches, DO NOT get next
      m_match = _mm512_kandn(m_no_conflict, m_match);

      /// step 7: NOT found, then insert
      v_next = _mm512_mask_i64gather_epi64(v_all_ones, _mm512_kandn(m_match, state.m_valid_probe), state.v_bucket_addrs, nullptr, 1);
      m_to_insert = _mm512_kand(_mm512_kandn(m_match, state.m_valid_probe), _mm512_cmpeq_epi64_mask(v_next, v_zero));
      // get rid of bucket address of matched probes
      v_next = _mm512_mask_blend_epi64(_mm512_kandn(m_match, state.m_valid_probe), v_all_ones, state.v_bucket_addrs);
      v_conflict = _mm512_conflict_epi64(v_next);
      m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
      m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
      if (nullptr == entry_addrs) {
        insertNewEntry();
        // insert the new addresses to the hash table
        _mm512_mask_i64scatter_epi64(0, m_no_conflict, state.v_bucket_addrs, u_new_addrs.reg, 1);
      } else {
        // set the next of entries = 0
        _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state.v_probe_offset, v_zero, 1);
        _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state.v_bucket_addrs, state.v_probe_offset, 1);
        _mm512_mask_compressstoreu_epi64((results_entry + found), m_no_conflict, state.v_probe_offset);
      }
      found += _mm_popcnt_u32(m_no_conflict);
      state.m_valid_probe = _mm512_kandn(m_no_conflict, state.m_valid_probe);
      v_next = _mm512_mask_i64gather_epi64(v_all_ones, state.m_valid_probe, state.v_bucket_addrs, nullptr, 1);
      // the remaining matches, DO NOT get next
      state.v_bucket_addrs = _mm512_mask_blend_epi64(m_match, v_next, state.v_bucket_addrs);
    }
  }

  return found;
}

inline void insertNewEntry(AggState& state, Vec8u& u_new_addrs, __mmask8 m_no_conflict, PartitionedDeque<PARTITION_SIZE>* partition, __m512i& u_offset_hash, __m512i& u_offset_k, __m512i& u_offset_v) {
  Vec8u u_probe_hash(state.v_probe_hash);
  for(int i=0;i<VECTORSIZE;++i) {
    u_new_addrs.entry[i] =0;
    if(m_no_conflict & (1<<i)) {
      u_new_addrs.entry[i] = (uint64_t)partition->partition_allocate(u_probe_hash.entry[i]);
    }
  }
  // write entry->next
  _mm512_mask_i64scatter_epi64(0,m_no_conflict,u_new_addrs.reg,v_zero,1);
  // write entry->hash
  _mm512_mask_i64scatter_epi64(0,m_no_conflict,u_new_addrs + u_offset_hash,state.v_probe_hash,1);
  // write entry->k , NOTE it is 32 bits
  _mm512_mask_i64scatter_epi32(0,m_no_conflict,u_new_addrs + u_offset_k,_mm512_cvtepi64_epi32(state.v_probe_keys),1);
  // write entry->v
  _mm512_mask_i64scatter_epi64(0,m_no_conflict,u_new_addrs + u_offset_v,state.v_probe_value,1);
}
inline void mergeKeys(AggState& state) {
  auto v_conflict = _mm512_conflict_epi64(state.v_probe_keys);
  auto m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
  m_no_conflict = _mm512_kand(m_no_conflict, state.m_valid_probe);
  auto m_conflict = _mm512_kandn(m_no_conflict, state.m_valid_probe);
  if(0==m_conflict) return;
  uint64_t* pos_v = (uint64_t*) &state.v_probe_value;
  auto v_lzeros = _mm512_lzcnt_epi64(v_conflict);
  v_lzeros = _mm512_sub_epi64(v_63, v_lzeros);
  uint64_t* pos_lz = (uint64_t*) &v_lzeros;
  for (int i = VECTORSIZE - 1; i >= 0; --i) {
    if ((m_conflict & (1 << i))) {
      pos_v[pos_lz[i]] += pos_v[i];
    }
  }
  state.m_valid_probe = m_no_conflict;
  state.v_probe_keys = _mm512_mask_blend_epi64(m_no_conflict, v_all_ones, state.v_probe_keys);
}
inline void insertAllNewEntry(AggState& state, Vec8u& u_new_addrs, __m512i& v_conflict, __mmask8 m_to_insert, __mmask8 m_no_conflict, PartitionedDeque<PARTITION_SIZE>* partition, __m512i& u_offset_hash, __m512i& u_offset_k, __m512i& u_offset_v) {
  Vec8u u_probe_hash(state.v_probe_hash);
  for(int i=0;i<VECTORSIZE;++i) {
    u_new_addrs.entry[i] =0;
    if(m_to_insert & (1<<i)) {
      u_new_addrs.entry[i] = (uint64_t)partition->partition_allocate(u_probe_hash.entry[i]);
    }
  }
  // write entry->next
  _mm512_mask_i64scatter_epi64(0,m_to_insert,u_new_addrs.reg,v_zero,1);
  // write entry->hash
  _mm512_mask_i64scatter_epi64(0,m_to_insert,u_new_addrs + u_offset_hash,state.v_probe_hash,1);
  // write entry->k , NOTE it is 32 bits
  _mm512_mask_i64scatter_epi32(0,m_to_insert,u_new_addrs + u_offset_k,_mm512_cvtepi64_epi32(state.v_probe_keys),1);
  // write entry->v
  _mm512_mask_i64scatter_epi64(0,m_to_insert,u_new_addrs + u_offset_v,state.v_probe_value,1);
  // write the addresses to the previous'next
  auto v_lzeros = _mm512_lzcnt_epi64(v_conflict);
  v_lzeros = _mm512_sub_epi64(v_63,v_lzeros);
  auto to_scatt= _mm512_kandn(m_no_conflict,m_to_insert);
  auto v_previous = _mm512_maskz_permutexvar_epi64(to_scatt,v_lzeros,u_new_addrs.reg);
  _mm512_mask_i64scatter_epi64(0,to_scatt,v_previous,u_new_addrs.reg,1);
}
size_t agg_imv_serial(size_t begin, size_t end, Database& db, Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>* hash_table,
                      PartitionedDeque<PARTITION_SIZE>* partition, void** entry_addrs, void** results_entry) {
  size_t found = 0, pos = 0, cur = begin;
  int k = 0, done = 0, buildkey, probeKey, valid_size, imvNum = vectorwise::Hashjoin::imvNum, imvNum1 = vectorwise::Hashjoin::imvNum + 1;
  AggState state[stateNumSIMD + 2];
  hash_t probeHash;
  auto& li = db["lineitem"];
#if ORDERKEY
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
#else
  auto l_orderkey = li["l_partkey"].data<types::Integer>();
#endif
  auto l_discount = li["l_discount"].data<types::Numeric<12, 2>>();
  using group_t = Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>::Entry;
  __m512i v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), v_zero = _mm512_set1_epi64(0);
  __m512i v_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(end - begin), v_seed = _mm512_set1_epi64(vectorwise::primitives::seed), v_all_ones =
      _mm512_set1_epi64(-1), v_conflict, v_ht_keys, v_hash_mask, v_ht_value, v_next;
  Vec8u u_new_addrs(uint64_t(0)), u_offset_hash(offsetof(group_t, h.hash)), u_offset_k(offsetof(group_t, k)), u_offset_v(offsetof(group_t, v));
  __mmask8 m_no_conflict, m_rest, m_match, m_to_insert, mask[VECTORSIZE + 1];
  void* probe_keys = (void*) l_orderkey, *probe_value = (void*) l_discount;
  __m256i v256_zero = _mm256_set1_epi32(0), v256_probe_keys, v256_probe_value, v256_ht_keys;
  uint8_t num, num_temp;
  for (int i = 0; i <= VECTORSIZE; ++i) {
    mask[i] = (1 << i) - 1;
  }

  while (true) {
    k = (k >= imvNum) ? 0 : k;
    if ((cur >= end)) {
      if (state[k].m_valid_probe == 0 && state[k].stage != 3) {
        ++done;
        state[k].stage = 3;
        ++k;
        continue;
      }
    }
    if (done >= imvNum) {
      if (state[imvNum1].m_valid_probe > 0) {
        k = imvNum1;
      } else {
        if (state[imvNum].m_valid_probe > 0) {
          k = imvNum;
        } else {
          break;
        }
      }
    }
    switch (state[k].stage) {
      case 1: {
        /// step 1: get offsets
        state[k].v_probe_offset = _mm512_add_epi64(_mm512_set1_epi64(cur), v_base_offset);
        cur += VECTORSIZE;
        state[k].m_valid_probe = -1;
        if (cur >= end) {
          state[k].m_valid_probe = (state[k].m_valid_probe >> (cur - end));
        }
        if (nullptr == entry_addrs) {
          /// step 2: gather probe keys and values
#if SEQ_PREFETCH
          _mm_prefetch((((char* )(l_orderkey+cur))+PDIS), _MM_HINT_T0);
          _mm_prefetch((((char* )(l_orderkey+cur))+PDIS+64), _MM_HINT_T0);
          _mm_prefetch((((char* )(l_discount+cur))+PDIS), _MM_HINT_T0);
          _mm_prefetch((((char* )(l_discount+cur))+PDIS+64), _MM_HINT_T0);
#endif
          v256_probe_keys = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid_probe, state[k].v_probe_offset, (const int* )probe_keys, 4);
          state[k].v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
          state[k].v_probe_value = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset, (const long long int* )probe_value, 8);
          /// step 3: compute hash values
          state[k].v_probe_hash = runtime::MurMurHash()((state[k].v_probe_keys), (v_seed));
        } else {
#if SEQ_PREFETCH
          _mm_prefetch((((char* )(entry_addrs[cur+PDISD]))), _MM_HINT_T0);
          _mm_prefetch((((char* )(entry_addrs[cur+PDISD]))+64), _MM_HINT_T0);
#endif
          // gather the addresses of entries
          state[k].v_probe_offset = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset, (const long long int* )entry_addrs, 8);
          v256_probe_keys = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid_probe, state[k].v_probe_offset + u_offset_k, nullptr, 1);
          state[k].v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
          state[k].v_probe_value = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset+u_offset_v, nullptr, 1);
          state[k].v_probe_hash = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset+u_offset_hash, nullptr, 1);
        }
        state[k].stage = 2;
        hash_table->prefetchEntry(state[k].v_probe_hash);
      }
        break;
      case 2: {
        /// step 4: find the addresses of corresponding buckets for new probes
        Vec8uM v_new_bucket_addrs = hash_table->find_chain(state[k].v_probe_hash);

        /// insert new nodes in the corresponding hash buckets
        m_to_insert = _mm512_kandn(v_new_bucket_addrs.mask, state[k].m_valid_probe);
        v_hash_mask = ((Vec8u(state[k].v_probe_hash) & Vec8u(hash_table->mask)));
        v_hash_mask = _mm512_mask_blend_epi64(state[k].m_valid_probe, v_all_ones, v_hash_mask);

        v_conflict = _mm512_conflict_epi64(v_hash_mask);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_no_conflict) {
          if (nullptr == entry_addrs) {
            insertNewEntry(state[k], u_new_addrs, m_no_conflict, partition, u_offset_hash.reg, u_offset_k.reg, u_offset_v.reg);
            // insert the new addresses to the hash table
            _mm512_mask_i64scatter_epi64((long long int* )hash_table->entries, m_no_conflict, v_hash_mask, u_new_addrs.reg, 8);
          } else {
            // set the next of entries = 0
            _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state[k].v_probe_offset, v_zero, 1);
            _mm512_mask_i64scatter_epi64((long long int* )hash_table->entries, m_no_conflict, v_hash_mask, state[k].v_probe_offset, 8);
            _mm512_mask_compressstoreu_epi64((results_entry + found), m_no_conflict, state[k].v_probe_offset);
          }
          found += _mm_popcnt_u32(m_no_conflict);
          // get rid of no-conflict elements
          state[k].m_valid_probe = _mm512_kandn(m_no_conflict, state[k].m_valid_probe);
        }
        state[k].v_bucket_addrs = _mm512_mask_i64gather_epi64(v_all_ones, state[k].m_valid_probe, v_hash_mask, (const long long int* )hash_table->entries, 8);
#if 0
        v_prefetch(state[k].v_bucket_addrs);
        state[k].stage = 0;
#else
        num = _mm_popcnt_u32(state[k].m_valid_probe);
        if (num == VECTORSIZE) {
          v_prefetch(state[k].v_bucket_addrs);
          state[k].stage = 0;
        } else {
          if ((done < imvNum)) {
            num_temp = _mm_popcnt_u32(state[imvNum1].m_valid_probe);
            if (num + num_temp < VECTORSIZE) {
              // compress imv_state[k]
              state[k].compress();
              // expand imv_state[k] -> imv_state[imvNum]
              state[imvNum1].expand(state[k]);
              state[imvNum1].m_valid_probe = mask[num + num_temp];
              state[k].m_valid_probe = 0;
              state[k].stage = 1;
              state[imvNum1].stage = 0;
              --k;
              break;
            } else {
              // expand imv_state[imvNum] -> expand imv_state[k]
              state[k].expand(state[imvNum1]);
              state[imvNum1].m_valid_probe = _mm512_kand(state[imvNum1].m_valid_probe, _mm512_knot(mask[VECTORSIZE - num]));
              // compress imv_state[imvNum]
              state[imvNum1].compress();
              state[imvNum1].m_valid_probe = state[imvNum1].m_valid_probe >> (VECTORSIZE - num);
              state[k].m_valid_probe = mask[VECTORSIZE];
              state[k].stage = 0;
              state[imvNum1].stage = 0;
              v_prefetch(state[k].v_bucket_addrs);
            }
          }
        }
#endif
      }
        break;
      case 0: {
        /// step 5: gather the all new build keys
        v256_ht_keys = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid_probe, _mm512_add_epi64(state[k].v_bucket_addrs, u_offset_k.reg), nullptr, 1);
        v_ht_keys = _mm512_cvtepi32_epi64(v256_ht_keys);
        /// step 6: compare the probe keys and build keys and write points
        m_match = _mm512_cmpeq_epi64_mask(state[k].v_probe_keys, v_ht_keys);
        m_match = _mm512_kand(m_match, state[k].m_valid_probe);
        /// update the aggregators
        v_conflict = _mm512_conflict_epi64(state[k].v_bucket_addrs);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_match);

        v_ht_value = _mm512_mask_i64gather_epi64(v_zero, m_no_conflict, _mm512_add_epi64(state[k].v_bucket_addrs, u_offset_v.reg), nullptr, 1);
        _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].v_bucket_addrs, u_offset_v.reg), _mm512_add_epi64(state[k].v_probe_value, v_ht_value), 1);

        state[k].m_valid_probe = _mm512_kandn(m_no_conflict, state[k].m_valid_probe);
        // the remaining matches, DO NOT get next
        m_match = _mm512_kandn(m_no_conflict, m_match);

        /// step 7: NOT found, then insert
        v_next = _mm512_mask_i64gather_epi64(v_all_ones, _mm512_kandn(m_match, state[k].m_valid_probe), state[k].v_bucket_addrs, nullptr, 1);
        m_to_insert = _mm512_kand(_mm512_kandn(m_match, state[k].m_valid_probe), _mm512_cmpeq_epi64_mask(v_next, v_zero));
        // get rid of bucket address of matched probes
        v_next = _mm512_mask_blend_epi64(_mm512_kandn(m_match, state[k].m_valid_probe), v_all_ones, state[k].v_bucket_addrs);
        v_conflict = _mm512_conflict_epi64(v_next);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_no_conflict) {
          if (nullptr == entry_addrs) {
            insertNewEntry(state[k], u_new_addrs, m_no_conflict, partition, u_offset_hash.reg, u_offset_k.reg, u_offset_v.reg);
            // insert the new addresses to the hash table
            _mm512_mask_i64scatter_epi64(0, m_no_conflict, state[k].v_bucket_addrs, u_new_addrs.reg, 1);
          } else {
            // set the next of entries = 0
            _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state[k].v_probe_offset, v_zero, 1);
            _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state[k].v_bucket_addrs, state[k].v_probe_offset, 1);
            _mm512_mask_compressstoreu_epi64((results_entry + found), m_no_conflict, state[k].v_probe_offset);
          }
          found += _mm_popcnt_u32(m_no_conflict);

          state[k].m_valid_probe = _mm512_kandn(m_no_conflict, state[k].m_valid_probe);
        }
        v_next = _mm512_mask_i64gather_epi64(v_all_ones, state[k].m_valid_probe, state[k].v_bucket_addrs, nullptr, 1);
        // the remaining matches, DO NOT get next
        state[k].v_bucket_addrs = _mm512_mask_blend_epi64(m_match, v_next, state[k].v_bucket_addrs);

        num = _mm_popcnt_u32(state[k].m_valid_probe);
        if (num == VECTORSIZE || done >= imvNum) {
          v_prefetch(state[k].v_bucket_addrs);
        } else {
          if ((done < imvNum)) {
            num_temp = _mm_popcnt_u32(state[imvNum].m_valid_probe);
            if (num + num_temp < VECTORSIZE) {
              // compress imv_state[k]
              state[k].compress();
              // expand imv_state[k] -> imv_state[imvNum]
              state[imvNum].expand(state[k]);
              state[imvNum].m_valid_probe = mask[num + num_temp];
              state[k].m_valid_probe = 0;
              state[k].stage = 1;
              state[imvNum].stage = 0;
              --k;
              break;
            } else {
              // expand imv_state[imvNum] -> expand imv_state[k]
              state[k].expand(state[imvNum]);
              state[imvNum].m_valid_probe = _mm512_kand(state[imvNum].m_valid_probe, _mm512_knot(mask[VECTORSIZE - num]));
              // compress imv_state[imvNum]
              state[imvNum].compress();
              state[imvNum].m_valid_probe = state[imvNum].m_valid_probe >> (VECTORSIZE - num);
              state[k].m_valid_probe = mask[VECTORSIZE];
              state[k].stage = 0;
              state[imvNum].stage = 0;
              v_prefetch(state[k].v_bucket_addrs);
            }
          }
        }

      }
        break;
    }
    ++k;
  }

  return found;
}
/*
 * hybrid:
 * first merge all possible equal keys at stage 2, but there are equal keys at stage 0, so adopt the serial way to update and insert
 */
size_t agg_imv_hybrid(size_t begin, size_t end, Database& db, Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>* hash_table,
                      PartitionedDeque<PARTITION_SIZE>* partition, void** entry_addrs, void** results_entry) {
  size_t found = 0, pos = 0, cur = begin;
  int k = 0, done = 0, buildkey, probeKey, valid_size, imvNum = vectorwise::Hashjoin::imvNum, imvNum1 = vectorwise::Hashjoin::imvNum + 1;
  AggState state[stateNumSIMD + 2];
  hash_t probeHash;
  auto& li = db["lineitem"];
#if ORDERKEY
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
#else
  auto l_orderkey = li["l_partkey"].data<types::Integer>();
#endif
  auto l_discount = li["l_discount"].data<types::Numeric<12, 2>>();
  using group_t = Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>::Entry;
  __m512i v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), v_zero = _mm512_set1_epi64(0), v_lzeros, v_63 = _mm512_set1_epi64(63);
  __m512i v_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(end - begin), v_seed = _mm512_set1_epi64(vectorwise::primitives::seed), v_all_ones =
      _mm512_set1_epi64(-1), v_conflict, v_ht_keys, v_hash_mask, v_ht_value, v_next;
  Vec8u u_new_addrs(uint64_t(0)), u_offset_hash(offsetof(group_t, h.hash)), u_offset_k(offsetof(group_t, k)), u_offset_v(offsetof(group_t, v));
  __mmask8 m_no_conflict, m_rest, m_match, m_to_insert, mask[VECTORSIZE + 1];
  void* probe_keys = (void*) l_orderkey, *probe_value = (void*) l_discount;
  __m256i v256_zero = _mm256_set1_epi32(0), v256_probe_keys, v256_probe_value, v256_ht_keys;
  uint8_t num, num_temp;
  for (int i = 0; i <= VECTORSIZE; ++i) {
    mask[i] = (1 << i) - 1;
  }

  while (true) {
    k = (k >= imvNum) ? 0 : k;
    if ((cur >= end)) {
      if (state[k].m_valid_probe == 0 && state[k].stage != 3) {
        ++done;
        state[k].stage = 3;
        ++k;
        continue;
      }
    }
    if (done >= imvNum) {
      if (state[imvNum1].m_valid_probe > 0) {
        k = imvNum1;
      } else {
        if (state[imvNum].m_valid_probe > 0) {
          k = imvNum;
        } else {
          break;
        }
      }
    }
    switch (state[k].stage) {
      case 1: {
        /// step 1: get offsets
        state[k].v_probe_offset = _mm512_add_epi64(_mm512_set1_epi64(cur), v_base_offset);
        cur += VECTORSIZE;
        state[k].m_valid_probe = -1;
        if (cur >= end) {
          state[k].m_valid_probe = (state[k].m_valid_probe >> (cur - end));
        }
        if (nullptr == entry_addrs) {
          /// step 2: gather probe keys and values
#if SEQ_PREFETCH
          _mm_prefetch((((char* )(l_orderkey+cur))+PDIS), _MM_HINT_T0);
          _mm_prefetch((((char* )(l_orderkey+cur))+PDIS+64), _MM_HINT_T0);
          _mm_prefetch((((char* )(l_discount+cur))+PDIS), _MM_HINT_T0);
          _mm_prefetch((((char* )(l_discount+cur))+PDIS+64), _MM_HINT_T0);
#endif
          v256_probe_keys = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid_probe, state[k].v_probe_offset, (const int* )probe_keys, 4);
          state[k].v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
          state[k].v_probe_value = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset, (const long long int* )probe_value, 8);
          /// step 3: compute hash values
          state[k].v_probe_hash = runtime::MurMurHash()((state[k].v_probe_keys), (v_seed));
        } else {
#if SEQ_PREFETCH
          _mm_prefetch((((char* )(entry_addrs[cur+PDISD]))), _MM_HINT_T0);
          _mm_prefetch((((char* )(entry_addrs[cur+PDISD]))+64), _MM_HINT_T0);
#endif
          // gather the addresses of entries
          state[k].v_probe_offset = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset, (const long long int * )entry_addrs, 8);
          v256_probe_keys = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid_probe, state[k].v_probe_offset + u_offset_k, nullptr, 1);
          state[k].v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
          state[k].v_probe_value = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset+u_offset_v, nullptr, 1);
          state[k].v_probe_hash = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset+u_offset_hash, nullptr, 1);
        }

        state[k].stage = 2;
        hash_table->prefetchEntry(state[k].v_probe_hash);
      }
        break;
      case 2: {

        mergeKeys(state[k]);
        /// step 4: find the addresses of corresponding buckets for new probes
        Vec8uM v_new_bucket_addrs = hash_table->find_chain(state[k].v_probe_hash);

        /// insert new nodes in the corresponding hash buckets
        m_to_insert = _mm512_kandn(v_new_bucket_addrs.mask, state[k].m_valid_probe);
        v_hash_mask = ((Vec8u(state[k].v_probe_hash) & Vec8u(hash_table->mask)));
        v_hash_mask = _mm512_mask_blend_epi64(state[k].m_valid_probe, v_all_ones, v_hash_mask);
        v_conflict = _mm512_conflict_epi64(v_hash_mask);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_no_conflict) {
          if (nullptr == entry_addrs) {
            insertNewEntry(state[k], u_new_addrs, m_no_conflict, partition, u_offset_hash.reg, u_offset_k.reg, u_offset_v.reg);
            // insert the new addresses to the hash table
            _mm512_mask_i64scatter_epi64((long long int* )hash_table->entries, m_no_conflict, v_hash_mask, u_new_addrs.reg, 8);
          } else {
            // set the next of entries = 0
            _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state[k].v_probe_offset, v_zero, 1);
            // must write back the merged values!!!!!!!
            _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state[k].v_probe_offset + u_offset_v, state[k].v_probe_value, 1);
            _mm512_mask_i64scatter_epi64((long long int* )hash_table->entries, m_no_conflict, v_hash_mask, state[k].v_probe_offset, 8);
            _mm512_mask_compressstoreu_epi64((results_entry + found), m_no_conflict, state[k].v_probe_offset);
          }
          found += _mm_popcnt_u32(m_no_conflict);
          // get rid of no-conflict elements
          state[k].m_valid_probe = _mm512_kandn(m_no_conflict, state[k].m_valid_probe);
        }
        state[k].v_bucket_addrs = _mm512_mask_i64gather_epi64(v_all_ones, state[k].m_valid_probe, v_hash_mask, (const long long int* )hash_table->entries, 8);
#if 0
        v_prefetch(state[k].v_bucket_addrs);
        state[k].stage = 0;
#else
        num = _mm_popcnt_u32(state[k].m_valid_probe);
        if (num == VECTORSIZE) {
          v_prefetch(state[k].v_bucket_addrs);
          state[k].stage = 0;
        } else {
          if ((done < imvNum)) {
            num_temp = _mm_popcnt_u32(state[imvNum1].m_valid_probe);
            if (num + num_temp < VECTORSIZE) {
              // compress imv_state[k]
              state[k].compress();
              // expand imv_state[k] -> imv_state[imvNum]
              state[imvNum1].expand(state[k]);
              state[imvNum1].m_valid_probe = mask[num + num_temp];
              state[k].m_valid_probe = 0;
              state[k].stage = 1;
              state[imvNum1].stage = 0;
              --k;
              break;
            } else {
              // expand imv_state[imvNum] -> expand imv_state[k]
              state[k].expand(state[imvNum1]);
              state[imvNum1].m_valid_probe = _mm512_kand(state[imvNum1].m_valid_probe, _mm512_knot(mask[VECTORSIZE - num]));
              // compress imv_state[imvNum]
              state[imvNum1].compress();
              state[imvNum1].m_valid_probe = state[imvNum1].m_valid_probe >> (VECTORSIZE - num);
              state[k].m_valid_probe = mask[VECTORSIZE];
              state[k].stage = 0;
              state[imvNum1].stage = 0;
              v_prefetch(state[k].v_bucket_addrs);
            }
          }
        }
#endif
      }
        break;
      case 0: {
        /// step 5: gather the all new build keys
        v256_ht_keys = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid_probe, _mm512_add_epi64(state[k].v_bucket_addrs, u_offset_k.reg), nullptr, 1);
        v_ht_keys = _mm512_cvtepi32_epi64(v256_ht_keys);
        /// step 6: compare the probe keys and build keys and write points
        m_match = _mm512_cmpeq_epi64_mask(state[k].v_probe_keys, v_ht_keys);
        m_match = _mm512_kand(m_match, state[k].m_valid_probe);
        /// update the aggregators
        v_conflict = _mm512_conflict_epi64(state[k].v_bucket_addrs);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_match);

        v_ht_value = _mm512_mask_i64gather_epi64(v_zero, m_no_conflict, _mm512_add_epi64(state[k].v_bucket_addrs, u_offset_v.reg), nullptr, 1);
        _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].v_bucket_addrs, u_offset_v.reg), _mm512_add_epi64(state[k].v_probe_value, v_ht_value), 1);

        state[k].m_valid_probe = _mm512_kandn(m_no_conflict, state[k].m_valid_probe);
        // the remaining matches, DO NOT get next
        m_match = _mm512_kandn(m_no_conflict, m_match);

        /// step 7: NOT found, then insert
        v_next = _mm512_mask_i64gather_epi64(v_all_ones, _mm512_kandn(m_match, state[k].m_valid_probe), state[k].v_bucket_addrs, nullptr, 1);
        m_to_insert = _mm512_kand(_mm512_kandn(m_match, state[k].m_valid_probe), _mm512_cmpeq_epi64_mask(v_next, v_zero));
        // get rid of bucket address of matched probes
        v_next = _mm512_mask_blend_epi64(_mm512_kandn(m_match, state[k].m_valid_probe), v_all_ones, state[k].v_bucket_addrs);
        v_conflict = _mm512_conflict_epi64(v_next);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_no_conflict) {
          if (nullptr == entry_addrs) {
            insertNewEntry(state[k], u_new_addrs, m_no_conflict, partition, u_offset_hash.reg, u_offset_k.reg, u_offset_v.reg);
            // insert the new addresses to the hash table
            _mm512_mask_i64scatter_epi64(0, m_no_conflict, state[k].v_bucket_addrs, u_new_addrs.reg, 1);
          } else {
            // set the next of entries = 0
            _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state[k].v_probe_offset, v_zero, 1);
            _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state[k].v_bucket_addrs, state[k].v_probe_offset, 1);
            _mm512_mask_compressstoreu_epi64((results_entry + found), m_no_conflict, state[k].v_probe_offset);
          }
          found += _mm_popcnt_u32(m_no_conflict);

          state[k].m_valid_probe = _mm512_kandn(m_no_conflict, state[k].m_valid_probe);
        }
        v_next = _mm512_mask_i64gather_epi64(v_all_ones, state[k].m_valid_probe, state[k].v_bucket_addrs, nullptr, 1);
        // the remaining matches, DO NOT get next
        state[k].v_bucket_addrs = _mm512_mask_blend_epi64(m_match, v_next, state[k].v_bucket_addrs);

        num = _mm_popcnt_u32(state[k].m_valid_probe);
        if (num == VECTORSIZE || done >= imvNum) {
          v_prefetch(state[k].v_bucket_addrs);
        } else {
          if ((done < imvNum)) {
            num_temp = _mm_popcnt_u32(state[imvNum].m_valid_probe);
            if (num + num_temp < VECTORSIZE) {
              // compress imv_state[k]
              state[k].compress();
              // expand imv_state[k] -> imv_state[imvNum]
              state[imvNum].expand(state[k]);
              state[imvNum].m_valid_probe = mask[num + num_temp];
              state[k].m_valid_probe = 0;
              state[k].stage = 1;
              state[imvNum].stage = 0;
              --k;
              break;
            } else {
              // expand imv_state[imvNum] -> expand imv_state[k]
              state[k].expand(state[imvNum]);
              state[imvNum].m_valid_probe = _mm512_kand(state[imvNum].m_valid_probe, _mm512_knot(mask[VECTORSIZE - num]));
              // compress imv_state[imvNum]
              state[imvNum].compress();
              state[imvNum].m_valid_probe = state[imvNum].m_valid_probe >> (VECTORSIZE - num);
              state[k].m_valid_probe = mask[VECTORSIZE];
              state[k].stage = 0;
              state[imvNum].stage = 0;
              v_prefetch(state[k].v_bucket_addrs);
            }
          }
        }

      }
        break;
    }
    ++k;
  }

  return found;
}

size_t agg_imv1(size_t begin, size_t end, Database& db, Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>* hash_table, PartitionedDeque<PARTITION_SIZE>* partition,
                void** entry_addrs, void** results_entry) {
  size_t found = 0, pos = 0, cur = begin;
  int k = 0, done = 0, buildkey, probeKey, valid_size, imvNum = vectorwise::Hashjoin::imvNum, imvNum1 = vectorwise::Hashjoin::imvNum + 1;
  AggState state[stateNumSIMD + 2];
  hash_t probeHash;
  auto& li = db["lineitem"];
#if ORDERKEY
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
#else
  auto l_orderkey = li["l_partkey"].data<types::Integer>();
#endif
  auto l_discount = li["l_discount"].data<types::Numeric<12, 2>>();
  using group_t = Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>::Entry;
  __m512i v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), v_zero = _mm512_set1_epi64(0);
  __m512i v_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(end), v_seed = _mm512_set1_epi64(vectorwise::primitives::seed), v_all_ones =
      _mm512_set1_epi64(-1), v_conflict, v_ht_keys, v_hash_mask, v_ht_value, v_next,v_ht_mask = _mm512_set1_epi64(hash_table->mask),v_new_addr;
  Vec8u u_new_addrs(uint64_t(0)), u_offset_hash(offsetof(group_t, h.hash)), u_offset_k(offsetof(group_t, k)), u_offset_v(offsetof(group_t, v));
  __mmask8 m_no_conflict, m_rest, m_match, m_to_insert, mask[VECTORSIZE + 1];
  void* probe_keys = (void*) l_orderkey, *probe_value = (void*) l_discount;
  __m256i v256_zero = _mm256_set1_epi32(0), v256_probe_keys, v256_probe_value, v256_ht_keys;
  uint8_t num, num_temp;
  for (int i = 0; i <= VECTORSIZE; ++i) {
    mask[i] = (1 << i) - 1;
  }
map<int,int>diff_first;
map<int,int>diff_tail;
auto count=[&](int flag){
//  auto num = _mm_popcnt_u32(_mm512_kandn(m_no_conflict,m_to_insert));
//  auto num = _mm_popcnt_u32((m_no_conflict));
  auto num = _mm_popcnt_u32((m_to_insert))-_mm_popcnt_u32((m_no_conflict));
  if(flag){
    diff_first[num]++;
  }else{
    diff_tail[num]++;
  }
};
imvNum1 = imvNum;
  while (true) {
    k = (k >= imvNum) ? 0 : k;
    if ((cur >= end)) {
      if (state[k].m_valid_probe == 0 && state[k].stage != 3) {
        ++done;
        state[k].stage = 3;
        ++k;
        continue;
      }
    }
    if (done >= imvNum) {
      if (state[imvNum1].m_valid_probe > 0) {
        k = imvNum1;
      } else {
        if (state[imvNum].m_valid_probe > 0) {
          k = imvNum;
        } else {
          break;
        }
      }
    }
    switch (state[k].stage) {
      case 1: {
        /// step 1: get offsets
        state[k].v_probe_offset = _mm512_add_epi64(_mm512_set1_epi64(cur), v_base_offset);
        cur += VECTORSIZE;
#if 0
        state[k].m_valid_probe = -1;
        if (cur >= end) {
          state[k].m_valid_probe = (state[k].m_valid_probe >> (cur - end));
        }
#else
        state[k].m_valid_probe = _mm512_cmpgt_epu64_mask(v_base_offset_upper, state[k].v_probe_offset);
#endif
        if (nullptr == entry_addrs) {
          /// step 2: gather probe keys and values
#if SEQ_PREFETCH
          _mm_prefetch((((char* )(l_orderkey+cur))+PDIS), _MM_HINT_T0);
          _mm_prefetch((((char* )(l_orderkey+cur))+PDIS+64), _MM_HINT_T0);
          _mm_prefetch((((char* )(l_discount+cur))+PDIS), _MM_HINT_T0);
          _mm_prefetch((((char* )(l_discount+cur))+PDIS+64), _MM_HINT_T0);
#endif
          v256_probe_keys = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid_probe, state[k].v_probe_offset, (const int* )probe_keys, 4);
          state[k].v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
          state[k].v_probe_value = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset, (const long long int* )probe_value, 8);
          /// step 3: compute hash values
          state[k].v_probe_hash = runtime::MurMurHash()((state[k].v_probe_keys), (v_seed));
        } else {
#if SEQ_PREFETCH
          _mm_prefetch((((char* )(entry_addrs[cur+PDISD]))), _MM_HINT_T0);
          _mm_prefetch((((char* )(entry_addrs[cur+PDISD]))+64), _MM_HINT_T0);
#endif
          // gather the addresses of entries
          state[k].v_probe_offset = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset, (const long long int* )entry_addrs, 8);
          v256_probe_keys = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid_probe, state[k].v_probe_offset + u_offset_k, nullptr, 1);
          state[k].v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
          state[k].v_probe_value = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset+u_offset_v, nullptr, 1);
          state[k].v_probe_hash = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset+u_offset_hash, nullptr, 1);
        }
        state[k].stage = 2;
        hash_table->prefetchEntry(state[k].v_probe_hash);
      }
        break;
      case 2: {
        /// step 4: find the addresses of corresponding buckets for new probes
#if 0
        Vec8uM v_new_bucket_addrs = hash_table->find_chain(state[k].v_probe_hash);

        /// insert new nodes in the corresponding hash buckets
        m_to_insert = _mm512_kandn(v_new_bucket_addrs.mask, state[k].m_valid_probe);
        v_hash_mask = ((Vec8u(state[k].v_probe_hash) & Vec8u(hash_table->mask)));
        v_hash_mask = _mm512_mask_blend_epi64(state[k].m_valid_probe, v_all_ones, v_hash_mask);
#else

        v_hash_mask = _mm512_and_epi64(state[k].v_probe_hash,v_ht_mask);
        v_new_addr = _mm512_mask_i64gather_epi64(v_all_ones,state[k].m_valid_probe,v_hash_mask, (const long long int* )hash_table->entries, 8);
        m_match = _mm512_cmpneq_epi64_mask(v_new_addr,v_zero);
        m_to_insert = _mm512_kandn(m_match,state[k].m_valid_probe);
        if(0==m_to_insert){
          state[k].v_bucket_addrs =v_new_addr;
          state[k].stage = 0;
          --k;
          break;
        }
        v_hash_mask = _mm512_mask_blend_epi64(state[k].m_valid_probe, v_all_ones, v_hash_mask);
#endif
        v_conflict = _mm512_conflict_epi64(v_hash_mask);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_no_conflict) {
//          count(1);
          if (nullptr == entry_addrs) {
            insertNewEntry(state[k], u_new_addrs, m_no_conflict, partition, u_offset_hash.reg, u_offset_k.reg, u_offset_v.reg);
            // insert the new addresses to the hash table
            _mm512_mask_i64scatter_epi64((long long int* )hash_table->entries, m_no_conflict, v_hash_mask, u_new_addrs.reg, 8);
          } else {
            // set the next of entries = 0
            _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state[k].v_probe_offset, v_zero, 1);
            _mm512_mask_i64scatter_epi64((long long int* )hash_table->entries, m_no_conflict, v_hash_mask, state[k].v_probe_offset, 8);
            _mm512_mask_compressstoreu_epi64((results_entry + found), m_no_conflict, state[k].v_probe_offset);
          }
          found += _mm_popcnt_u32(m_no_conflict);
          // get rid of no-conflict elements
          state[k].m_valid_probe = _mm512_kandn(m_no_conflict, state[k].m_valid_probe);
        }
        state[k].v_bucket_addrs = _mm512_mask_i64gather_epi64(v_all_ones, state[k].m_valid_probe, v_hash_mask, (const long long int* )hash_table->entries, 8);
#if 0
        v_prefetch(state[k].v_bucket_addrs);
        state[k].stage = 0;
#else
        num = _mm_popcnt_u32(state[k].m_valid_probe);
        if(0==num){
          state[k].stage = 1;
          --k;
        }else
        if (num == VECTORSIZE) {
          v_prefetch(state[k].v_bucket_addrs);
          state[k].stage = 0;
        } else {
          if ((done < imvNum)) {
            num_temp = _mm_popcnt_u32(state[imvNum1].m_valid_probe);
            if (num + num_temp < VECTORSIZE) {
              // compress imv_state[k]
              state[k].compress();
              // expand imv_state[k] -> imv_state[imvNum]
              state[imvNum1].expand(state[k]);
              state[imvNum1].m_valid_probe = mask[num + num_temp];
              state[k].m_valid_probe = 0;
              state[k].stage = 1;
              state[imvNum1].stage = 0;
              --k;
              break;
            } else {
              // expand imv_state[imvNum] -> expand imv_state[k]
              state[k].expand(state[imvNum1]);
              state[imvNum1].m_valid_probe = _mm512_kand(state[imvNum1].m_valid_probe, _mm512_knot(mask[VECTORSIZE - num]));
              // compress imv_state[imvNum]
              state[imvNum1].compress();
              state[imvNum1].m_valid_probe = state[imvNum1].m_valid_probe >> (VECTORSIZE - num);
              state[k].m_valid_probe = mask[VECTORSIZE];
              state[k].stage = 0;
              state[imvNum1].stage = 0;
              v_prefetch(state[k].v_bucket_addrs);
            }
          }
        }
#endif
      }
        break;
      case 0: {
        /// step 5: gather the all new build keys
        v256_ht_keys = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid_probe, _mm512_add_epi64(state[k].v_bucket_addrs, u_offset_k.reg), nullptr, 1);
        v_ht_keys = _mm512_cvtepi32_epi64(v256_ht_keys);
        /// step 6: compare the probe keys and build keys and write points
        m_match = _mm512_cmpeq_epi64_mask(state[k].v_probe_keys, v_ht_keys);
        m_match = _mm512_kand(m_match, state[k].m_valid_probe);
        if(m_match){
        /// update the aggregators
        v_conflict = _mm512_conflict_epi64(state[k].v_bucket_addrs);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_match);
          v_ht_value = _mm512_mask_i64gather_epi64(v_zero, m_no_conflict, _mm512_add_epi64(state[k].v_bucket_addrs, u_offset_v.reg), nullptr, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].v_bucket_addrs, u_offset_v.reg), _mm512_add_epi64(state[k].v_probe_value, v_ht_value), 1);

          state[k].m_valid_probe = _mm512_kandn(m_no_conflict, state[k].m_valid_probe);
          // the remaining matches, DO NOT get next
          m_match = _mm512_kandn(m_no_conflict, m_match);
        }
        /// step 7: NOT found, then insert
        v_next = _mm512_mask_i64gather_epi64(v_all_ones, _mm512_kandn(m_match, state[k].m_valid_probe), state[k].v_bucket_addrs, nullptr, 1);
        m_to_insert = _mm512_kand(_mm512_kandn(m_match, state[k].m_valid_probe), _mm512_cmpeq_epi64_mask(v_next, v_zero));
        // get rid of bucket address of matched probes
        v_next = _mm512_mask_blend_epi64(_mm512_kandn(m_match, state[k].m_valid_probe), v_all_ones, state[k].v_bucket_addrs);
        v_conflict = _mm512_conflict_epi64(v_next);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_no_conflict) {
//         count(0);
          if (nullptr == entry_addrs) {
            insertNewEntry(state[k], u_new_addrs, m_no_conflict, partition, u_offset_hash.reg, u_offset_k.reg, u_offset_v.reg);
            // insert the new addresses to the hash table
            _mm512_mask_i64scatter_epi64(0, m_no_conflict, state[k].v_bucket_addrs, u_new_addrs.reg, 1);
          } else {
            // set the next of entries = 0
            _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state[k].v_probe_offset, v_zero, 1);
            _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state[k].v_bucket_addrs, state[k].v_probe_offset, 1);
            _mm512_mask_compressstoreu_epi64((results_entry + found), m_no_conflict, state[k].v_probe_offset);
          }
          found += _mm_popcnt_u32(m_no_conflict);

          state[k].m_valid_probe = _mm512_kandn(m_no_conflict, state[k].m_valid_probe);
        }
        v_next = _mm512_mask_i64gather_epi64(v_all_ones, state[k].m_valid_probe, state[k].v_bucket_addrs, nullptr, 1);
        // the remaining matches, DO NOT get next
        state[k].v_bucket_addrs = _mm512_mask_blend_epi64(m_match, v_next, state[k].v_bucket_addrs);

        num = _mm_popcnt_u32(state[k].m_valid_probe);
        if (num == VECTORSIZE || done >= imvNum) {
          v_prefetch(state[k].v_bucket_addrs);
        } else  if(0==num){
          state[k].stage = 1;
          --k;
        }else{
          if ((done < imvNum)) {
            num_temp = _mm_popcnt_u32(state[imvNum].m_valid_probe);
            if (num + num_temp < VECTORSIZE) {
              // compress imv_state[k]
              state[k].compress();
              // expand imv_state[k] -> imv_state[imvNum]
              state[imvNum].expand(state[k]);
              state[imvNum].m_valid_probe = mask[num + num_temp];
              state[k].m_valid_probe = 0;
              state[k].stage = 1;
              state[imvNum].stage = 0;
              --k;
              break;
            } else {
              // expand imv_state[imvNum] -> expand imv_state[k]
              state[k].expand(state[imvNum]);
              state[imvNum].m_valid_probe = _mm512_kand(state[imvNum].m_valid_probe, _mm512_knot(mask[VECTORSIZE - num]));
              // compress imv_state[imvNum]
              state[imvNum].compress();
              state[imvNum].m_valid_probe = state[imvNum].m_valid_probe >> (VECTORSIZE - num);
              state[k].m_valid_probe = mask[VECTORSIZE];
              state[k].stage = 0;
              state[imvNum].stage = 0;
              v_prefetch(state[k].v_bucket_addrs);
            }
          }
        }

      }
        break;
    }
    ++k;
  }
//  for(auto iter:diff_first){
//    cout<<"first num = "<<iter.first<<" , times = "<<iter.second<<endl;
//  }
//  for(auto iter:diff_tail){
//    cout<<"tail num = "<<iter.first<<" , times = "<<iter.second<<endl;
//  }
  return found;
}

size_t agg_imv_merged(size_t begin, size_t end, Database& db, Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>* hash_table,
                      PartitionedDeque<PARTITION_SIZE>* partition, void** entry_addrs, void** results_entry) {
  size_t found = 0, pos = 0, cur = begin;
  int k = 0, done = 0, buildkey, probeKey, valid_size, imvNum = vectorwise::Hashjoin::imvNum, imvNum1 = vectorwise::Hashjoin::imvNum + 1;
  AggState state[stateNumSIMD + 2];
  hash_t probeHash;
  auto& li = db["lineitem"];
#if ORDERKEY
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
#else
  auto l_orderkey = li["l_partkey"].data<types::Integer>();
#endif
  auto l_discount = li["l_discount"].data<types::Numeric<12, 2>>();
  using group_t = Hashmapx<types::Integer, types::Numeric<12, 2>, hashFun, false>::Entry;
  __m512i v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), v_zero = _mm512_set1_epi64(0), v_lzeros, v_63 = _mm512_set1_epi64(63);
  __m512i v_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(end - begin), v_seed = _mm512_set1_epi64(vectorwise::primitives::seed), v_all_ones =
      _mm512_set1_epi64(-1), v_conflict, v_ht_keys, v_hash_mask, v_ht_value, v_next, v_previous;
  Vec8u u_new_addrs(uint64_t(0)), u_offset_hash(offsetof(group_t, h.hash)), u_offset_k(offsetof(group_t, k)), u_offset_v(offsetof(group_t, v));
  __mmask8 m_no_conflict, m_rest, m_match, m_to_insert, mask[VECTORSIZE + 1], to_scatt;
  void* probe_keys = (void*) l_orderkey, *probe_value = (void*) l_discount;
  __m256i v256_zero = _mm256_set1_epi32(0), v256_all_one = _mm256_set1_epi32(-1), v256_probe_keys, v256_probe_value, v256_ht_keys;
  uint8_t num, num_temp;
  for (int i = 0; i <= VECTORSIZE; ++i) {
    mask[i] = (1 << i) - 1;
  }

  while (true) {
    k = (k >= imvNum) ? 0 : k;
    if ((cur >= end)) {
      if (state[k].m_valid_probe == 0 && state[k].stage != 3) {
        ++done;
        state[k].stage = 3;
        ++k;
        continue;
      }
    }
    if (done >= imvNum) {
      if (state[imvNum1].m_valid_probe > 0) {
        k = imvNum1;
        mergeKeys(state[k]);
      } else {
        if (state[imvNum].m_valid_probe > 0) {
          k = imvNum;
          mergeKeys(state[k]);
        } else {
          break;
        }
      }
    }
    switch (state[k].stage) {
      case 1: {
        /// step 1: get offsets
        state[k].v_probe_offset = _mm512_add_epi64(_mm512_set1_epi64(cur), v_base_offset);
        cur += VECTORSIZE;
        state[k].m_valid_probe = -1;
        if (cur >= end) {
          state[k].m_valid_probe = (state[k].m_valid_probe >> (cur - end));
        }
        if (nullptr == entry_addrs) {
          /// step 2: gather probe keys and values
#if SEQ_PREFETCH
          _mm_prefetch((((char* )(l_orderkey+cur))+PDIS), _MM_HINT_T0);
          _mm_prefetch((((char* )(l_orderkey+cur))+PDIS+64), _MM_HINT_T0);
          _mm_prefetch((((char* )(l_discount+cur))+PDIS), _MM_HINT_T0);
          _mm_prefetch((((char* )(l_discount+cur))+PDIS+64), _MM_HINT_T0);
#endif
          v256_probe_keys = _mm512_mask_i64gather_epi32(v256_all_one, state[k].m_valid_probe, state[k].v_probe_offset, (const int* )probe_keys, 4);
          state[k].v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
          state[k].v_probe_value = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset, (const long long int* )probe_value, 8);
          /// step 3: compute hash values
          state[k].v_probe_hash = runtime::MurMurHash()((state[k].v_probe_keys), (v_seed));
        } else {
#if SEQ_PREFETCH
          _mm_prefetch((((char* )(entry_addrs[cur+PDISD]))), _MM_HINT_T0);
          _mm_prefetch((((char* )(entry_addrs[cur+PDISD]))+64), _MM_HINT_T0);
#endif
          // gather the addresses of entries
          state[k].v_probe_offset = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset, (const long long int* )entry_addrs, 8);
          v256_probe_keys = _mm512_mask_i64gather_epi32(v256_all_one, state[k].m_valid_probe, state[k].v_probe_offset + u_offset_k, nullptr, 1);
          state[k].v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
          state[k].v_probe_value = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset+u_offset_v, nullptr, 1);
          state[k].v_probe_hash = _mm512_mask_i64gather_epi64(v_zero, state[k].m_valid_probe, state[k].v_probe_offset+u_offset_hash, nullptr, 1);
        }

        state[k].stage = 2;
        hash_table->prefetchEntry(state[k].v_probe_hash);
      }
        break;
      case 2: {

        mergeKeys(state[k]);
        /// step 4: find the addresses of corresponding buckets for new probes
        Vec8uM v_new_bucket_addrs = hash_table->find_chain(state[k].v_probe_hash);

        /// insert new nodes in the corresponding hash buckets
        m_to_insert = _mm512_kandn(v_new_bucket_addrs.mask, state[k].m_valid_probe);
        v_hash_mask = ((Vec8u(state[k].v_probe_hash) & Vec8u(hash_table->mask)));
        v_hash_mask = _mm512_mask_blend_epi64(state[k].m_valid_probe, v_all_ones, v_hash_mask);
        v_conflict = _mm512_conflict_epi64(v_hash_mask);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_to_insert) {
          if (nullptr == entry_addrs) {
            insertAllNewEntry(state[k], u_new_addrs, v_conflict, m_to_insert, m_no_conflict, partition, u_offset_hash.reg, u_offset_k.reg, u_offset_v.reg);
            // insert the new addresses to the hash table
            _mm512_mask_i64scatter_epi64((long long int* )hash_table->entries, m_no_conflict, v_hash_mask, u_new_addrs.reg, 8);
          } else {
            // set the next of entries = 0
            _mm512_mask_i64scatter_epi64(nullptr, m_to_insert, state[k].v_probe_offset, v_zero, 1);
            // must write back the merged values!!!!!!!
            _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state[k].v_probe_offset + u_offset_v, state[k].v_probe_value, 1);
            // write the addresses to the previous'next
            v_lzeros = _mm512_lzcnt_epi64(v_conflict);
            v_lzeros = _mm512_sub_epi64(v_63, v_lzeros);
            v_previous = _mm512_maskz_permutexvar_epi64(_mm512_kandn(m_no_conflict, m_to_insert), v_lzeros, state[k].v_probe_offset);
            _mm512_mask_i64scatter_epi64(0, _mm512_kandn(m_no_conflict, m_to_insert), v_previous, state[k].v_probe_offset, 1);
            _mm512_mask_i64scatter_epi64((long long int* )hash_table->entries, m_no_conflict, v_hash_mask, state[k].v_probe_offset, 8);
            _mm512_mask_compressstoreu_epi64((results_entry + found), m_to_insert, state[k].v_probe_offset);
          }
          found += _mm_popcnt_u32(m_to_insert);
          // get rid of no-conflict elements
          state[k].m_valid_probe = _mm512_kandn(m_to_insert, state[k].m_valid_probe);
        }
        state[k].v_bucket_addrs = _mm512_mask_i64gather_epi64(v_all_ones, state[k].m_valid_probe, v_hash_mask, (const long long int* )hash_table->entries, 8);
        state[k].v_probe_keys = _mm512_mask_blend_epi64(state[k].m_valid_probe, v_all_ones, state[k].v_probe_keys);

#if 0
        v_prefetch(state[k].v_bucket_addrs);
        state[k].stage = 0;
#else
        num = _mm_popcnt_u32(state[k].m_valid_probe);
        if (num == VECTORSIZE) {
          v_prefetch(state[k].v_bucket_addrs);
          state[k].stage = 0;
        } else {
          if ((done < imvNum)) {
            num_temp = _mm_popcnt_u32(state[imvNum1].m_valid_probe);
            if (num + num_temp < VECTORSIZE) {
              // compress imv_state[k]
              state[k].compress();
              // expand imv_state[k] -> imv_state[imvNum]
              state[imvNum1].expand(state[k]);
              state[imvNum1].m_valid_probe = mask[num + num_temp];
              state[k].m_valid_probe = 0;
              state[k].stage = 1;
              state[imvNum1].stage = 0;
              --k;
              break;
            } else {
              // expand imv_state[imvNum] -> expand imv_state[k]
              state[k].expand(state[imvNum1]);
              state[imvNum1].m_valid_probe = _mm512_kand(state[imvNum1].m_valid_probe, _mm512_knot(mask[VECTORSIZE - num]));
              // compress imv_state[imvNum]
              state[imvNum1].compress();
              state[imvNum1].m_valid_probe = state[imvNum1].m_valid_probe >> (VECTORSIZE - num);
              state[k].m_valid_probe = mask[VECTORSIZE];
              state[k].stage = 0;
              state[imvNum1].stage = 0;
              mergeKeys(state[k]);
              v_prefetch(state[k].v_bucket_addrs);
            }
          }
        }
#endif
      }
        break;
      case 0: {
        /// step 5: gather the all new build keys
        v256_ht_keys = _mm512_mask_i64gather_epi32(v256_zero, state[k].m_valid_probe, _mm512_add_epi64(state[k].v_bucket_addrs, u_offset_k.reg), nullptr, 1);
        v_ht_keys = _mm512_cvtepi32_epi64(v256_ht_keys);
        /// step 6: compare the probe keys and build keys and write points
        m_match = _mm512_cmpeq_epi64_mask(state[k].v_probe_keys, v_ht_keys);
        m_match = _mm512_kand(m_match, state[k].m_valid_probe);
        if(m_match){
        v_ht_value = _mm512_mask_i64gather_epi64(v_zero, m_match, _mm512_add_epi64(state[k].v_bucket_addrs, u_offset_v.reg), nullptr, 1);
        _mm512_mask_i64scatter_epi64(0, m_match, _mm512_add_epi64(state[k].v_bucket_addrs, u_offset_v.reg), _mm512_add_epi64(state[k].v_probe_value, v_ht_value), 1);

        state[k].m_valid_probe = _mm512_kandn(m_match, state[k].m_valid_probe);
        }
        /// step 7: NOT found, then insert
        v_next = _mm512_mask_i64gather_epi64(v_all_ones, state[k].m_valid_probe, state[k].v_bucket_addrs, nullptr, 1);
        m_to_insert = _mm512_kand(state[k].m_valid_probe, _mm512_cmpeq_epi64_mask(v_next, v_zero));
        // get rid of bucket address of matched probes
        v_next = _mm512_mask_blend_epi64(state[k].m_valid_probe, v_all_ones, state[k].v_bucket_addrs);
        v_conflict = _mm512_conflict_epi64(v_next);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_to_insert) {
          if (nullptr == entry_addrs) {
            insertAllNewEntry(state[k], u_new_addrs, v_conflict, m_to_insert, m_no_conflict, partition, u_offset_hash.reg, u_offset_k.reg, u_offset_v.reg);
            // insert the new addresses to the hash table
            _mm512_mask_i64scatter_epi64(0, m_no_conflict, state[k].v_bucket_addrs, u_new_addrs.reg, 1);
          } else {
            // set the next of entries = 0
            _mm512_mask_i64scatter_epi64(nullptr, m_to_insert, state[k].v_probe_offset, v_zero, 1);
            // must write back the merged values!!!!!!!
            _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state[k].v_probe_offset + u_offset_v, state[k].v_probe_value, 1);
            // write the addresses to the previous'next
            v_lzeros = _mm512_lzcnt_epi64(v_conflict);
            v_lzeros = _mm512_sub_epi64(v_63, v_lzeros);
            v_previous = _mm512_maskz_permutexvar_epi64(_mm512_kandn(m_no_conflict, m_to_insert), v_lzeros, state[k].v_probe_offset);
            _mm512_mask_i64scatter_epi64(0, _mm512_kandn(m_no_conflict, m_to_insert), v_previous, state[k].v_probe_offset, 1);
            _mm512_mask_i64scatter_epi64(nullptr, m_no_conflict, state[k].v_bucket_addrs, state[k].v_probe_offset, 1);
            _mm512_mask_compressstoreu_epi64((results_entry + found), m_to_insert, state[k].v_probe_offset);
          }
          found += _mm_popcnt_u32(m_to_insert);

          state[k].m_valid_probe = _mm512_kandn(m_to_insert, state[k].m_valid_probe);
        }
        state[k].v_probe_keys = _mm512_mask_blend_epi64(state[k].m_valid_probe, v_all_ones, state[k].v_probe_keys);
        state[k].v_bucket_addrs = _mm512_mask_i64gather_epi64(v_all_ones, state[k].m_valid_probe, state[k].v_bucket_addrs, nullptr, 1);
#if 01
        num = _mm_popcnt_u32(state[k].m_valid_probe);
        if (num == VECTORSIZE) {
          v_prefetch(state[k].v_bucket_addrs);
        } else {
          if ((done < imvNum)) {
            num_temp = _mm_popcnt_u32(state[imvNum].m_valid_probe);
            if (num + num_temp < VECTORSIZE) {
              // compress imv_state[k]
              state[k].compress();
              // expand imv_state[k] -> imv_state[imvNum]
              state[imvNum].expand(state[k]);
              state[imvNum].m_valid_probe = mask[num + num_temp];
              state[k].m_valid_probe = 0;
              state[k].stage = 1;
              state[imvNum].stage = 0;
              --k;
              break;
            } else {
              // expand imv_state[imvNum] -> expand imv_state[k]
              state[k].expand(state[imvNum]);
              state[imvNum].m_valid_probe = _mm512_kand(state[imvNum].m_valid_probe, _mm512_knot(mask[VECTORSIZE - num]));
              // compress imv_state[imvNum]
              state[imvNum].compress();
              state[imvNum].m_valid_probe = state[imvNum].m_valid_probe >> (VECTORSIZE - num);
              state[k].m_valid_probe = mask[VECTORSIZE];
              state[k].stage = 0;
              state[imvNum].stage = 0;
              mergeKeys(state[k]);
              v_prefetch(state[k].v_bucket_addrs);
            }
          } else {
            mergeKeys(state[k]);
            v_prefetch(state[k].v_bucket_addrs);
          }
        }
#else
        if(state[k].m_valid_probe==0) {
          state[k].stage = 1;
        }
#endif
      }
        break;
    }
    ++k;
  }

  return found;
}
