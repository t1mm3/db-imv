#include "imv/Pipeline.hpp"
auto constrant_o_orderdate1 = types::Date::castString("1996-01-01");
types::Numeric<12, 2> constrant_l_quantity1 = types::Numeric<12, 2>(types::Integer(CONSTRANT_L_QUAN));
uint64_t constrants = constrant_l_quantity1.value;
size_t scan_filter_simd(types::Numeric<12, 2>* col, size_t& begin, size_t end, int constrants, uint64_t* pos_buff) {
  size_t found = 0;
  __mmask8 m_valid = -1, m_eval;
  __m512i v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), v_keys, v_const = _mm512_set1_epi64(constrants), v_offset;
  for (size_t & cur = begin; cur < end;) {
    v_offset = _mm512_add_epi64(_mm512_set1_epi64(cur), v_base_offset);
    v_keys = _mm512_maskz_loadu_epi64(m_valid, (col + cur));
    cur += VECTORSIZE;
    if (cur >= end) {
      m_valid = (m_valid >> (cur - end));
    }
    m_eval = _mm512_cmpgt_epu64_mask(v_const, v_keys);
    m_eval = _mm512_kand(m_eval, m_valid);
    _mm512_mask_compressstoreu_epi64(pos_buff + found, m_eval, v_offset);
    found += _mm_popcnt_u32((m_eval));
    if (found + VECTORSIZE >= ROF_VECTOR_SIZE) {
      return found;
    }
  }
  return found;
}
size_t scan_filter_scalar(types::Numeric<12, 2>* col, size_t& begin, size_t end, int constrants, uint64_t* pos_buff) {
  size_t found = 0;
  for (size_t & cur = begin; cur < end; ++cur) {
#if 1
    if (constrants > col[cur].value) {
      pos_buff[found++] = cur;
    }
#else
    pos_buff[found] =cur;
    found += (constrants > col[cur].value);
#endif
  }
  return found;
}
size_t filter_probe_simd_amac(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, void** output_build, uint32_t*output_probe, uint64_t* pos_buff) {
  size_t found = 0;
  auto& li = db["lineitem"];
  auto l_quantity_col = li["l_quantity"].data<types::Numeric<12, 2>>();
#if PIPELINE_ORDERED
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
#else
  auto l_orderkey = li["l_partkey"].data<types::Integer>();
#endif
#if 0
  for (size_t i = begin, size = 0; i < end; ) {
    size = i + ROF_VECTOR_SIZE < end ? ROF_VECTOR_SIZE : end -i;
    size = scan_filter_simd(l_quantity_col,i,i+size,constrants,pos_buff);
    found += probe_amac(l_orderkey, size, hash_table, output_build, output_probe, pos_buff);
  }
#else
  for (size_t i = begin, size = 0; i < end;) {
    size = scan_filter_simd(l_quantity_col, i, end, constrants, pos_buff);
    found += probe_amac(l_orderkey, size, hash_table, output_build, output_probe, pos_buff);
  }
#endif

  return found;
}
size_t filter_probe_scalar(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, void** output_build, uint32_t*output_probe, uint64_t* pos_buff) {
  size_t found = 0, pos = 0;
  auto& li = db["lineitem"];
  auto l_quantity_col = li["l_quantity"].data<types::Numeric<12, 2>>();
#if PIPELINE_ORDERED
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
#else
  auto l_orderkey = li["l_partkey"].data<types::Integer>();
#endif
  int build_key_off = sizeof(runtime::Hashmap::EntryHeader);
  for (size_t i = begin, size = 0; i < end; ++i) {
    if (constrants > l_quantity_col[i].value) {
      auto probeKey = l_orderkey[i].value;
      auto probeHash = runtime::MurMurHash()(probeKey, vectorwise::primitives::seed);
      auto buildMatch = hash_table->find_chain_tagged(probeHash);

      for (auto entry = buildMatch; entry != nullptr; entry = entry->next) {
        uint32_t buildkey = *((uint32_t*) (((void*) entry) + build_key_off));
        if ((buildkey == probeKey)) {
          pos = pos < morselSize ? pos : 0;
          output_build[pos] = ((void*) entry);
          output_probe[pos] = (i);
          ++found;
          ++pos;
#if EARLY_BREAK
          break;
#endif
        }
      }
    }
  }
  return found;
}
size_t filter_probe_simd_gp(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, void** output_build, uint32_t*output_probe, uint64_t* pos_buff) {
  size_t found = 0;
  auto& li = db["lineitem"];
  auto l_quantity_col = li["l_quantity"].data<types::Numeric<12, 2>>();
#if PIPELINE_ORDERED
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
#else
  auto l_orderkey = li["l_partkey"].data<types::Integer>();
#endif

  for (size_t i = begin, size = 0; i < end;) {
    size = scan_filter_simd(l_quantity_col, i, end, constrants, pos_buff);
    found += probe_gp(l_orderkey, size, hash_table, output_build, output_probe, pos_buff);
  }
  return found;
}
size_t filter_probe_simd_imv(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, void** output_build, uint32_t*output_probe, uint64_t* pos_buff) {
  size_t found = 0;
  auto& li = db["lineitem"];
  auto l_quantity_col = li["l_quantity"].data<types::Numeric<12, 2>>();
#if PIPELINE_ORDERED
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
#else
  auto l_orderkey = li["l_partkey"].data<types::Integer>();
#endif
#if 0
  for (size_t i = begin, size = 0; i < end; ) {
    size = i + ROF_VECTOR_SIZE < end ? ROF_VECTOR_SIZE : end -i;
    size = scan_filter_simd(l_quantity_col,i,i+size,constrants,pos_buff);
    found += probe_imv(l_orderkey, size, hash_table, output_build, output_probe, pos_buff);
  }
#elif 1
  for (size_t i = begin, size = 0; i < end;) {
    size = scan_filter_simd(l_quantity_col, i, end, constrants, pos_buff);
    found += probe_imv(l_orderkey, size, hash_table, output_build, output_probe, pos_buff);
  }
#else
  found += probe_imv(l_orderkey+begin, end -begin, hash_table, output_build, output_probe, nullptr);

#endif
  return found;
}
size_t filter_probe_imv1(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, void** output_build, uint32_t*output_probe, uint64_t* pos_buff) {
  auto& li = db["lineitem"];
  auto l_quantity_col = li["l_quantity"].data<types::Numeric<12, 2>>();
#if PIPELINE_ORDERED
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
#else
  auto l_orderkey = li["l_partkey"].data<types::Integer>();
#endif

  ///////////////////////////////
  auto probe_keys = l_orderkey;
  size_t found = 0, pos = 0;
  int k = 0, done = 0, keyOff = sizeof(runtime::Hashmap::EntryHeader), imvNum = vectorwise::Hashjoin::imvNum, imvNum1 = vectorwise::Hashjoin::imvNum + 1, nextProbe = begin,
      curProbe;
  // extra 2 for residual vectorized states
  IMVState* imv_state = new IMVState[vectorwise::Hashjoin::imvNum + 2];

  __attribute__((aligned(64)))         __mmask8 m_match = 0, m_new_probes = -1, mask[VECTORSIZE + 1];

  __m512i v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
  __m512i v_offset = _mm512_set1_epi64(0), v_new_build_key, v_build_keys;
  __m512i v_base_offset_upper = _mm512_set1_epi64(end);
  __m512i v_seed = _mm512_set1_epi64(vectorwise::primitives::seed), v_build_key_off = _mm512_set1_epi64(keyOff);
  __m512i v_zero = _mm512_set1_epi64(0), v_const = _mm512_set1_epi64(constrants);
  __m256i v256_zero = _mm256_set1_epi32(0), v256_probe_keys, v256_build_keys;
  uint64_t * ht_pos = nullptr;
  uint8_t num, num_temp;
  for (int i = 0; i <= VECTORSIZE; ++i) {
    mask[i] = (1 << i) - 1;
  }
  while (true) {
    k = (k >= imvNum) ? 0 : k;
    if ((nextProbe >= end)) {
      if (imv_state[k].m_valid_probe == 0 && imv_state[k].stage != 3) {
        ++done;
        imv_state[k].stage = 3;
        ++k;
        continue;
      }
    }
    if (done >= imvNum) {
      if (imv_state[imvNum].m_valid_probe > 0) {
        k = imvNum;
        imv_state[imvNum].stage = 0;
      } else {
        break;
      }
    }
    switch (imv_state[k].stage) {
      case 1: {
#if 0
        /// step 1: load the offsets of probing tuples
        imv_state[k].v_probe_offset = _mm512_add_epi64(_mm512_set1_epi64(nextProbe), v_base_offset);
        imv_state[k].m_valid_probe = _mm512_cmpgt_epu64_mask(v_base_offset_upper, imv_state[k].v_probe_offset);
        /// step 2: gather the probe keys // why is load() so faster than gather()?
        v256_probe_keys = _mm512_mask_i64gather_epi32(v256_zero, imv_state[k].m_valid_probe, imv_state[k].v_probe_offset, (void* )probe_keys, 4);
        // v256_probe_keys = _mm256_maskz_loadu_epi32(imv_state[k].m_valid_probe, (char*)(probe_keys+nextProbe));
        imv_state[k].v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
        /// step 2.5: filter
        imv_state[k].v_probe_hash = _mm512_maskz_loadu_epi64(imv_state[k].m_valid_probe, (l_quantity_col+nextProbe));
        m_match = _mm512_cmpgt_epu64_mask(v_const, imv_state[k].v_probe_hash);
        imv_state[k].m_valid_probe = _mm512_kand(imv_state[k].m_valid_probe ,m_match);
        /// step 3: compute the hash values of probe keys
        imv_state[k].v_probe_hash = runtime::MurMurHash()((imv_state[k].v_probe_keys), (v_seed));
        nextProbe += VECTORSIZE;
        imv_state[k].stage = 2;
#if SEQ_PREFETCH
        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS+64), _MM_HINT_T0);
        _mm_prefetch((((char* )(l_quantity_col+nextProbe))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(l_quantity_col+nextProbe))+PDIS+64), _MM_HINT_T0);
#endif
        hash_table->prefetchEntry((imv_state[k].v_probe_hash));

#else
        /// step 1: load the offsets of probing tuples
        v_offset = _mm512_add_epi64(_mm512_set1_epi64(nextProbe), v_base_offset);
        imv_state[k].v_probe_offset = _mm512_mask_expand_epi64(imv_state[k].v_probe_offset, _mm512_knot(imv_state[k].m_valid_probe), v_offset);
        // count the number of empty tuples
        m_new_probes = _mm512_knot(imv_state[k].m_valid_probe);
        nextProbe = nextProbe + _mm_popcnt_u32(m_new_probes);
        imv_state[k].m_valid_probe = _mm512_cmpgt_epu64_mask(v_base_offset_upper, imv_state[k].v_probe_offset);
        m_new_probes = _mm512_kand(m_new_probes, imv_state[k].m_valid_probe);
        /// step 2: gather the probe keys
        v256_probe_keys = _mm512_mask_i64gather_epi32(v256_zero, imv_state[k].m_valid_probe, imv_state[k].v_probe_offset, (void* )probe_keys, 4);
        imv_state[k].v_probe_keys = _mm512_mask_blend_epi64(imv_state[k].m_valid_probe, imv_state[k].v_probe_keys, _mm512_cvtepi32_epi64(v256_probe_keys));
#if SEQ_PREFETCH

        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS+64), _MM_HINT_T0);
        _mm_prefetch((((char* )(l_quantity_col+nextProbe))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(l_quantity_col+nextProbe))+PDIS+64), _MM_HINT_T0);

#endif

        /// step 2.5: filter
        imv_state[k].v_probe_hash = _mm512_mask_i64gather_epi64(imv_state[k].v_probe_hash, m_new_probes, imv_state[k].v_probe_offset, (void* )l_quantity_col, 8);
        m_match = _mm512_cmpgt_epu64_mask(v_const, imv_state[k].v_probe_hash);
        if (m_match == imv_state[k].m_valid_probe) {

          /// step 3: compute the hash values of probe keys
          imv_state[k].v_probe_hash = runtime::MurMurHash()((imv_state[k].v_probe_keys), (v_seed));
          imv_state[k].stage = 2;
          hash_table->prefetchEntry((imv_state[k].v_probe_hash));

        } else {
          imv_state[k].m_valid_probe = _mm512_kand(imv_state[k].m_valid_probe, m_match);
          //  --k;
        }
#endif
      }
        break;
      case 2: {
        /// step 4: find the addresses of corresponding buckets for new probes
        Vec8uM v_new_bucket_addrs = hash_table->find_chain_tagged((imv_state[k].v_probe_hash));
        imv_state[k].m_valid_probe = _mm512_kand(imv_state[k].m_valid_probe, v_new_bucket_addrs.mask);
        imv_state[k].v_bucket_addrs = v_new_bucket_addrs.vec;
        imv_state[k].stage = 0;
        v_prefetch(imv_state[k].v_bucket_addrs);

      }
        break;
      case 0: {
        /// step 5: gather the all new build keys
        v256_build_keys = _mm512_mask_i64gather_epi32(v256_zero, imv_state[k].m_valid_probe, _mm512_add_epi64(imv_state[k].v_bucket_addrs, v_build_key_off), nullptr, 1);
        v_build_keys = _mm512_cvtepi32_epi64(v256_build_keys);
        /// step 6: compare the probe keys and build keys and write points
        m_match = _mm512_cmpeq_epi64_mask(imv_state[k].v_probe_keys, v_build_keys);
        m_match = _mm512_kand(m_match, imv_state[k].m_valid_probe);
#if WRITE_RESULTS
        pos = pos + VECTORSIZE < morselSize ? pos : 0;
#if WRITE_SEQ_PREFETCH
        _mm_prefetch((char *)(((char *)(output_build+pos)) + WRITE_PDIS), _MM_HINT_T0);
        _mm_prefetch((char *)(((char *)(output_build+pos)) + WRITE_PDIS + 64), _MM_HINT_T0);
        _mm_prefetch((char *)(((char *)(output_probe+pos)) + WRITE_PDIS), _MM_HINT_T0);
        _mm_prefetch((char *)(((char *)(output_probe+pos)) + WRITE_PDIS + 64), _MM_HINT_T0);
#endif
        _mm512_mask_compressstoreu_epi64((output_build + pos), m_match, imv_state[k].v_bucket_addrs);
        _mm256_mask_compressstoreu_epi32((output_probe + pos), m_match, _mm512_cvtepi64_epi32(imv_state[k].v_probe_offset));
        pos += _mm_popcnt_u32(m_match);
#endif
        found += _mm_popcnt_u32(m_match);
#if EARLY_BREAK
        imv_state[k].m_valid_probe = _mm512_kandn(m_match, imv_state[k].m_valid_probe);
#endif
        /// step 7: move to the next bucket nodes
        imv_state[k].v_bucket_addrs = _mm512_mask_i64gather_epi64(v_zero, imv_state[k].m_valid_probe, imv_state[k].v_bucket_addrs, nullptr, 1);
        imv_state[k].m_valid_probe = _mm512_kand(imv_state[k].m_valid_probe, _mm512_cmpneq_epi64_mask(imv_state[k].v_bucket_addrs, v_zero));

        num = _mm_popcnt_u32(imv_state[k].m_valid_probe);
        if (num == VECTORSIZE) {
          v_prefetch(imv_state[k].v_bucket_addrs);

        } else {
          if ((done < imvNum)) {
            num_temp = _mm_popcnt_u32(imv_state[imvNum].m_valid_probe);
            if (num + num_temp < VECTORSIZE) {
              // compress imv_state[k]
              compress(&imv_state[k]);
              // expand imv_state[k] -> imv_state[imvNum]
              expand(&imv_state[k], &imv_state[imvNum]);
              imv_state[imvNum].m_valid_probe = mask[num + num_temp];
              imv_state[k].m_valid_probe = 0;
              imv_state[k].stage = 1;
            } else {
              // expand imv_state[imvNum] -> expand imv_state[k]
              expand(&imv_state[imvNum], &imv_state[k]);
              imv_state[imvNum].m_valid_probe = _mm512_kand(imv_state[imvNum].m_valid_probe, _mm512_knot(mask[VECTORSIZE - num]));
              // compress imv_state[imvNum]
              compress(&imv_state[imvNum]);
              imv_state[imvNum].m_valid_probe = imv_state[imvNum].m_valid_probe >> (VECTORSIZE - num);
              imv_state[k].m_valid_probe = mask[VECTORSIZE];
              imv_state[k].stage = 0;
              v_prefetch(imv_state[k].v_bucket_addrs);

            }
          }
        }
      }
        break;
    }
    ++k;
  }
  delete[] imv_state;
  imv_state = nullptr;
  return found;
}
size_t filter_probe_imv(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, void** output_build, uint32_t*output_probe, uint64_t* pos_buff) {
  auto& li = db["lineitem"];
  auto l_quantity_col = li["l_quantity"].data<types::Numeric<12, 2>>();
#if PIPELINE_ORDERED
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
#else
  auto l_orderkey = li["l_partkey"].data<types::Integer>();
#endif

  ///////////////////////////////
  auto probe_keys = l_orderkey;
  size_t found = 0, pos = 0;
  int k = 0, done = 0, keyOff = sizeof(runtime::Hashmap::EntryHeader), imvNum = vectorwise::Hashjoin::imvNum, imvNum1 = vectorwise::Hashjoin::imvNum + 1, nextProbe = begin,
      curProbe;
  IMVState* imv_state = new IMVState[vectorwise::Hashjoin::imvNum + 2];

  __attribute__((aligned(64)))         __mmask8 m_match = 0, m_new_probes = -1, mask[VECTORSIZE + 1];

  __m512i v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
  __m512i v_offset = _mm512_set1_epi64(0), v_new_build_key, v_build_keys;
  __m512i v_base_offset_upper = _mm512_set1_epi64(end);
  __m512i v_seed = _mm512_set1_epi64(vectorwise::primitives::seed), v_build_key_off = _mm512_set1_epi64(keyOff);
  __m512i v_zero = _mm512_set1_epi64(0), v_const = _mm512_set1_epi64(constrants);
  __m256i v256_zero = _mm256_set1_epi32(0), v256_probe_keys, v256_build_keys;
  uint64_t * ht_pos = nullptr;
  uint8_t num, num_temp;
  for (int i = 0; i <= VECTORSIZE; ++i) {
    mask[i] = (1 << i) - 1;
  }
  while (true) {
    k = (k >= imvNum) ? 0 : k;
    if ((nextProbe >= end)) {
      if (imv_state[k].m_valid_probe == 0 && imv_state[k].stage != 3) {
        ++done;
        imv_state[k].stage = 3;
        ++k;
        continue;
      }
    }
    if (done >= imvNum) {
      if (imv_state[imvNum1].m_valid_probe > 0) {
        k = imvNum1;
      } else {
        if (imv_state[imvNum].m_valid_probe > 0) {
          k = imvNum;
        } else {
          break;
        }
      }
    }
    switch (imv_state[k].stage) {
      case 1: {
#if 0
        /// step 1: load the offsets of probing tuples
        imv_state[k].v_probe_offset = _mm512_add_epi64(_mm512_set1_epi64(nextProbe), v_base_offset);
        imv_state[k].m_valid_probe = _mm512_cmpgt_epu64_mask(v_base_offset_upper, imv_state[k].v_probe_offset);
        /// step 2: gather the probe keys // why is load() so faster than gather()?
        v256_probe_keys = _mm512_mask_i64gather_epi32(v256_zero, imv_state[k].m_valid_probe, imv_state[k].v_probe_offset, (void* )probe_keys, 4);
        // v256_probe_keys = _mm256_maskz_loadu_epi32(imv_state[k].m_valid_probe, (char*)(probe_keys+nextProbe));
        imv_state[k].v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
        /// step 2.5: filter
        imv_state[k].v_probe_hash = _mm512_maskz_loadu_epi64(imv_state[k].m_valid_probe, (l_quantity_col+nextProbe));
        m_match = _mm512_cmpgt_epu64_mask(v_const, imv_state[k].v_probe_hash);
        imv_state[k].m_valid_probe = _mm512_kand(imv_state[k].m_valid_probe ,m_match);
        /// step 3: compute the hash values of probe keys
        imv_state[k].v_probe_hash = runtime::MurMurHash()((imv_state[k].v_probe_keys), (v_seed));
        nextProbe += VECTORSIZE;
        imv_state[k].stage = 2;
#if SEQ_PREFETCH
        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS+64), _MM_HINT_T0);
        _mm_prefetch((((char* )(l_quantity_col+nextProbe))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(l_quantity_col+nextProbe))+PDIS+64), _MM_HINT_T0);

#endif
        hash_table->prefetchEntry((imv_state[k].v_probe_hash));
#else
#if SEQ_PREFETCH

        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS+64), _MM_HINT_T0);
        _mm_prefetch((((char* )(l_quantity_col+nextProbe))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(l_quantity_col+nextProbe))+PDIS+64), _MM_HINT_T0);

#endif
        /// step 1: load the offsets of probing tuples
        imv_state[k].v_probe_offset = _mm512_add_epi64(_mm512_set1_epi64(nextProbe), v_base_offset);
        imv_state[k].m_valid_probe = _mm512_cmpgt_epu64_mask(v_base_offset_upper, imv_state[k].v_probe_offset);
        /// step 2: gather the probe keys // why is load() so faster than gather()?
        // v256_probe_keys = _mm512_mask_i64gather_epi32(v256_zero, imv_state[k].m_valid_probe, imv_state[k].v_probe_offset, (void* )probe_keys, 4);
        v256_probe_keys = _mm256_maskz_loadu_epi32(imv_state[k].m_valid_probe, (char*) (probe_keys + nextProbe));
        imv_state[k].v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
        /// step 2.5: filter
        imv_state[k].v_probe_hash = _mm512_maskz_loadu_epi64(imv_state[k].m_valid_probe, (l_quantity_col + nextProbe));
        m_match = _mm512_cmpgt_epu64_mask(v_const, imv_state[k].v_probe_hash);
        imv_state[k].m_valid_probe = _mm512_kand(imv_state[k].m_valid_probe, m_match);
        nextProbe += VECTORSIZE;
        num = _mm_popcnt_u32(imv_state[k].m_valid_probe);
        if (num == VECTORSIZE || done >= imvNum) {
          imv_state[k].stage = 4;
        } else if (num == 0) {
          --k;
          break;
        } else {
          num_temp = _mm_popcnt_u32(imv_state[imvNum1].m_valid_probe);
          if (num + num_temp < VECTORSIZE) {
            // compress imv_state[k]
            compress(&imv_state[k]);
            // expand imv_state[k] -> imv_state[imvNum1]
            expand(&imv_state[k], &imv_state[imvNum1]);
            imv_state[imvNum1].m_valid_probe = mask[num + num_temp];
            imv_state[k].m_valid_probe = 0;
            imv_state[k].stage = 1;
            imv_state[imvNum1].stage = 4;

          } else {
            // expand imv_state[imvNum1] -> expand imv_state[k]
            expand(&imv_state[imvNum1], &imv_state[k]);
            imv_state[imvNum1].m_valid_probe = _mm512_kand(imv_state[imvNum1].m_valid_probe, _mm512_knot(mask[VECTORSIZE - num]));
            // compress imv_state[imvNum]
            compress(&imv_state[imvNum1]);
            imv_state[imvNum1].m_valid_probe = imv_state[imvNum1].m_valid_probe >> (VECTORSIZE - num);
            imv_state[k].m_valid_probe = mask[VECTORSIZE];
            imv_state[imvNum1].stage = 4;
            imv_state[k].stage = 4;
          }
        }
        --k;
        break;
#endif
      }
        break;
      case 4: {

        /// step 3: compute the hash values of probe keys
        imv_state[k].v_probe_hash = hashFun()((imv_state[k].v_probe_keys), (v_seed));
        hash_table->prefetchEntry((imv_state[k].v_probe_hash));
        imv_state[k].stage = 2;
      }
        break;
      case 2: {
        /// step 4: find the addresses of corresponding buckets for new probes
        Vec8uM v_new_bucket_addrs = hash_table->find_chain_tagged((imv_state[k].v_probe_hash));
        imv_state[k].m_valid_probe = _mm512_kand(imv_state[k].m_valid_probe, v_new_bucket_addrs.mask);
        imv_state[k].v_bucket_addrs = v_new_bucket_addrs.vec;
#if 0
        imv_state[k].stage = 0;
        v_prefetch(imv_state[k].v_bucket_addrs);
#else
        num = _mm_popcnt_u32(imv_state[k].m_valid_probe);
        if (num == 0) {
          imv_state[k].stage = 1;
          --k;
        } else if (num == VECTORSIZE) {
          imv_state[k].stage = 0;
          v_prefetch(imv_state[k].v_bucket_addrs);
        } else {
          if ((done < imvNum) || (imv_state[imvNum1].m_valid_probe > 0)) {
            num_temp = _mm_popcnt_u32(imv_state[imvNum].m_valid_probe);
            if (num + num_temp < VECTORSIZE) {
              // compress imv_state[k]
              compress(&imv_state[k]);
              // expand imv_state[k] -> imv_state[imvNum]
              expand(&imv_state[k], &imv_state[imvNum]);
              imv_state[imvNum].m_valid_probe = mask[num + num_temp];
              imv_state[k].m_valid_probe = 0;
              imv_state[k].stage = 1;
              imv_state[imvNum].stage = 0;
              --k;
              break;
            } else {
              // expand imv_state[imvNum] -> expand imv_state[k]
              expand(&imv_state[imvNum], &imv_state[k]);
              imv_state[imvNum].m_valid_probe = _mm512_kand(imv_state[imvNum].m_valid_probe, _mm512_knot(mask[VECTORSIZE - num]));
              // compress imv_state[imvNum]
              compress(&imv_state[imvNum]);
              imv_state[imvNum].m_valid_probe = imv_state[imvNum].m_valid_probe >> (VECTORSIZE - num);
              imv_state[k].m_valid_probe = mask[VECTORSIZE];
              imv_state[k].stage = 0;
              imv_state[imvNum].stage = 0;
              v_prefetch(imv_state[k].v_bucket_addrs);
            }
          }
        }
#endif
      }
        break;
      case 0: {
        /// step 5: gather the all new build keys
        v256_build_keys = _mm512_mask_i64gather_epi32(v256_zero, imv_state[k].m_valid_probe, _mm512_add_epi64(imv_state[k].v_bucket_addrs, v_build_key_off), nullptr, 1);
        v_build_keys = _mm512_cvtepi32_epi64(v256_build_keys);
        /// step 6: compare the probe keys and build keys and write points
        m_match = _mm512_cmpeq_epi64_mask(imv_state[k].v_probe_keys, v_build_keys);
        m_match = _mm512_kand(m_match, imv_state[k].m_valid_probe);
#if WRITE_RESULTS
        pos = pos + VECTORSIZE < morselSize ? pos : 0;
#if WRITE_SEQ_PREFETCH
        _mm_prefetch((char *)(((char *)(output_build+pos)) + WRITE_PDIS), _MM_HINT_T0);
        _mm_prefetch((char *)(((char *)(output_build+pos)) + WRITE_PDIS + 64), _MM_HINT_T0);
        _mm_prefetch((char *)(((char *)(output_probe+pos)) + WRITE_PDIS), _MM_HINT_T0);
        _mm_prefetch((char *)(((char *)(output_probe+pos)) + WRITE_PDIS + 64), _MM_HINT_T0);
#endif
        _mm512_mask_compressstoreu_epi64((output_build + pos), m_match, imv_state[k].v_bucket_addrs);
        _mm256_mask_compressstoreu_epi32((output_probe + pos), m_match, _mm512_cvtepi64_epi32(imv_state[k].v_probe_offset));
        pos += _mm_popcnt_u32(m_match);
#endif
        found += _mm_popcnt_u32(m_match);
#if EARLY_BREAK
        imv_state[k].m_valid_probe = _mm512_kandn(m_match, imv_state[k].m_valid_probe);
#endif
        /// step 7: move to the next bucket nodes
        imv_state[k].v_bucket_addrs = _mm512_mask_i64gather_epi64(v_zero, imv_state[k].m_valid_probe, imv_state[k].v_bucket_addrs, nullptr, 1);
        imv_state[k].m_valid_probe = _mm512_kand(imv_state[k].m_valid_probe, _mm512_cmpneq_epi64_mask(imv_state[k].v_bucket_addrs, v_zero));

        num = _mm_popcnt_u32(imv_state[k].m_valid_probe);
        if (num == 0) {
          imv_state[k].stage = 1;
          --k;
        } else if (num == VECTORSIZE) {
          v_prefetch(imv_state[k].v_bucket_addrs);
        } else {
          if ((done < imvNum) || (imv_state[imvNum1].m_valid_probe > 0)) {
            num_temp = _mm_popcnt_u32(imv_state[imvNum].m_valid_probe);
            if (num + num_temp < VECTORSIZE) {
              // compress imv_state[k]
              compress(&imv_state[k]);
              // expand imv_state[k] -> imv_state[imvNum]
              expand(&imv_state[k], &imv_state[imvNum]);
              imv_state[imvNum].m_valid_probe = mask[num + num_temp];
              imv_state[k].m_valid_probe = 0;
              imv_state[k].stage = 1;
              imv_state[imvNum].stage = 0;
              --k;
              break;
            } else {
              // expand imv_state[imvNum] -> expand imv_state[k]
              expand(&imv_state[imvNum], &imv_state[k]);
              imv_state[imvNum].m_valid_probe = _mm512_kand(imv_state[imvNum].m_valid_probe, _mm512_knot(mask[VECTORSIZE - num]));
              // compress imv_state[imvNum]
              compress(&imv_state[imvNum]);
              imv_state[imvNum].m_valid_probe = imv_state[imvNum].m_valid_probe >> (VECTORSIZE - num);
              imv_state[k].m_valid_probe = mask[VECTORSIZE];
              imv_state[k].stage = 0;
              imv_state[imvNum].stage = 0;
              v_prefetch(imv_state[k].v_bucket_addrs);
            }
          }
        }
      }
        break;
    }
    ++k;
  }
  delete[] imv_state;
  imv_state = nullptr;
  return found;
}

size_t filter_probe_imv2(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, void** output_build, uint32_t*output_probe, uint64_t* pos_buff) {
  auto& li = db["lineitem"];
  auto l_quantity_col = li["l_quantity"].data<types::Numeric<12, 2>>();
#if PIPELINE_ORDERED
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
#else
  auto l_orderkey = li["l_partkey"].data<types::Integer>();
#endif

  ///////////////////////////////
  auto probe_keys = l_orderkey;
  size_t found = 0, pos = 0;
  int k = 0, done = 0, keyOff = sizeof(runtime::Hashmap::EntryHeader), imvNum = vectorwise::Hashjoin::imvNum, imvNum1 = vectorwise::Hashjoin::imvNum + 1, nextProbe = begin,
      curProbe;
  IMVState* imv_state = new IMVState[vectorwise::Hashjoin::imvNum + 2];

  __attribute__((aligned(64)))         __mmask8 m_match = 0, m_new_probes = -1, mask[VECTORSIZE + 1];

  __m512i v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
  __m512i v_offset = _mm512_set1_epi64(0), v_new_build_key, v_build_keys;
  __m512i v_base_offset_upper = _mm512_set1_epi64(end);
  __m512i v_seed = _mm512_set1_epi64(vectorwise::primitives::seed), v_build_key_off = _mm512_set1_epi64(keyOff);
  __m512i v_zero = _mm512_set1_epi64(0), v_const = _mm512_set1_epi64(constrants);
  __m256i v256_zero = _mm256_set1_epi32(0), v256_probe_keys, v256_build_keys;
  uint64_t * ht_pos = nullptr;
  uint8_t num, num_temp;
  for (int i = 0; i <= VECTORSIZE; ++i) {
    mask[i] = (1 << i) - 1;
  }
  while (true) {
    k = (k >= imvNum) ? 0 : k;
    if ((nextProbe >= end)) {
      if (imv_state[k].m_valid_probe == 0 && imv_state[k].stage != 3) {
        ++done;
        imv_state[k].stage = 3;
        ++k;
        continue;
      }
    }
    if (done >= imvNum) {
      if (imv_state[imvNum1].m_valid_probe > 0) {
        k = imvNum1;
      } else {
        if (imv_state[imvNum].m_valid_probe > 0) {
          k = imvNum;
        } else {
          break;
        }
      }
    }
    switch (imv_state[k].stage) {
      case 1: {
#if 0
        /// step 1: load the offsets of probing tuples
        imv_state[k].v_probe_offset = _mm512_add_epi64(_mm512_set1_epi64(nextProbe), v_base_offset);
        imv_state[k].m_valid_probe = _mm512_cmpgt_epu64_mask(v_base_offset_upper, imv_state[k].v_probe_offset);
        /// step 2: gather the probe keys // why is load() so faster than gather()?
        v256_probe_keys = _mm512_mask_i64gather_epi32(v256_zero, imv_state[k].m_valid_probe, imv_state[k].v_probe_offset, (void* )probe_keys, 4);
        // v256_probe_keys = _mm256_maskz_loadu_epi32(imv_state[k].m_valid_probe, (char*)(probe_keys+nextProbe));
        imv_state[k].v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
        /// step 2.5: filter
        imv_state[k].v_probe_hash = _mm512_maskz_loadu_epi64(imv_state[k].m_valid_probe, (l_quantity_col + nextProbe));
        m_match = _mm512_cmpgt_epu64_mask(v_const, imv_state[k].v_probe_hash);
        imv_state[k].m_valid_probe = _mm512_kand(imv_state[k].m_valid_probe, m_match);
        /// step 3: compute the hash values of probe keys
        imv_state[k].v_probe_hash = runtime::MurMurHash()((imv_state[k].v_probe_keys), (v_seed));
        nextProbe += VECTORSIZE;
        imv_state[k].stage = 2;
#if SEQ_PREFETCH
        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS+64), _MM_HINT_T0);
        _mm_prefetch((((char* )(l_quantity_col+nextProbe))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(l_quantity_col+nextProbe))+PDIS+64), _MM_HINT_T0);

#endif
        hash_table->prefetchEntry((imv_state[k].v_probe_hash));
#else
#if SEQ_PREFETCH

        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS+64), _MM_HINT_T0);
        _mm_prefetch((((char* )(l_quantity_col+nextProbe))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(l_quantity_col+nextProbe))+PDIS+64), _MM_HINT_T0);

#endif
        /// step 1: load the offsets of probing tuples
        imv_state[k].v_probe_offset = _mm512_add_epi64(_mm512_set1_epi64(nextProbe), v_base_offset);
        imv_state[k].m_valid_probe = _mm512_cmpgt_epu64_mask(v_base_offset_upper, imv_state[k].v_probe_offset);
        /// step 2: gather the probe keys // why is load() so faster than gather()?
        // v256_probe_keys = _mm512_mask_i64gather_epi32(v256_zero, imv_state[k].m_valid_probe, imv_state[k].v_probe_offset, (void* )probe_keys, 4);
        v256_probe_keys = _mm256_maskz_loadu_epi32(imv_state[k].m_valid_probe, (char*) (probe_keys + nextProbe));
        imv_state[k].v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
        /// step 2.5: filter
        imv_state[k].v_probe_hash = _mm512_maskz_loadu_epi64(imv_state[k].m_valid_probe, (l_quantity_col + nextProbe));
        m_match = _mm512_cmpgt_epu64_mask(v_const, imv_state[k].v_probe_hash);
        imv_state[k].m_valid_probe = _mm512_kand(imv_state[k].m_valid_probe, m_match);
        nextProbe += VECTORSIZE;
        num = _mm_popcnt_u32(imv_state[k].m_valid_probe);
        if (num == VECTORSIZE || done >= imvNum) {
          imv_state[k].stage = 4;
        } else {
          num_temp = _mm_popcnt_u32(imv_state[imvNum1].m_valid_probe);
          if (num + num_temp < VECTORSIZE) {
            // compress imv_state[k]
            compress(&imv_state[k]);
            // expand imv_state[k] -> imv_state[imvNum1]
            expand(&imv_state[k], &imv_state[imvNum1]);
            imv_state[imvNum1].m_valid_probe = mask[num + num_temp];
            imv_state[k].m_valid_probe = 0;
            imv_state[k].stage = 1;
            imv_state[imvNum1].stage = 4;

          } else {
            // expand imv_state[imvNum1] -> expand imv_state[k]
            expand(&imv_state[imvNum1], &imv_state[k]);
            imv_state[imvNum1].m_valid_probe = _mm512_kand(imv_state[imvNum1].m_valid_probe, _mm512_knot(mask[VECTORSIZE - num]));
            // compress imv_state[imvNum]
            compress(&imv_state[imvNum1]);
            imv_state[imvNum1].m_valid_probe = imv_state[imvNum1].m_valid_probe >> (VECTORSIZE - num);
            imv_state[k].m_valid_probe = mask[VECTORSIZE];
            imv_state[imvNum1].stage = 4;
            imv_state[k].stage = 4;
          }
        }
        --k;
#endif
      }
        break;
      case 4: {

        /// step 3: compute the hash values of probe keys
        imv_state[k].v_probe_hash = hashFun()((imv_state[k].v_probe_keys), (v_seed));
        hash_table->prefetchEntry((imv_state[k].v_probe_hash));
        imv_state[k].stage = 2;
      }
        break;
      case 2: {
        /// step 4: find the addresses of corresponding buckets for new probes
        Vec8uM v_new_bucket_addrs = hash_table->find_chain_tagged((imv_state[k].v_probe_hash));
        imv_state[k].m_valid_probe = _mm512_kand(imv_state[k].m_valid_probe, v_new_bucket_addrs.mask);
        imv_state[k].v_bucket_addrs = v_new_bucket_addrs.vec;
        imv_state[k].stage = 0;
        v_prefetch(imv_state[k].v_bucket_addrs);

      }
        break;
      case 0: {
        /// step 5: gather the all new build keys
        v256_build_keys = _mm512_mask_i64gather_epi32(v256_zero, imv_state[k].m_valid_probe, _mm512_add_epi64(imv_state[k].v_bucket_addrs, v_build_key_off), nullptr, 1);
        v_build_keys = _mm512_cvtepi32_epi64(v256_build_keys);
        /// step 6: compare the probe keys and build keys and write points
        m_match = _mm512_cmpeq_epi64_mask(imv_state[k].v_probe_keys, v_build_keys);
        m_match = _mm512_kand(m_match, imv_state[k].m_valid_probe);
#if WRITE_RESULTS
        pos = pos + VECTORSIZE < morselSize ? pos : 0;
#if WRITE_SEQ_PREFETCH
        _mm_prefetch((char *)(((char *)(output_build+pos)) + WRITE_PDIS), _MM_HINT_T0);
        _mm_prefetch((char *)(((char *)(output_build+pos)) + WRITE_PDIS + 64), _MM_HINT_T0);
        _mm_prefetch((char *)(((char *)(output_probe+pos)) + WRITE_PDIS), _MM_HINT_T0);
        _mm_prefetch((char *)(((char *)(output_probe+pos)) + WRITE_PDIS + 64), _MM_HINT_T0);
#endif
        _mm512_mask_compressstoreu_epi64((output_build + pos), m_match, imv_state[k].v_bucket_addrs);
        _mm256_mask_compressstoreu_epi32((output_probe + pos), m_match, _mm512_cvtepi64_epi32(imv_state[k].v_probe_offset));
        pos += _mm_popcnt_u32(m_match);
#endif
        found += _mm_popcnt_u32(m_match);
#if EARLY_BREAK
        imv_state[k].m_valid_probe = _mm512_kandn(m_match, imv_state[k].m_valid_probe);
#endif
        /// step 7: move to the next bucket nodes
        imv_state[k].v_bucket_addrs = _mm512_mask_i64gather_epi64(v_zero, imv_state[k].m_valid_probe, imv_state[k].v_bucket_addrs, nullptr, 1);
        imv_state[k].m_valid_probe = _mm512_kand(imv_state[k].m_valid_probe, _mm512_cmpneq_epi64_mask(imv_state[k].v_bucket_addrs, v_zero));

        num = _mm_popcnt_u32(imv_state[k].m_valid_probe);
        if (num == VECTORSIZE) {
          v_prefetch(imv_state[k].v_bucket_addrs);
        } else {
          if ((done < imvNum) || (imv_state[imvNum1].m_valid_probe > 0)) {
            num_temp = _mm_popcnt_u32(imv_state[imvNum].m_valid_probe);
            if (num + num_temp < VECTORSIZE) {
              // compress imv_state[k]
              compress(&imv_state[k]);
              // expand imv_state[k] -> imv_state[imvNum]
              expand(&imv_state[k], &imv_state[imvNum]);
              imv_state[imvNum].m_valid_probe = mask[num + num_temp];
              imv_state[k].m_valid_probe = 0;
              imv_state[k].stage = 1;
              imv_state[imvNum].stage = 0;
              --k;
              break;
            } else {
              // expand imv_state[imvNum] -> expand imv_state[k]
              expand(&imv_state[imvNum], &imv_state[k]);
              imv_state[imvNum].m_valid_probe = _mm512_kand(imv_state[imvNum].m_valid_probe, _mm512_knot(mask[VECTORSIZE - num]));
              // compress imv_state[imvNum]
              compress(&imv_state[imvNum]);
              imv_state[imvNum].m_valid_probe = imv_state[imvNum].m_valid_probe >> (VECTORSIZE - num);
              imv_state[k].m_valid_probe = mask[VECTORSIZE];
              imv_state[k].stage = 0;
              imv_state[imvNum].stage = 0;
              v_prefetch(imv_state[k].v_bucket_addrs);
            }
          }
        }
      }
        break;
    }
    ++k;
  }
  delete[] imv_state;
  imv_state = nullptr;
  return found;
}
