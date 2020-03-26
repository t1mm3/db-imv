#include "imv/PipelineSSB.hpp"
inline uint64_t v_sum(__m512i& vec) {
  uint64_t * ht_pos = (uint64_t*)&vec;
  uint64_t sum =0;
  for(int i=0;i<VECTORSIZE;++i) {
    sum += ht_pos[i];
  }
  return sum;
}
size_t pipeline_imv_q1x(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, uint64_t& results) {
  // --- scan lineorder
  auto& lo = db["lineorder"];
  auto lo_orderdate = lo["lo_orderdate"].data<types::Integer>();
  auto lo_quantity = lo["lo_quantity"].data<types::Integer>();
  auto lo_discount = lo["lo_discount"].data<types::Numeric<18, 2>>();
  auto lo_extendedprice = lo["lo_extendedprice"].data<types::Numeric<18, 2>>();

  ///////////////////////////////
  auto probe_keys = lo_orderdate;
  size_t found = 0, pos = 0, nextProbe = begin;
  int k = 0, done = 0, keyOff = sizeof(runtime::Hashmap::EntryHeader), imvNum = vectorwise::Hashjoin::imvNum, imvNum1 = vectorwise::Hashjoin::imvNum + 1;
  IMVState* imv_state = new IMVState[vectorwise::Hashjoin::imvNum + 2];

  __attribute__((aligned(64)))        __mmask8 m_match = 0, m_new_probes = -1, mask[VECTORSIZE + 1];

  __m512i v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
  __m512i v_offset = _mm512_set1_epi64(0), v_new_build_key, v_build_keys;
  __m512i v_base_offset_upper = _mm512_set1_epi64(end);
  __m512i v_seed = _mm512_set1_epi64(vectorwise::primitives::seed), v_build_key_off = _mm512_set1_epi64(keyOff);
  __m512i v_zero = _mm512_set1_epi64(0), v_filter_value;
  __m512i v_100 = _mm512_set1_epi64(100), v_300 = _mm512_set1_epi64(300), v_price, v_discount, v_results = _mm512_set1_epi64(0);
  __m256i v256_zero = _mm256_set1_epi32(0), v256_probe_keys, v256_build_keys, v256_25 = _mm256_set1_epi32(25), v256_filter_value;
  uint64_t * ht_pos = nullptr;
  uint8_t num, num_temp;

  auto filterQ11 = [&](IMVState& state, size_t cur) {
    // LO_DISCOUNT BETWEEN 1 AND 3
      v_filter_value = _mm512_maskz_loadu_epi64(state.m_valid_probe, (lo_discount + cur));
      m_match = _mm512_kand( _mm512_cmpge_epu64_mask(v_300, v_filter_value),_mm512_cmpge_epu64_mask(v_filter_value,v_100));
      // LO_QUANTITY < 25
      v256_filter_value = _mm256_maskz_loadu_epi32(state.m_valid_probe,(lo_quantity + cur));
      m_match = _mm512_kand(m_match,_mm256_cmpgt_epu32_mask(v256_25,v256_filter_value));
      state.m_valid_probe = _mm512_kand(state.m_valid_probe,m_match);
    };

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
        imv_state[k].v_probe_hash = _mm512_maskz_loadu_epi64(imv_state[k].m_valid_probe, (lo_discount+nextProbe));
        m_match = _mm512_cmpgt_epu64_mask(v_const, imv_state[k].v_probe_hash);
        imv_state[k].m_valid_probe = _mm512_kand(imv_state[k].m_valid_probe ,m_match);
        /// step 3: compute the hash values of probe keys
        imv_state[k].v_probe_hash = runtime::MurMurHash()((imv_state[k].v_probe_keys), (v_seed));
        nextProbe += VECTORSIZE;
        imv_state[k].stage = 2;
#if SEQ_PREFETCH
        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS+64), _MM_HINT_T0);
#endif
        hash_table->prefetchEntry((imv_state[k].v_probe_hash));
#else
#if SEQ_PREFETCH
        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS), _MM_HINT_T0);
        _mm_prefetch((((char* )(probe_keys+nextProbe))+PDIS+64), _MM_HINT_T0);

#endif
        /// step 1: load the offsets of probing tuples
        imv_state[k].v_probe_offset = _mm512_add_epi64(_mm512_set1_epi64(nextProbe), v_base_offset);
        imv_state[k].m_valid_probe = _mm512_cmpgt_epu64_mask(v_base_offset_upper, imv_state[k].v_probe_offset);
        /// step 2: gather the probe keys // why is load() so faster than gather()?
        // v256_probe_keys = _mm512_mask_i64gather_epi32(v256_zero, imv_state[k].m_valid_probe, imv_state[k].v_probe_offset, (void* )probe_keys, 4);
        v256_probe_keys = _mm256_maskz_loadu_epi32(imv_state[k].m_valid_probe, (char*) (probe_keys + nextProbe));
        imv_state[k].v_probe_keys = _mm512_cvtepi32_epi64(v256_probe_keys);
        /// step 2.5: filter
        filterQ11(imv_state[k], nextProbe);

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
            --k;
            break;
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
#endif
      }
        break;
      case 4: {

        /// step 3: compute the hash values of probe keys
        imv_state[k].v_probe_hash = runtime::MurMurHash()((imv_state[k].v_probe_keys), (v_seed));
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
        ht_pos = (uint64_t *) &imv_state[k].v_bucket_addrs;
        for (int i = 0; i < VECTORSIZE; ++i) {
          _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
          _mm_prefetch(((char * )(ht_pos[i]) + 64), _MM_HINT_T0);
        }
      }
        break;
      case 0: {
        /// step 5: gather the all new build keys
        v256_build_keys = _mm512_mask_i64gather_epi32(v256_zero, imv_state[k].m_valid_probe, _mm512_add_epi64(imv_state[k].v_bucket_addrs, v_build_key_off), nullptr, 1);
        v_build_keys = _mm512_cvtepi32_epi64(v256_build_keys);
        /// step 6: compare the probe keys and build keys and write points
        m_match = _mm512_cmpeq_epi64_mask(imv_state[k].v_probe_keys, v_build_keys);
        m_match = _mm512_kand(m_match, imv_state[k].m_valid_probe);
#if 0
        pos = pos + VECTORSIZE < morselSize ? pos : 0;
#if WRITE_SEQ_PREFETCH
        _mm_prefetch((char *)(((char *)(output_build+pos)) + PDIS), _MM_HINT_T0);
        _mm_prefetch((char *)(((char *)(output_build+pos)) + PDIS + 64), _MM_HINT_T0);
        _mm_prefetch((char *)(((char *)(output_probe+pos)) + PDIS), _MM_HINT_T0);
        _mm_prefetch((char *)(((char *)(output_probe+pos)) + PDIS + 64), _MM_HINT_T0);
#endif
        _mm512_mask_compressstoreu_epi64((output_build + pos), m_match, imv_state[k].v_bucket_addrs);
        _mm256_mask_compressstoreu_epi32((output_probe + pos), m_match, _mm512_cvtepi64_epi32(imv_state[k].v_probe_offset));
        pos += _mm_popcnt_u32(m_match);
#else
        v_price = _mm512_mask_i64gather_epi64(v_zero, imv_state[k].m_valid_probe, imv_state[k].v_probe_offset, (const long long int*)lo_extendedprice, 8);
        v_discount = _mm512_mask_i64gather_epi64(v_zero, imv_state[k].m_valid_probe, imv_state[k].v_probe_offset, (const long long int*)lo_discount, 8);
        v_price = _mm512_mullo_epi64(v_price, v_discount);
        v_results = _mm512_mask_add_epi64(v_results, m_match, v_price, v_results);
        imv_state[k].m_valid_probe = _mm512_kandn(m_match, imv_state[k].m_valid_probe);
#endif
        found += _mm_popcnt_u32(m_match);

        /// step 7: move to the next bucket nodes
        imv_state[k].v_bucket_addrs = _mm512_mask_i64gather_epi64(v_zero, imv_state[k].m_valid_probe, imv_state[k].v_bucket_addrs, nullptr, 1);
        imv_state[k].m_valid_probe = _mm512_kand(imv_state[k].m_valid_probe, _mm512_cmpneq_epi64_mask(imv_state[k].v_bucket_addrs, v_zero));

        num = _mm_popcnt_u32(imv_state[k].m_valid_probe);
        if (num == VECTORSIZE) {
          ht_pos = (uint64_t *) &imv_state[k].v_bucket_addrs;
          for (int i = 0; i < VECTORSIZE; ++i) {
            _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
            _mm_prefetch(((char * )(ht_pos[i]) + 64), _MM_HINT_T0);
          }
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
              ht_pos = (uint64_t *) &imv_state[k].v_bucket_addrs;
              for (int i = 0; i < VECTORSIZE; ++i) {
                _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
                _mm_prefetch(((char * )(ht_pos[i]) + 64), _MM_HINT_T0);
              }
            }
          }
        }
      }
        break;
    }
    ++k;
  }
  results = v_sum(v_results);
  delete[] imv_state;
  imv_state = nullptr;
  return found;
}
