#include "imv-operator/no_partitioning_join.h"
#include "imv-operator/tree_node.h"
#define WORDSIZE 8
// target for 8B keys and 8B payload
int64_t bts_simd(tree_t *tree, relation_t *rel, void *output) {
  int64_t matches = 0;
  int32_t new_add = 0;
  __mmask8 m_match = 0, m_have_tuple = 0, m_new_cells = -1, m_valid_bucket = 0, m_less = 0, m_more = 0;
  __m512i v_offset = _mm512_set1_epi64(0), v_addr_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(rel->num_tuples * sizeof(tuple_t)), v_tuple_key =
      _mm512_set1_epi64(0), v_base_offset, v_node_key, v_cell_hash, v_tree_pos = _mm512_set1_epi64(tree->first_node), v_neg_one512 = _mm512_set1_epi64(-1), v_zero512 =
      _mm512_set1_epi64(0), v_next_addr = _mm512_set1_epi64(0), v_write_index = _mm512_set1_epi64(0), v_tree_addr = _mm512_set1_epi64(tree->first_node), v_word_size =
      _mm512_set1_epi64(WORDSIZE), v_tuple_size = _mm512_set1_epi64(sizeof(tuple_t)), v_bucket_size = _mm512_set1_epi64(sizeof(bucket_t)), v_lnext_off = _mm512_set1_epi64(16),
      v_tuple_payload, v_node_payload, v_rnext_off = _mm512_set1_epi64(24), v_payload_off = _mm512_set1_epi64(8);
  chainedtuplebuffer_t *chainedbuf = (chainedtuplebuffer_t *) output;
  tuple_t *join_res = NULL;
  __attribute__((aligned(64)))  uint64_t cur_offset = 0, base_off[16], *ht_pos;

  for (int i = 0; i <= VECTOR_SCALE; ++i) {
    base_off[i] = i * sizeof(tuple_t);
  }
  v_base_offset = _mm512_load_epi64(base_off);
  for (uint64_t cur = 0; cur < rel->num_tuples || m_have_tuple;) {
    ///////// step 1: load new tuples' address offsets
    // the offset should be within MAX_32INT_
    // the tail depends on the number of joins and tuples in each bucket
    /*#if SEQPREFETCH
     _mm_prefetch((char *)(((void *)rel->tuples) + cur_offset + PDIS),
     _MM_HINT_T0);
     _mm_prefetch((char *)(((void *)rel->tuples) + cur_offset + PDIS + 64),
     _MM_HINT_T0);
     _mm_prefetch((char *)(((void *)rel->tuples) + cur_offset + PDIS + 128),
     _MM_HINT_T0);
     _mm_prefetch((char *)(((void *)rel->tuples) + cur_offset + PDIS + 192),
     _MM_HINT_T0);
     _mm_prefetch((char *)(((void *)rel->tuples) + cur_offset + PDIS + 256),
     _MM_HINT_T0);
     #endif
     */
    // directly use cur, instead of cur_offset to control the offset to rel.
    // In this case, using step = 16 to gather data, but step is larger
    // than the scale 1,2,4 or 8
    v_offset = _mm512_add_epi64(_mm512_set1_epi64(cur_offset), v_base_offset);
    v_addr_offset = _mm512_mask_expand_epi64(v_addr_offset, _mm512_knot(m_have_tuple), v_offset);
    // count the number of empty tuples
    m_new_cells = _mm512_knot(m_have_tuple);
    new_add = _mm_popcnt_u32(m_new_cells);
    cur_offset = cur_offset + base_off[new_add];
    cur = cur + new_add;
    m_have_tuple = _mm512_cmpgt_epi64_mask(v_base_offset_upper, v_addr_offset);
    ///// step 2: load new cells from right tuples;
    m_new_cells = _mm512_kand(m_new_cells, m_have_tuple);
    // maybe need offset within a tuple
    v_tuple_key = _mm512_mask_i64gather_epi64(v_tuple_key, m_new_cells, v_addr_offset, ((void * )rel->tuples), 1);
    v_tuple_payload = _mm512_mask_i64gather_epi64(v_tuple_payload, m_new_cells, _mm512_add_epi64(v_addr_offset, v_word_size), ((void * )rel->tuples), 1);

    // step 3: load new nodes from tree
    v_tree_pos = _mm512_mask_blend_epi64(m_new_cells, v_tree_pos, v_tree_addr);

    /////////////////// random access
    v_node_key = _mm512_mask_i64gather_epi64(v_neg_one512, m_have_tuple, v_tree_pos, 0, 1);
    ///// step 4: compare;
    m_match = _mm512_cmpeq_epi64_mask(v_tuple_key, v_node_key);
    m_match = _mm512_kand(m_match, m_have_tuple);
    // case: tuple->key == node->key
    new_add = _mm_popcnt_u32(m_match);
    matches += new_add;
    m_have_tuple = _mm512_kandn(m_match, m_have_tuple);

    // gather payloads and scatter equal results
    v_node_payload = _mm512_mask_i64gather_epi64(v_neg_one512, m_match, _mm512_add_epi64(v_tree_pos, v_payload_off), 0, 1);
    join_res = cb_next_n_writepos(chainedbuf, new_add);
    v_write_index = _mm512_mask_expand_epi64(v_zero512, m_match, v_base_offset);
    _mm512_mask_i64scatter_epi64((void * )join_res, m_match, v_write_index, v_tuple_payload, 1);
    v_write_index = _mm512_add_epi64(v_write_index, v_word_size);
    _mm512_mask_i64scatter_epi64((void * )join_res, m_match, v_write_index, v_node_payload, 1);

    // case: tuple->key != node->key
    m_less = _mm512_cmplt_epi64_mask(v_tuple_key, v_node_key);
    m_more = _mm512_cmpgt_epi64_mask(v_tuple_key, v_node_key);
    v_next_addr = _mm512_mask_add_epi64(v_tree_pos, m_less, v_tree_pos, v_lnext_off);
    v_next_addr = _mm512_mask_add_epi64(v_next_addr, m_more, v_tree_pos, v_rnext_off);

    v_tree_pos = _mm512_mask_i64gather_epi64(v_tree_pos, m_have_tuple, v_next_addr, 0, 1);
    m_have_tuple = _mm512_kand(_mm512_cmpneq_epi64_mask(v_tree_pos, v_zero512), m_have_tuple);
  }
  return matches;
}

int64_t bts_simd_amac_raw(tree_t *tree, relation_t *rel, void *output) {
  int64_t matches = 0;
  int32_t new_add = 0, k = 0, done = 0;
  __mmask8 m_match = 0, m_new_cells = -1, m_valid_bucket = 0, m_less, m_more;
  __m512i v_offset = _mm512_set1_epi64(0), v_addr_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(rel->num_tuples * sizeof(tuple_t)), v_tuple_key =
      _mm512_set1_epi64(0), v_base_offset, v_node_key, v_neg_one512 = _mm512_set1_epi64(-1), v_zero512 = _mm512_set1_epi64(0), v_next_addr = _mm512_set1_epi64(0), v_write_index =
      _mm512_set1_epi64(0), v_tree_addr = _mm512_set1_epi64(tree->first_node), v_word_size = _mm512_set1_epi64(WORDSIZE), v_tuple_size = _mm512_set1_epi64(sizeof(tuple_t)),
      v_bucket_size = _mm512_set1_epi64(sizeof(bucket_t)), v_lnext_off = _mm512_set1_epi64(16), v_tuple_payload, v_node_payload, v_rnext_off = _mm512_set1_epi64(24),
      v_payload_off = _mm512_set1_epi64(8);
  chainedtuplebuffer_t *chainedbuf = (chainedtuplebuffer_t *) output;
  tuple_t *join_res = NULL;
  __attribute__((aligned(64)))  uint64_t cur_offset = 0, base_off[16], *ht_pos;

  for (int i = 0; i <= VECTOR_SCALE; ++i) {
    base_off[i] = i * sizeof(tuple_t);
  }
  v_base_offset = _mm512_load_epi64(base_off);
  StateSIMD state[SIMDStateSize];
  // init # of the state
  for (int i = 0; i < SIMDStateSize; ++i) {
    state[i].stage = 1;
    state[i].m_have_tuple = 0;
    state[i].ht_off = _mm512_set1_epi64(0);
    state[i].tb_off = _mm512_set1_epi64(0);
    state[i].key = _mm512_set1_epi64(0);
  }
  for (uint64_t cur = 0; (cur < rel->num_tuples) || (done < SIMDStateSize);) {
    k = (k >= SIMDStateSize) ? 0 : k;
    if (cur >= rel->num_tuples) {
      if (state[k].m_have_tuple == 0 && state[k].stage != 3) {
        ++done;
        state[k].stage = 3;
        ++k;
        continue;
      }
    }
    switch (state[k].stage) {
      case 1: {
///////// step 1: load new tuples' address offsets
// the offset should be within MAX_32INT_
// the tail depends on the number of joins and tuples in each bucket
#if SEQPREFETCH
        _mm_prefetch((char *)(((void *)rel->tuples) + cur_offset + PDIS), _MM_HINT_T0);
        _mm_prefetch((char *)(((void *)rel->tuples) + cur_offset + PDIS + 64), _MM_HINT_T0);
        _mm_prefetch((char *)(((void *)rel->tuples) + cur_offset + PDIS + 128), _MM_HINT_T0);

#endif
        // directly use cur, instead of cur_offset to control the offset to rel.
        // In this case, using step = 16 to gather data, but step is larger
        // than the scale 1,2,4 or 8
        v_offset = _mm512_add_epi64(_mm512_set1_epi64(cur_offset), v_base_offset);
        // count the number of empty tuples
        cur_offset = cur_offset + base_off[VECTOR_SCALE];
        cur = cur + VECTOR_SCALE;
        state[k].m_have_tuple = _mm512_cmpgt_epi64_mask(v_base_offset_upper, v_offset);
        ///// step 2: load new cells from right tuples;
        // maybe need offset within a tuple
        state[k].key = _mm512_mask_i64gather_epi64(state[k].key, state[k].m_have_tuple, v_offset, ((void * )rel->tuples), 1);
        state[k].payload = _mm512_mask_i64gather_epi64(state[k].payload, state[k].m_have_tuple, _mm512_add_epi64(v_offset, v_word_size), ((void * )rel->tuples), 1);

        ///// step 3: load new values from hash tables;
        state[k].ht_off = v_tree_addr;
        state[k].stage = 0;
#if KNL
        _mm512_mask_prefetch_i64gather_pd(
            state[k].ht_off, state[k].m_have_tuple, 0, 1, _MM_HINT_T0);
#elif DIR_PREFETCH
        ht_pos = (uint64_t *) &state[k].ht_off;
        for (int i = 0; i < VECTOR_SCALE; ++i) {
          _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
        }
#else
        m_have_tuple = state[k].m_have_tuple;
        ht_pos = (uint64_t *)&state[k].ht_off;
        for (int i = 0; (i < VECTOR_SCALE) & (m_have_tuple);
            ++i, (m_have_tuple >> 1)) {
          if (m_have_tuple & 1) {
            _mm_prefetch((char *)(ht_pos[i]), _MM_HINT_T0);
          }
        }
#endif
      }
        break;
      case 0: {
        v_node_key = _mm512_mask_i64gather_epi64(v_neg_one512, state[k].m_have_tuple, state[k].ht_off, 0, 1);
        ///// step 4: compare;
        m_match = _mm512_cmpeq_epi64_mask(state[k].key, v_node_key);
        m_match = _mm512_kand(m_match, state[k].m_have_tuple);
        // case: tuple->key == node->key
        new_add = _mm_popcnt_u32(m_match);
        matches += new_add;
        state[k].m_have_tuple = _mm512_kandn(m_match, state[k].m_have_tuple);

        // gather payloads
        v_node_payload = _mm512_mask_i64gather_epi64(v_neg_one512, m_match, _mm512_add_epi64(state[k].ht_off, v_payload_off), 0, 1);

        // to scatter join results
        join_res = cb_next_n_writepos(chainedbuf, new_add);
#if SEQPREFETCH
        _mm_prefetch((char *)(((void *)join_res) + PDIS), _MM_HINT_T0);
        _mm_prefetch((char *)(((void *)join_res) + PDIS + 64), _MM_HINT_T0);
        _mm_prefetch((char *)(((void *)join_res) + PDIS + 128), _MM_HINT_T0);
#endif
        v_write_index = _mm512_mask_expand_epi64(v_zero512, m_match, v_base_offset);
        _mm512_mask_i64scatter_epi64((void * )join_res, m_match, v_write_index, state[k].payload, 1);
        v_write_index = _mm512_add_epi64(v_write_index, v_word_size);
        _mm512_mask_i64scatter_epi64((void * )join_res, m_match, v_write_index, v_node_payload, 1);

        // case: tuple->key != node->key
        m_less = _mm512_cmplt_epi64_mask(state[k].key, v_node_key);
        m_more = _mm512_cmpgt_epi64_mask(state[k].key, v_node_key);
        v_next_addr = _mm512_mask_add_epi64(state[k].ht_off, m_less, state[k].ht_off, v_lnext_off);
        v_next_addr = _mm512_mask_add_epi64(v_next_addr, m_more, state[k].ht_off, v_rnext_off);

        // update next
        state[k].ht_off = _mm512_mask_i64gather_epi64(state[k].ht_off, state[k].m_have_tuple, v_next_addr, 0, 1);
        state[k].m_have_tuple = _mm512_kand(_mm512_cmpneq_epi64_mask(state[k].ht_off, v_zero512), state[k].m_have_tuple);
        new_add = _mm_popcnt_u32(state[k].m_have_tuple);
        if (new_add == 0) {
          state[k].stage = 1;
        } else {
#if KNL
          _mm512_mask_prefetch_i64gather_pd(
              state[k].ht_off, state[k].m_have_tuple, 0, 1, _MM_HINT_T0);
#elif DIR_PREFETCH
          ht_pos = (uint64_t *) &state[k].ht_off;
          for (int i = 0; i < VECTOR_SCALE; ++i) {
            _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
          }
#else
          m_have_tuple = state[k].m_have_tuple;
          ht_pos = (uint64_t *)&state[k].ht_off;
          for (int i = 0; (i < VECTOR_SCALE) & (m_have_tuple);
              ++i, (m_have_tuple >> 1)) {
            if (m_have_tuple & 1) {
              _mm_prefetch((char *)(ht_pos[i]), _MM_HINT_T0);
            }
          }
#endif
        }
      }
        break;
    }
    ++k;
  }
  return matches;
}
int64_t bts_simd_amac(tree_t *tree, relation_t *rel, void *output) {
  int64_t matches = 0;
  int32_t new_add = 0, k = 0, done = 0;
  __mmask8 m_match = 0, m_new_cells = -1, m_valid_bucket = 0, m_less, m_more;
  __m512i v_offset = _mm512_set1_epi64(0), v_addr_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(rel->num_tuples * sizeof(tuple_t)), v_tuple_key =
      _mm512_set1_epi64(0), v_base_offset, v_node_key, v_neg_one512 = _mm512_set1_epi64(-1), v_zero512 = _mm512_set1_epi64(0), v_next_addr = _mm512_set1_epi64(0), v_write_index =
      _mm512_set1_epi64(0), v_tree_addr = _mm512_set1_epi64(tree->first_node), v_word_size = _mm512_set1_epi64(WORDSIZE), v_tuple_size = _mm512_set1_epi64(sizeof(tuple_t)),
      v_bucket_size = _mm512_set1_epi64(sizeof(bucket_t)), v_lnext_off = _mm512_set1_epi64(16), v_tuple_payload, v_node_payload, v_rnext_off = _mm512_set1_epi64(24),
      v_payload_off = _mm512_set1_epi64(8);
  chainedtuplebuffer_t *chainedbuf = (chainedtuplebuffer_t *) output;
  tuple_t *join_res = NULL;
  __attribute__((aligned(64)))  uint64_t cur_offset = 0, base_off[16], *ht_pos;

  for (int i = 0; i <= VECTOR_SCALE; ++i) {
    base_off[i] = i * sizeof(tuple_t);
  }
  v_base_offset = _mm512_load_epi64(base_off);
  StateSIMD state[SIMDStateSize];
  // init # of the state
  for (int i = 0; i < SIMDStateSize; ++i) {
    state[i].stage = 1;
    state[i].m_have_tuple = 0;
    state[i].ht_off = _mm512_set1_epi64(0);
    state[i].tb_off = _mm512_set1_epi64(0);
    state[i].key = _mm512_set1_epi64(0);
  }
  for (uint64_t cur = 0; (cur < rel->num_tuples) || (done < SIMDStateSize);) {
    k = (k >= SIMDStateSize) ? 0 : k;
    if (cur >= rel->num_tuples) {
      if (state[k].m_have_tuple == 0 && state[k].stage != 3) {
        ++done;
        state[k].stage = 3;
        ++k;
        continue;
      }
    }
    switch (state[k].stage) {
      case 1: {
///////// step 1: load new tuples' address offsets
// the offset should be within MAX_32INT_
// the tail depends on the number of joins and tuples in each bucket
#if SEQPREFETCH
        _mm_prefetch((char *)(((void *)rel->tuples) + cur_offset + PDIS), _MM_HINT_T0);
        _mm_prefetch((char *)(((void *)rel->tuples) + cur_offset + PDIS + 64), _MM_HINT_T0);
        _mm_prefetch((char *)(((void *)rel->tuples) + cur_offset + PDIS + 128), _MM_HINT_T0);

#endif
        // directly use cur, instead of cur_offset to control the offset to rel.
        // In this case, using step = 16 to gather data, but step is larger
        // than the scale 1,2,4 or 8
        v_offset = _mm512_add_epi64(_mm512_set1_epi64(cur_offset), v_base_offset);
        state[k].tb_off = _mm512_mask_expand_epi64(state[k].tb_off, _mm512_knot(state[k].m_have_tuple), v_offset);
        // count the number of empty tuples
        m_new_cells = _mm512_knot(state[k].m_have_tuple);
        new_add = _mm_popcnt_u32(m_new_cells);
        cur_offset = cur_offset + base_off[new_add];
        cur = cur + new_add;
        state[k].m_have_tuple = _mm512_cmpgt_epi64_mask(v_base_offset_upper, state[k].tb_off);
        ///// step 2: load new cells from right tuples;
        m_new_cells = _mm512_kand(m_new_cells, state[k].m_have_tuple);

        // maybe need offset within a tuple
        state[k].key = _mm512_mask_i64gather_epi64(state[k].key, m_new_cells, state[k].tb_off, ((void * )rel->tuples), 1);
        state[k].payload = _mm512_mask_i64gather_epi64(state[k].payload, state[k].m_have_tuple, _mm512_add_epi64(state[k].tb_off, v_word_size), ((void * )rel->tuples), 1);
        ///// step 3: load new values from hash tables;
        state[k].ht_off = _mm512_mask_blend_epi64(m_new_cells, state[k].ht_off, v_tree_addr);
        state[k].stage = 0;
#if KNL
        _mm512_mask_prefetch_i64gather_pd(
            state[k].ht_off, state[k].m_have_tuple, 0, 1, _MM_HINT_T0);
#elif DIR_PREFETCH
        ht_pos = (uint64_t *) &state[k].ht_off;
        for (int i = 0; i < VECTOR_SCALE; ++i) {
          _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
        }
#else
        m_have_tuple = state[k].m_have_tuple;
        ht_pos = (uint64_t *)&state[k].ht_off;
        for (int i = 0; (i < VECTOR_SCALE) & (m_have_tuple);
            ++i, (m_have_tuple >> 1)) {
          if (m_have_tuple & 1) {
            _mm_prefetch((char *)(ht_pos[i]), _MM_HINT_T0);
          }
        }
#endif
      }
        break;
      case 0: {
        v_node_key = _mm512_mask_i64gather_epi64(v_neg_one512, state[k].m_have_tuple, state[k].ht_off, 0, 1);
        ///// step 4: compare;
        m_match = _mm512_cmpeq_epi64_mask(state[k].key, v_node_key);
        m_match = _mm512_kand(m_match, state[k].m_have_tuple);
        // case: tuple->key == node->key
        new_add = _mm_popcnt_u32(m_match);
        matches += new_add;
        state[k].m_have_tuple = _mm512_kandn(m_match, state[k].m_have_tuple);

        // gather payloads
        v_node_payload = _mm512_mask_i64gather_epi64(v_neg_one512, m_match, _mm512_add_epi64(state[k].ht_off, v_payload_off), 0, 1);

        // to scatter join results
        join_res = cb_next_n_writepos(chainedbuf, new_add);
#if SEQPREFETCH
        _mm_prefetch((char *)(((void *)join_res) + PDIS), _MM_HINT_T0);
        _mm_prefetch((char *)(((void *)join_res) + PDIS + 64), _MM_HINT_T0);
        _mm_prefetch((char *)(((void *)join_res) + PDIS + 128), _MM_HINT_T0);
#endif
        v_write_index = _mm512_mask_expand_epi64(v_zero512, m_match, v_base_offset);
        _mm512_mask_i64scatter_epi64((void * )join_res, m_match, v_write_index, state[k].payload, 1);
        v_write_index = _mm512_add_epi64(v_write_index, v_word_size);
        _mm512_mask_i64scatter_epi64((void * )join_res, m_match, v_write_index, v_node_payload, 1);

        // case: tuple->key != node->key
        m_less = _mm512_cmplt_epi64_mask(state[k].key, v_node_key);
        m_more = _mm512_cmpgt_epi64_mask(state[k].key, v_node_key);
        v_next_addr = _mm512_mask_add_epi64(state[k].ht_off, m_less, state[k].ht_off, v_lnext_off);
        v_next_addr = _mm512_mask_add_epi64(v_next_addr, m_more, state[k].ht_off, v_rnext_off);

        // update next
        state[k].ht_off = _mm512_mask_i64gather_epi64(state[k].ht_off, state[k].m_have_tuple, v_next_addr, 0, 1);
        state[k].m_have_tuple = _mm512_kand(_mm512_cmpneq_epi64_mask(state[k].ht_off, v_zero512), state[k].m_have_tuple);
        // return back every time
        state[k].stage = 1;
      }
        break;
    }
    ++k;
  }
  return matches;
}
int64_t bts_smv(tree_t *tree, relation_t *rel, void *output) {
  int64_t matches = 0;
  int32_t new_add = 0, k = 0, done = 0, num, num_temp;
  __mmask8 m_match = 0, m_new_cells = -1, m_valid_bucket = 0, mask[VECTOR_SCALE + 1], m_less, m_more;
  __m512i v_offset = _mm512_set1_epi64(0), v_addr_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(rel->num_tuples * sizeof(tuple_t)), v_tuple_key =
      _mm512_set1_epi64(0), v_base_offset, v_node_key, v_neg_one512 = _mm512_set1_epi64(-1), v_zero512 = _mm512_set1_epi64(0), v_next_addr = _mm512_set1_epi64(0), v_write_index =
      _mm512_set1_epi64(0), v_tree_addr = _mm512_set1_epi64(tree->first_node), v_word_size = _mm512_set1_epi64(WORDSIZE), v_tuple_size = _mm512_set1_epi64(sizeof(tuple_t)),
      v_bucket_size = _mm512_set1_epi64(sizeof(bucket_t)), v_lnext_off = _mm512_set1_epi64(16), v_tuple_payload, v_node_payload, v_rnext_off = _mm512_set1_epi64(24),
      v_payload_off = _mm512_set1_epi64(8);
  chainedtuplebuffer_t *chainedbuf = (chainedtuplebuffer_t *) output;
  tuple_t *join_res = NULL;
  __attribute__((aligned(64)))  uint64_t cur_offset = 0, base_off[16], *ht_pos;

  for (int i = 0; i <= VECTOR_SCALE; ++i) {
    base_off[i] = i * sizeof(tuple_t);
    mask[i] = (1 << i) - 1;
  }
  v_base_offset = _mm512_load_epi64(base_off);
  StateSIMD state[SIMDStateSize + 1];
  // init # of the state
  for (int i = 0; i <= SIMDStateSize; ++i) {
    state[i].stage = 1;
    state[i].m_have_tuple = 0;
    state[i].ht_off = _mm512_set1_epi64(0);
    state[i].payload = _mm512_set1_epi64(0);
    state[i].key = _mm512_set1_epi64(0);
  }
  for (uint64_t cur = 0; 1;) {
    k = (k >= SIMDStateSize) ? 0 : k;
    if (cur >= rel->num_tuples) {
      if (state[k].m_have_tuple == 0 && state[k].stage != 3) {
        ++done;
        state[k].stage = 3;
        ++k;
        continue;
      }
      if ((done >= SIMDStateSize)) {
        if (state[SIMDStateSize].m_have_tuple > 0) {
          k = SIMDStateSize;
          state[SIMDStateSize].stage = 0;
        } else {
          break;
        }
      }
    }
    switch (state[k].stage) {
      case 1: {
///////// step 1: load new tuples' address offsets
// the offset should be within MAX_32INT_
// the tail depends on the number of joins and tuples in each bucket
#if SEQPREFETCH
        _mm_prefetch((char *)(((void *)rel->tuples) + cur_offset + PDIS), _MM_HINT_T0);
        _mm_prefetch((char *)(((void *)rel->tuples) + cur_offset + PDIS + 64), _MM_HINT_T0);
        _mm_prefetch((char *)(((void *)rel->tuples) + cur_offset + PDIS + 128), _MM_HINT_T0);

#endif
        // directly use cur, instead of cur_offset to control the offset to rel.
        // In this case, using step = 16 to gather data, but step is larger
        // than the scale 1,2,4 or 8
        v_offset = _mm512_add_epi64(_mm512_set1_epi64(cur_offset), v_base_offset);
        // count the number of empty tuples
        cur_offset = cur_offset + base_off[VECTOR_SCALE];
        cur = cur + VECTOR_SCALE;
        state[k].m_have_tuple = _mm512_cmpgt_epi64_mask(v_base_offset_upper, v_offset);
        ///// step 2: load new cells from right tuples;
        // maybe need offset within a tuple
        state[k].key = _mm512_mask_i64gather_epi64(state[k].key, state[k].m_have_tuple, v_offset, ((void * )rel->tuples), 1);
        state[k].payload = _mm512_mask_i64gather_epi64(state[k].payload, state[k].m_have_tuple, _mm512_add_epi64(v_offset, v_word_size), ((void * )rel->tuples), 1);

        ///// step 3: load new values from hash tables;
        state[k].ht_off = v_tree_addr;
        state[k].stage = 0;
#if KNL
        _mm512_mask_prefetch_i64gather_pd(
            state[k].ht_off, state[k].m_have_tuple, 0, 1, _MM_HINT_T0);
#elif DIR_PREFETCH
        ht_pos = (uint64_t *) &state[k].ht_off;
        for (int i = 0; i < VECTOR_SCALE; ++i) {
          _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
        }
#else
        m_have_tuple = state[k].m_have_tuple;
        ht_pos = (uint64_t *)&state[k].ht_off;
        for (int i = 0; (i < VECTOR_SCALE) & (m_have_tuple);
            ++i, (m_have_tuple >> 1)) {
          if (m_have_tuple & 1) {
            _mm_prefetch((char *)(ht_pos[i]), _MM_HINT_T0);
          }
        }
#endif
      }
        break;
      case 0: {
        v_node_key = _mm512_mask_i64gather_epi64(v_neg_one512, state[k].m_have_tuple, state[k].ht_off, 0, 1);
        ///// step 4: compare;
        m_match = _mm512_cmpeq_epi64_mask(state[k].key, v_node_key);
        m_match = _mm512_kand(m_match, state[k].m_have_tuple);
        // case: tuple->key == node->key
        new_add = _mm_popcnt_u32(m_match);
        matches += new_add;
        state[k].m_have_tuple = _mm512_kandn(m_match, state[k].m_have_tuple);

        // gather payloads
        v_node_payload = _mm512_mask_i64gather_epi64(v_neg_one512, m_match, _mm512_add_epi64(state[k].ht_off, v_payload_off), 0, 1);

        // to scatter join results
        join_res = cb_next_n_writepos(chainedbuf, new_add);
#if SEQPREFETCH
        _mm_prefetch((char *)(((void *)join_res) + PDIS), _MM_HINT_T0);
        _mm_prefetch((char *)(((void *)join_res) + PDIS + 64), _MM_HINT_T0);
        _mm_prefetch((char *)(((void *)join_res) + PDIS + 128), _MM_HINT_T0);
#endif
        v_write_index = _mm512_mask_expand_epi64(v_zero512, m_match, v_base_offset);
        _mm512_mask_i64scatter_epi64((void * )join_res, m_match, v_write_index, state[k].payload, 1);
        v_write_index = _mm512_add_epi64(v_write_index, v_word_size);
        _mm512_mask_i64scatter_epi64((void * )join_res, m_match, v_write_index, v_node_payload, 1);

        // case: tuple->key != node->key
        m_less = _mm512_cmplt_epi64_mask(state[k].key, v_node_key);
        m_more = _mm512_cmpgt_epi64_mask(state[k].key, v_node_key);
        v_next_addr = _mm512_mask_add_epi64(state[k].ht_off, m_less, state[k].ht_off, v_lnext_off);
        v_next_addr = _mm512_mask_add_epi64(v_next_addr, m_more, state[k].ht_off, v_rnext_off);

        // update next
        state[k].ht_off = _mm512_mask_i64gather_epi64(state[k].ht_off, state[k].m_have_tuple, v_next_addr, 0, 1);
        state[k].m_have_tuple = _mm512_kand(_mm512_cmpneq_epi64_mask(state[k].ht_off, v_zero512), state[k].m_have_tuple);

        num = _mm_popcnt_u32(state[k].m_have_tuple);
#if 1
        if (num == VECTOR_SCALE) {
#if KNL
          _mm512_mask_prefetch_i64gather_pd(
              state[k].ht_off, state[k].m_have_tuple, 0, 1, _MM_HINT_T0);
#else
          ht_pos = (uint64_t *) &state[k].ht_off;
          for (int i = 0; i < VECTOR_SCALE; ++i) {
            _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
          }
#endif
        } else
#endif
        {
          if ((done < SIMDStateSize)) {
            num_temp = _mm_popcnt_u32(state[SIMDStateSize].m_have_tuple);
            if (num + num_temp < VECTOR_SCALE) {
              // compress v
              state[k].ht_off = _mm512_maskz_compress_epi64(state[k].m_have_tuple, state[k].ht_off);
              state[k].key = _mm512_maskz_compress_epi64(state[k].m_have_tuple, state[k].key);
              state[k].payload = _mm512_maskz_compress_epi64(state[k].m_have_tuple, state[k].payload);
              // expand v -> temp
              state[SIMDStateSize].ht_off = _mm512_mask_expand_epi64(state[SIMDStateSize].ht_off, _mm512_knot(state[SIMDStateSize].m_have_tuple), state[k].ht_off);
              state[SIMDStateSize].key = _mm512_mask_expand_epi64(state[SIMDStateSize].key, _mm512_knot(state[SIMDStateSize].m_have_tuple), state[k].key);
              state[SIMDStateSize].payload = _mm512_mask_expand_epi64(state[SIMDStateSize].payload, _mm512_knot(state[SIMDStateSize].m_have_tuple), state[k].payload);
              state[SIMDStateSize].m_have_tuple = mask[num + num_temp];
              state[k].m_have_tuple = 0;
              state[k].stage = 1;

            } else {
              // expand temp -> v
              state[k].ht_off = _mm512_mask_expand_epi64(state[k].ht_off, _mm512_knot(state[k].m_have_tuple), state[SIMDStateSize].ht_off);
              state[k].key = _mm512_mask_expand_epi64(state[k].key, _mm512_knot(state[k].m_have_tuple), state[SIMDStateSize].key);

              state[k].payload = _mm512_mask_expand_epi64(state[k].payload, _mm512_knot(state[k].m_have_tuple), state[SIMDStateSize].payload);
              // compress temp
              state[SIMDStateSize].m_have_tuple = _mm512_kand(state[SIMDStateSize].m_have_tuple, _mm512_knot(mask[VECTOR_SCALE - num]));
              state[SIMDStateSize].ht_off = _mm512_maskz_compress_epi64(state[SIMDStateSize].m_have_tuple, state[SIMDStateSize].ht_off);
              state[SIMDStateSize].key = _mm512_maskz_compress_epi64(state[SIMDStateSize].m_have_tuple, state[SIMDStateSize].key);
              state[SIMDStateSize].payload = _mm512_maskz_compress_epi64(state[SIMDStateSize].m_have_tuple, state[SIMDStateSize].payload);
              state[k].m_have_tuple = mask[VECTOR_SCALE];
              state[SIMDStateSize].m_have_tuple = (state[SIMDStateSize].m_have_tuple >> (VECTOR_SCALE - num));
              state[k].stage = 0;
#if KNL
              _mm512_mask_prefetch_i64gather_pd(
                  state[k].ht_off, state[k].m_have_tuple, 0, 1, _MM_HINT_T0);
#elif DIR_PREFETCH
              ht_pos = (uint64_t *) &state[k].ht_off;
              for (int i = 0; i < VECTOR_SCALE; ++i) {
                _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
              }
#else
              m_have_tuple = state[k].m_have_tuple;
              ht_pos = (uint64_t *)&state[k].ht_off;
              for (int i = 0; (i < VECTOR_SCALE) & (m_have_tuple);
                  ++i, (m_have_tuple >> 1)) {
                if (m_have_tuple & 1) {
                  _mm_prefetch((char *)(ht_pos[i]), _MM_HINT_T0);
                }
              }
#endif
            }
          }
        }
      }
        break;
    }
    ++k;
  }
  return matches;
}
