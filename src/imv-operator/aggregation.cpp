#include "imv-operator/aggregation.hpp"
#include "common/runtime/SIMD.hpp"
#include <stdlib.h>
#include <unistd.h>
#include <map>
using namespace std;
#define CTIMES 0
#define PRINT_HT 0
volatile static char g_lock = 0, g_lock_morse = 0;
volatile static uint64_t total_num = 0, global_curse = 0, global_upper,global_morse_size=0;
typedef int64_t (*AGGFun)(hashtable_t *ht, relation_t *rel, bucket_buffer_t **overflowbuf);
volatile static struct Fun {
  AGGFun fun_ptr;
  char fun_name[8];
} pfun[10];
volatile static int pf_num = 0;
size_t agg_raw(hashtable_t *ht, relation_t *rel, bucket_buffer_t **overflowbuf) {
  uint64_t key, hash_value, new_add = 0;
  tuple_t *dest;
  bucket_t *cur_node, *next_node, *bucket;
  for (size_t cur = 0; cur < rel->num_tuples; ++cur) {
    // step 1: get key
    key = rel->tuples[cur].key;
    // step 2: hash
    hash_value = HASH(key, ht->hash_mask, ht->skip_bits);
    // step 3: search hash bucket
    cur_node = ht->buckets + hash_value;
    next_node = cur_node->next;
    // step 4.1: insert new bucket at the first
    while (true) {
      if (cur_node->count == 0) {
        cur_node->count = 1;
        cur_node->tuples->key = key;
        cur_node->tuples->payload = 1;
        cur_node->next = NULL;
        ++new_add;
        break;
      } else {
        //  step 4.2: update agg
        if (cur_node->tuples->key == key) {
          cur_node->tuples->payload++;
          break;
        } else {
          next_node = cur_node->next;
          if (next_node == NULL) {
            // step 4.3: insert new bucket at the tail
            get_new_bucket(&bucket, overflowbuf);
            cur_node->next = bucket;
            bucket->next = NULL;
            bucket->tuples->key = key;
            bucket->tuples->payload = 1;
            bucket->count = 1;
            ++new_add;
            break;
          } else {
            cur_node = next_node;
          }
        }
      }
    }
  }
  return new_add;
}
size_t agg_amac(hashtable_t *ht, relation_t *rel, bucket_buffer_t **overflowbuf) {
  uint64_t key, hash_value, new_add = 0, cur = 0, end = rel->num_tuples;
  tuple_t *dest;
  bucket_t *cur_node, *next_node, *bucket;
  scalar_state_t state[ScalarStateSize];
  for (int i = 0; i < ScalarStateSize; ++i) {
    state[i].stage = 1;
  }
  int done = 0, k = 0;

  while (done < ScalarStateSize) {
    k = (k >= ScalarStateSize) ? 0 : k;
    switch (state[k].stage) {
      case 1: {
        if (cur >= end) {
          ++done;
          state[k].stage = 3;
          break;
        }
#if SEQPREFETCH
        _mm_prefetch(((char *)(rel->tuples + cur) + PDIS), _MM_HINT_T0);
#endif
        // step 1: get key
        key = rel->tuples[cur].key;
        // step 2: hash
        hash_value = HASH(key, ht->hash_mask, ht->skip_bits);
        state[k].b = ht->buckets + hash_value;
        state[k].stage = 0;
        state[k].key = key;
        ++cur;
        _mm_prefetch((char * )(state[k].b), _MM_HINT_T0);

      }
        break;
      case 0: {
        cur_node = state[k].b;
        if (cur_node->count == 0) {
          cur_node->count = 1;
          cur_node->tuples->key = state[k].key;
          cur_node->tuples->payload = 1;
          cur_node->next = NULL;
          ++new_add;
          state[k].stage = 1;
          --k;
          break;
        } else {
          //  step 4.2: update agg
          if (cur_node->tuples->key == state[k].key) {
            cur_node->tuples->payload++;
            state[k].stage = 1;
            --k;
            break;
          } else {
            next_node = cur_node->next;
            if (next_node == NULL) {
              // step 4.3: insert new bucket at the tail
              get_new_bucket(&bucket, overflowbuf);
              cur_node->next = bucket;
              bucket->next = NULL;
              bucket->tuples->key = state[k].key;
              bucket->tuples->payload = 1;
              bucket->count = 1;
              ++new_add;
              state[k].stage = 1;
              --k;
              break;
            } else {
              state[k].b = next_node;
            }
          }
        }
      }
    }
    ++k;
  }

  return new_add;
}
#define WORDSIZE 8
__m512i v_one = _mm512_set1_epi64(1), v_63 = _mm512_set1_epi64(63), v_all_ones = _mm512_set1_epi64(-1);

void print_vector(string name, __m512i vec) {
  uint64_t* pos = (uint64_t*) &vec;
  cout << name << " : ";
  for (int i = 0; i < VECTOR_SCALE; ++i) {
    cout << pos[i] << " , ";
  }
  cout << endl;
}
void mergeKeys(StateSIMD& state) {
  __m512i v_key = _mm512_mask_blend_epi64(state.m_have_tuple, v_all_ones, state.key);
  __m512i v_conflict = _mm512_conflict_epi64(v_key);
  __mmask8 m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
  m_no_conflict = _mm512_kand(m_no_conflict, state.m_have_tuple);
  __mmask8 m_conflict = _mm512_kandn(m_no_conflict, state.m_have_tuple);
  if(0==m_conflict) return;
  __m512i v_lzeros = _mm512_lzcnt_epi64(v_conflict);
  v_lzeros = _mm512_sub_epi64(v_63, v_lzeros);
  uint64_t* pos_lz = (uint64_t*) &v_lzeros;
  __m512i v_one = _mm512_set1_epi64(1);
  uint64_t* pos_v = (uint64_t*) &v_one;
  for (int i = VECTOR_SCALE - 1; i >= 0; --i) {
    if ((m_conflict & (1 << i))) {
      pos_v[pos_lz[i]] += pos_v[i];
    }
  }
  state.m_have_tuple = m_no_conflict;
  state.key = _mm512_mask_blend_epi64(m_no_conflict, v_all_ones, state.key);
}
size_t agg_imv_old(hashtable_t *ht, relation_t *rel, bucket_buffer_t **overflowbuf) {
  int32_t found = 0, k = 0, done = 0, num, num_temp;
  __attribute__((aligned(64)))       __mmask8 m_to_scatt, m_match = 0, m_new_cells = -1, m_valid_bucket = 0, mask[VECTOR_SCALE + 1], m_to_insert = 0, m_no_conflict;
  __m512i v_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(rel->num_tuples * sizeof(tuple_t)), v_base_offset, v_ht_cell, v_factor = _mm512_set1_epi64(
      ht->hash_mask), v_shift = _mm512_set1_epi64(ht->skip_bits), v_cell_hash, v_neg_one512 = _mm512_set1_epi64(-1), v_zero512 = _mm512_set1_epi64(0), v_write_index =
      _mm512_set1_epi64(0), v_ht_addr = _mm512_set1_epi64((uint64_t) ht->buckets), v_word_size = _mm512_set1_epi64(WORDSIZE), v_tuple_size = _mm512_set1_epi64(sizeof(tuple_t)),
      v_bucket_size = _mm512_set1_epi64(sizeof(bucket_t)), v_next_off = _mm512_set1_epi64(offsetof(bucket_t, next)), v_right_payload, v_payload_off = _mm512_set1_epi64(24), v_addr,
      v_all_ones = _mm512_set1_epi64(-1), v_63 = _mm512_set1_epi64(63), v_conflict, v_one = _mm512_set1_epi64(1), v_new_bucket, v_next, v_lzeros, v_previous, v_key_off =
          _mm512_set1_epi64(offsetof(bucket_t, tuples[0].key));
  __m256i v256_one = _mm256_set1_epi32(1);
  tuple_t *join_res = NULL;
  uint64_t *pos = NULL, *new_bucket = (uint64_t*) &v_new_bucket;
  bucket_t * bucket;
  int tail_add = 0;
  __attribute__((aligned(64)))               uint64_t cur_offset = 0, base_off[16], *ht_pos;
  for (int i = 0; i <= VECTOR_SCALE; ++i) {
    base_off[i] = i * sizeof(tuple_t);
    mask[i] = (1 << i) - 1;
  }
  v_base_offset = _mm512_load_epi64(base_off);
  __attribute__((aligned(64)))               StateSIMD state[SIMDStateSize + 1];
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
    if (UNLIKELY(cur >= rel->num_tuples)) {
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
        // hash the cell values
        v_cell_hash = _mm512_and_epi64(state[k].key, v_factor);
        v_cell_hash = _mm512_srlv_epi64(v_cell_hash, v_shift);
        v_cell_hash = _mm512_mullo_epi64(v_cell_hash, v_bucket_size);
        state[k].ht_off = _mm512_mask_add_epi64(state[k].ht_off, state[k].m_have_tuple, v_cell_hash, v_ht_addr);
        state[k].stage = 2;
        v_prefetch(state[k].ht_off);

//        _mm_prefetch((char * )((*overflowbuf)->buf+(*overflowbuf)->count)+PDIS, _MM_HINT_T0);
//        _mm_prefetch((char * )((*overflowbuf)->buf+(*overflowbuf)->count)+PDIS+64, _MM_HINT_T0);

      }
        break;
      case 2: {
        /////////////////// random access
        // check valid bucket
//        mergeKeys(state[k]);
        v_ht_cell = _mm512_mask_i64gather_epi64(v_neg_one512, state[k].m_have_tuple, state[k].ht_off, 0, 1);
        // inset new nodes
        m_to_insert = _mm512_cmpeq_epi64_mask(v_ht_cell, v_zero512);
        m_to_insert = _mm512_kand(m_to_insert, state[k].m_have_tuple);
        if (m_to_insert == 0) {
          state[k].stage = 0;
          --k;
          break;
        }
        v_addr = _mm512_mask_blend_epi64(state[k].m_have_tuple, v_all_ones, state[k].ht_off);
        v_conflict = _mm512_conflict_epi64(v_addr);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_no_conflict) {
          // write the key , payload, count, next to the nodes
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].key))), state[k].key, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].payload))), v_one, 1);
          _mm512_mask_i64scatter_epi32(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,count))), v256_one, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,next))), v_zero512, 1);

          state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
          found += _mm_popcnt_u32(m_no_conflict);
        }

        num = _mm_popcnt_u32(state[k].m_have_tuple);
        if (num == VECTOR_SCALE || done >= SIMDStateSize) {
          state[k].stage = 0;
          --k;
        } else if (num == 0) {
          state[k].stage = 1;
          --k;
        } else {
          if (LIKELY(done < SIMDStateSize)) {
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
              --k;
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
              v_prefetch(state[k].ht_off);
            }
          }
        }

      }
        break;
      case 0: {
        v_ht_cell = _mm512_mask_i64gather_epi64(v_neg_one512, state[k].m_have_tuple, _mm512_add_epi64(state[k].ht_off, v_tuple_size), 0, 1);  // note the offset of the tuple in %bucket_t%

        ///// step 4: compare;
        m_match = _mm512_cmpeq_epi64_mask(state[k].key, v_ht_cell);
        m_match = _mm512_kand(m_match, state[k].m_have_tuple);
        /// update the aggregators
        v_addr = _mm512_mask_blend_epi64(state[k].m_have_tuple, v_all_ones, state[k].ht_off);
        v_conflict = _mm512_conflict_epi64(v_addr);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_match);
//        if(m_no_conflict!=m_match){
//          mergeKeys(state[k]);
//        }
        // gather and scatter payloads
        v_right_payload = _mm512_mask_i64gather_epi64(v_neg_one512, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_payload_off), 0, 1);
        v_right_payload = _mm512_add_epi64(v_right_payload, v_one);
        _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_payload_off), v_right_payload, 1);

#if 1
        state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
        m_match = _mm512_kandn(m_no_conflict, m_match);
#else
        state[k].m_have_tuple = _mm512_kandn(m_match, state[k].m_have_tuple);
//        m_match=0;
#endif
        // step 7: NOT found, then insert
        v_addr = _mm512_mask_i64gather_epi64(v_all_ones, _mm512_kandn(m_match, state[k].m_have_tuple), _mm512_add_epi64(state[k].ht_off, v_next_off), 0, 1);
        m_to_insert = _mm512_kand(_mm512_kandn(m_match, state[k].m_have_tuple), _mm512_cmpeq_epi64_mask(v_addr, v_zero512));
        v_addr = _mm512_mask_blend_epi64(_mm512_kandn(m_match, state[k].m_have_tuple), v_all_ones, state[k].ht_off);
        v_conflict = _mm512_conflict_epi64(v_addr);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
#if 0
        if (m_no_conflict) {
          for (int i = 0; i < VECTOR_SCALE; ++i) {
            new_bucket[i] = 0;
            if (m_no_conflict & (1 << i)) {
              get_new_bucket(&bucket, overflowbuf);
              new_bucket[i] = bucket;
            }
          }
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].key))), state[k].key, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].payload))), v_one, 1);
//          _mm512_mask_i64scatter_epi32(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,count))), v256_one, 1);
//          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,next))), v_zero512, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_next_off), v_new_bucket, 1);

          found += _mm_popcnt_u32(m_no_conflict);
          state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
          tail_add += _mm_popcnt_u32(m_no_conflict);
        }
#else
        if (m_to_insert) {
          for (int i = 0; i < VECTOR_SCALE; ++i) {
            new_bucket[i] = 0;
            if (m_to_insert & (1 << i)) {
              get_new_bucket(&bucket, overflowbuf);
              new_bucket[i] = bucket;
            }
          }
          _mm512_mask_i64scatter_epi64(0, m_to_insert, _mm512_add_epi64(v_new_bucket, v_key_off), state[k].key, 1);
          _mm512_mask_i64scatter_epi64(0, m_to_insert, _mm512_add_epi64(v_new_bucket, v_payload_off), v_one, 1);
          _mm512_mask_i64scatter_epi64(0, m_to_insert, _mm512_add_epi64(v_new_bucket, v_next_off), v_zero512, 1);

          // conflict-free insert
          v_lzeros = _mm512_lzcnt_epi64(v_conflict);
          v_lzeros = _mm512_sub_epi64(v_63, v_lzeros);
          m_to_scatt = _mm512_kandn(m_no_conflict, m_to_insert);
          v_previous = _mm512_maskz_permutexvar_epi64(m_to_scatt, v_lzeros, v_new_bucket);
          _mm512_mask_i64scatter_epi64(0, m_to_scatt, _mm512_add_epi64(v_previous, v_next_off), v_new_bucket, 1);

          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_next_off), v_new_bucket, 1);
          found += _mm_popcnt_u32(m_to_insert);

          state[k].m_have_tuple = _mm512_kandn(m_to_insert, state[k].m_have_tuple);
        }
#endif
        v_addr = _mm512_mask_i64gather_epi64(v_all_ones, state[k].m_have_tuple, _mm512_add_epi64(state[k].ht_off, v_next_off), 0, 1);
        // the remaining matches, DO NOT get next
        state[k].ht_off = _mm512_mask_blend_epi64(m_match, v_addr, state[k].ht_off);

        num = _mm_popcnt_u32(state[k].m_have_tuple);
#if 1

        if (num == VECTOR_SCALE || done >= SIMDStateSize) {
#if KNL
          _mm512_mask_prefetch_i64gather_pd(
              state[k].ht_off, state[k].m_have_tuple, 0, 1, _MM_HINT_T0);
#else
          ht_pos = (uint64_t *) &state[k].ht_off;
          for (int i = 0; i < VECTOR_SCALE; ++i) {
            _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
          }
#endif
        } else if (num == 0) {
          state[k].stage = 1;
          --k;
          break;
        } else
#endif
        {
          if (LIKELY(done < SIMDStateSize)) {
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
              --k;
              break;
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
#else
              ht_pos = (uint64_t *) &state[k].ht_off;
              for (int i = 0; i < VECTOR_SCALE; ++i) {
                _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
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

  return found;
}
static map<int64_t,int64_t>ctimes_first,ctimes_match,ctimes_tail;
size_t agg_imv_copy(hashtable_t *ht, relation_t *rel, bucket_buffer_t **overflowbuf) {
  int32_t found = 0, k = 0, done = 0, num, num_temp;
  __attribute__((aligned(64)))               __mmask8 m_match = 0, m_new_cells = -1, m_valid_bucket = 0, mask[VECTOR_SCALE + 1], m_to_insert = 0, m_no_conflict;
  __m512i v_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(rel->num_tuples * sizeof(tuple_t)), v_base_offset, v_ht_cell, v_factor = _mm512_set1_epi64(
      ht->hash_mask), v_shift = _mm512_set1_epi64(ht->skip_bits), v_cell_hash, v_neg_one512 = _mm512_set1_epi64(-1), v_zero512 = _mm512_set1_epi64(0), v_write_index =
      _mm512_set1_epi64(0), v_ht_addr = _mm512_set1_epi64((uint64_t) ht->buckets), v_word_size = _mm512_set1_epi64(WORDSIZE), v_tuple_size = _mm512_set1_epi64(sizeof(tuple_t)),
      v_bucket_size = _mm512_set1_epi64(sizeof(bucket_t)), v_next_off = _mm512_set1_epi64(offsetof(bucket_t, next)), v_right_payload, v_addr, v_all_ones = _mm512_set1_epi64(-1),
      v_conflict, v_one = _mm512_set1_epi64(1), v_new_bucket, v_key_off = _mm512_set1_epi64(offsetof(bucket_t, tuples[0].key)), v_lzeros, v_previous, v_payload_off =
          _mm512_set1_epi64(offsetof(bucket_t, tuples[0].payload)), v_count_off = _mm512_set1_epi64(offsetof(bucket_t, count));
  __m256i v256_one = _mm256_set1_epi32(1);
  tuple_t *join_res = NULL;
  uint64_t *pos = NULL, *new_bucket = (uint64_t*) &v_new_bucket;
  bucket_t * bucket;
  int tail_add = 0;
  __attribute__((aligned(64)))               uint64_t cur_offset = 0, base_off[16], *ht_pos;
  for (int i = 0; i <= VECTOR_SCALE; ++i) {
    base_off[i] = i * sizeof(tuple_t);
    mask[i] = (1 << i) - 1;
  }
  v_base_offset = _mm512_load_epi64(base_off);
  __attribute__((aligned(64)))               StateSIMD state[SIMDStateSize + 1];
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
    if (UNLIKELY(cur >= rel->num_tuples)) {
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
        // hash the cell values
        v_cell_hash = _mm512_and_epi64(state[k].key, v_factor);
        v_cell_hash = _mm512_srlv_epi64(v_cell_hash, v_shift);
        v_cell_hash = _mm512_mullo_epi64(v_cell_hash, v_bucket_size);
        state[k].ht_off = _mm512_mask_add_epi64(state[k].ht_off, state[k].m_have_tuple, v_cell_hash, v_ht_addr);
        state[k].stage = 2;
        v_prefetch(state[k].ht_off);
      }
        break;
      case 2: {
        /////////////////// random access
        // check valid bucket
        v_ht_cell = _mm512_mask_i64gather_epi64(v_neg_one512, state[k].m_have_tuple, state[k].ht_off, 0, 1);
        // inset new nodes
        m_to_insert = _mm512_cmpeq_epi64_mask(v_ht_cell, v_zero512);
        m_to_insert = _mm512_kand(m_to_insert, state[k].m_have_tuple);
        if (m_to_insert == 0) {
          state[k].stage = 0;
          --k;
          break;
        }
        v_addr = _mm512_mask_blend_epi64(state[k].m_have_tuple, v_all_ones, state[k].ht_off);
        v_conflict = _mm512_conflict_epi64(v_addr);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_no_conflict) {
          // write the key , payload, count, next to the nodes
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].key))), state[k].key, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].payload))), v_one, 1);
          _mm512_mask_i64scatter_epi32(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,count))), v256_one, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,next))), v_zero512, 1);

          state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
          found += _mm_popcnt_u32(m_no_conflict);
#if CTIMES
          lock(&g_lock);
          ++ctimes_first[_mm_popcnt_u32(m_no_conflict)];
          unlock(&g_lock);
#endif
        }
        state[k].stage = 0;
        --k;
        break;
        num = _mm_popcnt_u32(state[k].m_have_tuple);
        if (num == VECTOR_SCALE || done >= SIMDStateSize) {
          state[k].stage = 0;
          --k;
        } else if (num == 0) {
          state[k].stage = 1;
          --k;
        } else {
          if (LIKELY(done < SIMDStateSize)) {
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
              --k;
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
              --k;
            }
          }
        }

      }
        break;
      case 0: {
        v_ht_cell = _mm512_mask_i64gather_epi64(v_neg_one512, state[k].m_have_tuple, _mm512_add_epi64(state[k].ht_off, v_key_off), 0, 1);  // note the offset of the tuple in %bucket_t%

        ///// step 4: compare;
        m_match = _mm512_cmpeq_epi64_mask(state[k].key, v_ht_cell);
        m_match = _mm512_kand(m_match, state[k].m_have_tuple);
#if 0
        if (true) {
          /// update the aggregators
          v_addr = _mm512_mask_blend_epi64(m_match, v_all_ones, state[k].ht_off);
          v_conflict = _mm512_conflict_epi64(v_addr);
          m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
          m_no_conflict = _mm512_kand(m_no_conflict, m_match);
////
         Vec8u u_one = Vec8u(1);
         if(m_no_conflict){
          __mmask8 m_conflict = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
          Vec8u u_lzeros = _mm512_lzcnt_epi64(v_conflict);
          for (int i = VECTOR_SCALE - 1; i >= 0; --i) {
            if ((m_conflict & (1 << i))) {
              u_one.entry[63 - u_lzeros.entry[i]] += u_one.entry[i];
            }
          }
         }
    ////
#if CTIMES
         lock(&g_lock);
          ctimes_match[_mm_popcnt_u32(m_match)-_mm_popcnt_u32(m_no_conflict)]++;
          unlock(&g_lock);
#endif
          // gather and scatter payloads
          v_right_payload = _mm512_mask_i64gather_epi64(v_neg_one512, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_payload_off), 0, 1);
          v_right_payload = _mm512_add_epi64(v_right_payload, u_one.reg);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_payload_off), v_right_payload, 1);
          m_no_conflict = m_match;
          state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
          m_match = _mm512_kandn(m_no_conflict, m_match);
        }
#else
        /// update the aggregators
        v_addr = _mm512_mask_blend_epi64(state[k].m_have_tuple, v_all_ones, state[k].ht_off);
        v_conflict = _mm512_conflict_epi64(v_addr);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_match);

        // gather and scatter payloads
        v_right_payload = _mm512_mask_i64gather_epi64(v_neg_one512, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_payload_off), 0, 1);
        v_right_payload = _mm512_add_epi64(v_right_payload, v_one);
        _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_payload_off), v_right_payload, 1);

        state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
        m_match = _mm512_kandn(m_no_conflict, m_match);
#endif
        // step 7: NOT found, then insert
        v_addr = _mm512_mask_i64gather_epi64(v_all_ones, _mm512_kandn(m_match, state[k].m_have_tuple), _mm512_add_epi64(state[k].ht_off, v_next_off), 0, 1);
        m_to_insert = _mm512_kand(_mm512_kandn(m_match, state[k].m_have_tuple), _mm512_cmpeq_epi64_mask(v_addr, v_zero512));
        v_addr = _mm512_mask_blend_epi64(_mm512_kandn(m_match, state[k].m_have_tuple), v_all_ones, state[k].ht_off);
        v_conflict = _mm512_conflict_epi64(v_addr);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
#if CTIMES
          lock(&g_lock);
          ctimes_tail[_mm_popcnt_u32(m_to_insert) - _mm_popcnt_u32(m_no_conflict)]++;
          unlock(&g_lock);
#endif
        if (m_no_conflict) {
          for (int i = 0; i < VECTOR_SCALE; ++i) {
            new_bucket[i] = 0;
            if (m_no_conflict & (1 << i)) {
              get_new_bucket(&bucket, overflowbuf);
              new_bucket[i] = bucket;
            }
          }
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].key))), state[k].key, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].payload))), v_one, 1);
//          _mm512_mask_i64scatter_epi32(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,count))), v256_one, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,next))), v_zero512, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_next_off), v_new_bucket, 1);

          found += _mm_popcnt_u32(m_no_conflict);
          state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);

        }

        v_addr = _mm512_mask_i64gather_epi64(v_all_ones, state[k].m_have_tuple, _mm512_add_epi64(state[k].ht_off, v_next_off), 0, 1);
        // the remaining matches, DO NOT get next
        state[k].ht_off = _mm512_mask_blend_epi64(m_match, v_addr, state[k].ht_off);

        num = _mm_popcnt_u32(state[k].m_have_tuple);
#if 1

        if (num == VECTOR_SCALE || done >= SIMDStateSize) {
          v_prefetch(state[k].ht_off);
        } else if (num == 0) {
          state[k].stage = 1;
          --k;
          break;
        } else
#endif
        {
          if (LIKELY(done < SIMDStateSize)) {
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
              --k;
              break;
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
//              mergeKeys(state[k]);
              v_prefetch(state[k].ht_off);

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

size_t agg_imv(hashtable_t *ht, relation_t *rel, bucket_buffer_t **overflowbuf) {
  int32_t found = 0, k = 0, done = 0, num, num_temp;
  __attribute__((aligned(64)))               __mmask8 m_match = 0, m_new_cells = -1, m_valid_bucket = 0, mask[VECTOR_SCALE + 1], m_to_insert = 0, m_no_conflict;
  __m512i v_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(rel->num_tuples * sizeof(tuple_t)), v_base_offset, v_ht_cell, v_factor = _mm512_set1_epi64(
      ht->hash_mask), v_shift = _mm512_set1_epi64(ht->skip_bits), v_cell_hash, v_neg_one512 = _mm512_set1_epi64(-1), v_zero512 = _mm512_set1_epi64(0), v_write_index =
      _mm512_set1_epi64(0), v_ht_addr = _mm512_set1_epi64((uint64_t) ht->buckets), v_word_size = _mm512_set1_epi64(WORDSIZE), v_tuple_size = _mm512_set1_epi64(sizeof(tuple_t)),
      v_bucket_size = _mm512_set1_epi64(sizeof(bucket_t)), v_next_off = _mm512_set1_epi64(offsetof(bucket_t, next)), v_right_payload, v_payload_off = _mm512_set1_epi64(24), v_addr,
      v_all_ones = _mm512_set1_epi64(-1), v_conflict, v_one = _mm512_set1_epi64(1), v_new_bucket;
  __m256i v256_one = _mm256_set1_epi32(1);
  tuple_t *join_res = NULL;
  uint64_t *pos = NULL, *new_bucket = (uint64_t*) &v_new_bucket;
  bucket_t * bucket;
  int tail_add = 0;
  __attribute__((aligned(64)))               uint64_t cur_offset = 0, base_off[16], *ht_pos;
  for (int i = 0; i <= VECTOR_SCALE; ++i) {
    base_off[i] = i * sizeof(tuple_t);
    mask[i] = (1 << i) - 1;
  }
  v_base_offset = _mm512_load_epi64(base_off);
  __attribute__((aligned(64)))               StateSIMD state[SIMDStateSize + 1];
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
    if (UNLIKELY(cur >= rel->num_tuples)) {
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
        // hash the cell values
        v_cell_hash = _mm512_and_epi64(state[k].key, v_factor);
        v_cell_hash = _mm512_srlv_epi64(v_cell_hash, v_shift);
        v_cell_hash = _mm512_mullo_epi64(v_cell_hash, v_bucket_size);
        state[k].ht_off = _mm512_mask_add_epi64(state[k].ht_off, state[k].m_have_tuple, v_cell_hash, v_ht_addr);
        state[k].stage = 2;
#if KNL
        _mm512_mask_prefetch_i64gather_pd(
            state[k].ht_off, state[k].m_have_tuple, 0, 1, _MM_HINT_T0);
#else
        ht_pos = (uint64_t *) &state[k].ht_off;
        for (int i = 0; i < VECTOR_SCALE; ++i) {
          _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
        }
#endif
//        _mm_prefetch((char * )((*overflowbuf)->buf+(*overflowbuf)->count)+PDIS, _MM_HINT_T0);
//        _mm_prefetch((char * )((*overflowbuf)->buf+(*overflowbuf)->count)+PDIS+64, _MM_HINT_T0);

      }
        break;
      case 2: {
        /////////////////// random access
        // check valid bucket
//        mergeKeys(state[k]);
        v_ht_cell = _mm512_mask_i64gather_epi64(v_neg_one512, state[k].m_have_tuple, state[k].ht_off, 0, 1);
        // inset new nodes
        m_to_insert = _mm512_cmpeq_epi64_mask(v_ht_cell, v_zero512);
        m_to_insert = _mm512_kand(m_to_insert, state[k].m_have_tuple);
        if (m_to_insert == 0) {
          state[k].stage = 0;
          --k;
          break;
        }
        v_addr = _mm512_mask_blend_epi64(state[k].m_have_tuple, v_all_ones, state[k].ht_off);
        v_conflict = _mm512_conflict_epi64(v_addr);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_no_conflict) {
          // write the key , payload, count, next to the nodes
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].key))), state[k].key, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].payload))), v_one, 1);
          _mm512_mask_i64scatter_epi32(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,count))), v256_one, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,next))), v_zero512, 1);

          state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
          found += _mm_popcnt_u32(m_no_conflict);
        }

        num = _mm_popcnt_u32(state[k].m_have_tuple);
        if (num == VECTOR_SCALE || done >= SIMDStateSize) {
          state[k].stage = 0;
          --k;
        } else if (num == 0) {
          state[k].stage = 1;
          --k;
        } else {
          if (LIKELY(done < SIMDStateSize)) {
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
              --k;
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
              --k;
            }
          }
        }

      }
        break;
      case 0: {
        v_ht_cell = _mm512_mask_i64gather_epi64(v_neg_one512, state[k].m_have_tuple, _mm512_add_epi64(state[k].ht_off, v_tuple_size), 0, 1);  // note the offset of the tuple in %bucket_t%

        ///// step 4: compare;
        m_match = _mm512_cmpeq_epi64_mask(state[k].key, v_ht_cell);
        m_match = _mm512_kand(m_match, state[k].m_have_tuple);

        /// update the aggregators
        v_addr = _mm512_mask_blend_epi64(state[k].m_have_tuple, v_all_ones, state[k].ht_off);
        v_conflict = _mm512_conflict_epi64(v_addr);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_match);

        // gather and scatter payloads
        v_right_payload = _mm512_mask_i64gather_epi64(v_neg_one512, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_payload_off), 0, 1);
        v_right_payload = _mm512_add_epi64(v_right_payload, v_one);
        _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_payload_off), v_right_payload, 1);

        state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
        m_match = _mm512_kandn(m_no_conflict, m_match);

        // step 7: NOT found, then insert
        v_addr = _mm512_mask_i64gather_epi64(v_all_ones, _mm512_kandn(m_match, state[k].m_have_tuple), _mm512_add_epi64(state[k].ht_off, v_next_off), 0, 1);
        m_to_insert = _mm512_kand(_mm512_kandn(m_match, state[k].m_have_tuple), _mm512_cmpeq_epi64_mask(v_addr, v_zero512));
        v_addr = _mm512_mask_blend_epi64(_mm512_kandn(m_match, state[k].m_have_tuple), v_all_ones, state[k].ht_off);
        v_conflict = _mm512_conflict_epi64(v_addr);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_no_conflict) {
          for (int i = 0; i < VECTOR_SCALE; ++i) {
            new_bucket[i] = 0;
            if (m_no_conflict & (1 << i)) {
              get_new_bucket(&bucket, overflowbuf);
              new_bucket[i] = bucket;
            }
          }
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].key))), state[k].key, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].payload))), v_one, 1);
          _mm512_mask_i64scatter_epi32(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,count))), v256_one, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,next))), v_zero512, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_next_off), v_new_bucket, 1);

          found += _mm_popcnt_u32(m_no_conflict);
          state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
        }

        v_addr = _mm512_mask_i64gather_epi64(v_all_ones, state[k].m_have_tuple, _mm512_add_epi64(state[k].ht_off, v_next_off), 0, 1);
        // the remaining matches, DO NOT get next
        state[k].ht_off = _mm512_mask_blend_epi64(m_match, v_addr, state[k].ht_off);

        num = _mm_popcnt_u32(state[k].m_have_tuple);
#if 1

        if (num == VECTOR_SCALE || done >= SIMDStateSize) {
#if KNL
          _mm512_mask_prefetch_i64gather_pd(
              state[k].ht_off, state[k].m_have_tuple, 0, 1, _MM_HINT_T0);
#else
          ht_pos = (uint64_t *) &state[k].ht_off;
          for (int i = 0; i < VECTOR_SCALE; ++i) {
            _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
          }
#endif
        } else if (num == 0) {
          state[k].stage = 1;
          --k;
          break;
        } else
#endif
        {
          if (LIKELY(done < SIMDStateSize)) {
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
              --k;
              break;
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
#else
              ht_pos = (uint64_t *) &state[k].ht_off;
              for (int i = 0; i < VECTOR_SCALE; ++i) {
                _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
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

  return found;
}

size_t agg_DVA(hashtable_t *ht, relation_t *rel, bucket_buffer_t **overflowbuf) {
  int32_t found = 0, k = 0, done = 0, num, num_temp;
  __attribute__((aligned(64)))               __mmask8 m_match = 0, m_new_cells = -1, m_valid_bucket = 0, mask[VECTOR_SCALE + 1], m_to_insert = 0, m_no_conflict;
  __m512i v_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(rel->num_tuples * sizeof(tuple_t)), v_base_offset, v_ht_cell, v_factor = _mm512_set1_epi64(
      ht->hash_mask), v_shift = _mm512_set1_epi64(ht->skip_bits), v_cell_hash, v_neg_one512 = _mm512_set1_epi64(-1), v_zero512 = _mm512_set1_epi64(0), v_write_index =
      _mm512_set1_epi64(0), v_ht_addr = _mm512_set1_epi64((uint64_t) ht->buckets), v_word_size = _mm512_set1_epi64(WORDSIZE), v_tuple_size = _mm512_set1_epi64(sizeof(tuple_t)),
      v_bucket_size = _mm512_set1_epi64(sizeof(bucket_t)), v_next_off = _mm512_set1_epi64(offsetof(bucket_t, next)), v_right_payload, v_payload_off = _mm512_set1_epi64(24), v_addr,
      v_all_ones = _mm512_set1_epi64(-1), v_conflict, v_one = _mm512_set1_epi64(1), v_new_bucket;
  __m256i v256_one = _mm256_set1_epi32(1);
  tuple_t *join_res = NULL;
  uint64_t *pos = NULL, *new_bucket = (uint64_t*) &v_new_bucket;
  bucket_t * bucket;
  int tail_add = 0;
  __attribute__((aligned(64)))               uint64_t cur_offset = 0, base_off[16], *ht_pos;
  for (int i = 0; i <= VECTOR_SCALE; ++i) {
    base_off[i] = i * sizeof(tuple_t);
    mask[i] = (1 << i) - 1;
  }
  v_base_offset = _mm512_load_epi64(base_off);
  __attribute__((aligned(64)))               StateSIMD state[SIMDStateSize + 1];
  // init # of the state
  for (int i = 0; i <= SIMDStateSize; ++i) {
    state[i].stage = 1;
    state[i].m_have_tuple = 0;
    state[i].ht_off = _mm512_set1_epi64(0);
    state[i].payload = _mm512_set1_epi64(0);
    state[i].key = _mm512_set1_epi64(0);
  }
  for (uint64_t cur = 0; (cur < rel->num_tuples) || (done < SIMDStateSize);) {
    k = (k >= SIMDStateSize) ? 0 : k;
    if (UNLIKELY(cur >= rel->num_tuples)) {
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
        // hash the cell values
        v_cell_hash = _mm512_and_epi64(state[k].key, v_factor);
        v_cell_hash = _mm512_srlv_epi64(v_cell_hash, v_shift);
        v_cell_hash = _mm512_mullo_epi64(v_cell_hash, v_bucket_size);
        state[k].ht_off = _mm512_mask_add_epi64(state[k].ht_off, state[k].m_have_tuple, v_cell_hash, v_ht_addr);
        state[k].stage = 2;
#if KNL
        _mm512_mask_prefetch_i64gather_pd(
            state[k].ht_off, state[k].m_have_tuple, 0, 1, _MM_HINT_T0);
#else
        ht_pos = (uint64_t *) &state[k].ht_off;
        for (int i = 0; i < VECTOR_SCALE; ++i) {
          _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
        }
#endif
        _mm_prefetch((char * )((*overflowbuf)->buf+(*overflowbuf)->count)+PDIS, _MM_HINT_T0);
        _mm_prefetch((char * )((*overflowbuf)->buf+(*overflowbuf)->count)+PDIS+64, _MM_HINT_T0);

      }
        break;
      case 2: {
        /////////////////// random access
        // check valid bucket
        v_ht_cell = _mm512_mask_i64gather_epi64(v_neg_one512, state[k].m_have_tuple, state[k].ht_off, 0, 1);
        // inset new nodes
        m_to_insert = _mm512_cmpeq_epi64_mask(v_ht_cell, v_zero512);
        m_to_insert = _mm512_kand(m_to_insert, state[k].m_have_tuple);

        v_addr = _mm512_mask_blend_epi64(state[k].m_have_tuple, v_all_ones, state[k].ht_off);
        v_conflict = _mm512_conflict_epi64(v_addr);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_no_conflict) {
          // write the key , payload, count, next to the nodes
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].key))), state[k].key, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].payload))), v_one, 1);
          _mm512_mask_i64scatter_epi32(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,count))), v256_one, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,next))), v_zero512, 1);

          state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
          found += _mm_popcnt_u32(m_no_conflict);
        }

        state[k].stage = 0;
      }
        break;
      case 0: {
        v_ht_cell = _mm512_mask_i64gather_epi64(v_neg_one512, state[k].m_have_tuple, _mm512_add_epi64(state[k].ht_off, v_tuple_size), 0, 1);  // note the offset of the tuple in %bucket_t%

        ///// step 4: compare;
        m_match = _mm512_cmpeq_epi64_mask(state[k].key, v_ht_cell);
        m_match = _mm512_kand(m_match, state[k].m_have_tuple);

        /// update the aggregators
        v_addr = _mm512_mask_blend_epi64(state[k].m_have_tuple, v_all_ones, state[k].ht_off);
        v_conflict = _mm512_conflict_epi64(v_addr);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_match);

        // gather and scatter payloads
        v_right_payload = _mm512_mask_i64gather_epi64(v_neg_one512, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_payload_off), 0, 1);
        v_right_payload = _mm512_add_epi64(v_right_payload, v_one);
        _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_payload_off), v_right_payload, 1);

        state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
        m_match = _mm512_kandn(m_no_conflict, m_match);

        // step 7: NOT found, then insert
        v_addr = _mm512_mask_i64gather_epi64(v_all_ones, _mm512_kandn(m_match, state[k].m_have_tuple), _mm512_add_epi64(state[k].ht_off, v_next_off), 0, 1);
        m_to_insert = _mm512_kand(_mm512_kandn(m_match, state[k].m_have_tuple), _mm512_cmpeq_epi64_mask(v_addr, v_zero512));
        v_addr = _mm512_mask_blend_epi64(_mm512_kandn(m_match, state[k].m_have_tuple), v_all_ones, state[k].ht_off);
        v_conflict = _mm512_conflict_epi64(v_addr);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_no_conflict) {
          for (int i = 0; i < VECTOR_SCALE; ++i) {
            new_bucket[i] = 0;
            if (m_no_conflict & (1 << i)) {
              get_new_bucket(&bucket, overflowbuf);
              new_bucket[i] = bucket;
            }
          }
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].key))), state[k].key, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].payload))), v_one, 1);
          _mm512_mask_i64scatter_epi32(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,count))), v256_one, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,next))), v_zero512, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_next_off), v_new_bucket, 1);

          found += _mm_popcnt_u32(m_no_conflict);
          state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
          tail_add += _mm_popcnt_u32(m_no_conflict);
        }

        v_addr = _mm512_mask_i64gather_epi64(v_all_ones, state[k].m_have_tuple, _mm512_add_epi64(state[k].ht_off, v_next_off), 0, 1);
        // the remaining matches, DO NOT get next
        state[k].ht_off = _mm512_mask_blend_epi64(m_match, v_addr, state[k].ht_off);

        if (0 == state[k].m_have_tuple) {
          state[k].stage = 1;
          --k;
        } else {
          ht_pos = (uint64_t *) &state[k].ht_off;
          for (int i = 0; i < VECTOR_SCALE; ++i) {
            _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
          }
        }
      }
        break;
    }
    ++k;
  }

  return found;
}
size_t agg_FVA(hashtable_t *ht, relation_t *rel, bucket_buffer_t **overflowbuf) {
  int32_t new_add = 0, k = 0, done = 0, num, num_temp, found = 0;
  __attribute__((aligned(64)))               __mmask8 m_match = 0, m_new_cells = -1, m_valid_bucket = 0, mask[VECTOR_SCALE + 1], m_to_insert = 0, m_no_conflict, m_full = -1;
  __m512i v_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(rel->num_tuples * sizeof(tuple_t)), v_base_offset, v_ht_cell, v_factor = _mm512_set1_epi64(
      ht->hash_mask), v_shift = _mm512_set1_epi64(ht->skip_bits), v_cell_hash, v_neg_one512 = _mm512_set1_epi64(-1), v_zero512 = _mm512_set1_epi64(0), v_write_index =
      _mm512_set1_epi64(0), v_ht_addr = _mm512_set1_epi64((uint64_t) ht->buckets), v_word_size = _mm512_set1_epi64(WORDSIZE), v_tuple_size = _mm512_set1_epi64(sizeof(tuple_t)),
      v_bucket_size = _mm512_set1_epi64(sizeof(bucket_t)), v_next_off = _mm512_set1_epi64(offsetof(bucket_t, next)), v_right_payload, v_payload_off = _mm512_set1_epi64(24), v_addr,
      v_all_ones = _mm512_set1_epi64(-1), v_conflict, v_one = _mm512_set1_epi64(1), v_new_bucket;
  __m256i v256_one = _mm256_set1_epi32(1);
  tuple_t *join_res = NULL;
  uint64_t *pos = NULL, *new_bucket = (uint64_t*) &v_new_bucket;
  bucket_t * bucket;
  int tail_add = 0;
  __attribute__((aligned(64)))               uint64_t cur_offset = 0, base_off[16], *ht_pos;
  for (int i = 0; i <= VECTOR_SCALE; ++i) {
    base_off[i] = i * sizeof(tuple_t);
    mask[i] = (1 << i) - 1;
  }
  v_base_offset = _mm512_load_epi64(base_off);
  __attribute__((aligned(64)))               StateSIMD state[SIMDStateSize];
  // init # of the state
  for (int i = 0; i < SIMDStateSize; ++i) {
    state[i].stage = 1;
    state[i].m_have_tuple = 0;
    state[i].ht_off = _mm512_set1_epi64(0);
    state[i].payload = _mm512_set1_epi64(0);
    state[i].key = _mm512_set1_epi64(0);
  }
  for (uint64_t cur = 0; (cur < rel->num_tuples) || (done < SIMDStateSize);) {
    k = (k >= SIMDStateSize) ? 0 : k;
    if (UNLIKELY(cur >= rel->num_tuples)) {
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
        // hash the cell values
        v_cell_hash = _mm512_and_epi64(state[k].key, v_factor);
        v_cell_hash = _mm512_srlv_epi64(v_cell_hash, v_shift);
        v_cell_hash = _mm512_mullo_epi64(v_cell_hash, v_bucket_size);
        state[k].ht_off = _mm512_mask_add_epi64(state[k].ht_off, m_new_cells, v_cell_hash, v_ht_addr);
        state[k].stage = 2;
#if KNL
        _mm512_mask_prefetch_i64gather_pd(
            state[k].ht_off, state[k].m_have_tuple, 0, 1, _MM_HINT_T0);
#else
        ht_pos = (uint64_t *) &state[k].ht_off;
        for (int i = 0; i < VECTOR_SCALE; ++i) {
          _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
        }
#endif
      }
        break;
      case 2: {
        /////////////////// random access
        // check valid bucket
        v_ht_cell = _mm512_mask_i64gather_epi64(v_neg_one512, state[k].m_have_tuple, state[k].ht_off, 0, 1);
        // inset new nodes
        m_to_insert = _mm512_cmpeq_epi64_mask(v_ht_cell, v_zero512);
        m_to_insert = _mm512_kand(m_to_insert, state[k].m_have_tuple);

        v_addr = _mm512_mask_blend_epi64(state[k].m_have_tuple, v_all_ones, state[k].ht_off);
        v_conflict = _mm512_conflict_epi64(v_addr);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_no_conflict) {
          // write the key , payload, count, next to the nodes
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].key))), state[k].key, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].payload))), v_one, 1);
          _mm512_mask_i64scatter_epi32(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,count))), v256_one, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,next))), v_zero512, 1);

          state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
          found += _mm_popcnt_u32(m_no_conflict);
        }
        if (m_full == state[k].m_have_tuple || cur >= rel->num_tuples) {
          state[k].stage = 0;
        } else {
          state[k].stage = 1;
        }
        --k;
      }
        break;
      case 0: {
        v_ht_cell = _mm512_mask_i64gather_epi64(v_neg_one512, state[k].m_have_tuple, _mm512_add_epi64(state[k].ht_off, v_tuple_size), 0, 1);  // note the offset of the tuple in %bucket_t%

        ///// step 4: compare;
        m_match = _mm512_cmpeq_epi64_mask(state[k].key, v_ht_cell);
        m_match = _mm512_kand(m_match, state[k].m_have_tuple);

        /// update the aggregators
        v_addr = _mm512_mask_blend_epi64(state[k].m_have_tuple, v_all_ones, state[k].ht_off);
        v_conflict = _mm512_conflict_epi64(v_addr);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_match);

        // gather and scatter payloads
        v_right_payload = _mm512_mask_i64gather_epi64(v_neg_one512, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_payload_off), 0, 1);
        v_right_payload = _mm512_add_epi64(v_right_payload, v_one);
        _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_payload_off), v_right_payload, 1);

        state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
        m_match = _mm512_kandn(m_no_conflict, m_match);

        // step 7: NOT found, then insert
        v_addr = _mm512_mask_i64gather_epi64(v_all_ones, _mm512_kandn(m_match, state[k].m_have_tuple), _mm512_add_epi64(state[k].ht_off, v_next_off), 0, 1);
        m_to_insert = _mm512_kand(_mm512_kandn(m_match, state[k].m_have_tuple), _mm512_cmpeq_epi64_mask(v_addr, v_zero512));
        v_addr = _mm512_mask_blend_epi64(_mm512_kandn(m_match, state[k].m_have_tuple), v_all_ones, state[k].ht_off);
        v_conflict = _mm512_conflict_epi64(v_addr);
        m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
        m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
        if (m_no_conflict) {
          for (int i = 0; i < VECTOR_SCALE; ++i) {
            new_bucket[i] = 0;
            if (m_no_conflict & (1 << i)) {
              get_new_bucket(&bucket, overflowbuf);
              new_bucket[i] = bucket;
            }
          }
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].key))), state[k].key, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].payload))), v_one, 1);
          _mm512_mask_i64scatter_epi32(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,count))), v256_one, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,next))), v_zero512, 1);
          _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_next_off), v_new_bucket, 1);

          found += _mm_popcnt_u32(m_no_conflict);
          state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
          tail_add += _mm_popcnt_u32(m_no_conflict);
        }

        v_addr = _mm512_mask_i64gather_epi64(v_all_ones, state[k].m_have_tuple, _mm512_add_epi64(state[k].ht_off, v_next_off), 0, 1);
        // the remaining matches, DO NOT get next
        state[k].ht_off = _mm512_mask_blend_epi64(m_match, v_addr, state[k].ht_off);

        if (m_full == state[k].m_have_tuple) {
          ht_pos = (uint64_t *) &state[k].ht_off;
          for (int i = 0; i < VECTOR_SCALE; ++i) {
            _mm_prefetch((char * )(ht_pos[i]), _MM_HINT_T0);
          }
        } else {
          state[k].stage = 1;
          --k;
        }
      }
        break;
    }
    ++k;
  }

  return found;
}
size_t agg_SIMD(hashtable_t *ht, relation_t *rel, bucket_buffer_t **overflowbuf) {
  int32_t new_add = 0, k = 0, done = 0, num, num_temp, found = 0;
  __attribute__((aligned(64)))               __mmask8 m_match = 0, m_new_cells = -1, m_valid_bucket = 0, mask[VECTOR_SCALE + 1], m_to_insert = 0, m_no_conflict;
  __m512i v_offset = _mm512_set1_epi64(0), v_base_offset_upper = _mm512_set1_epi64(rel->num_tuples * sizeof(tuple_t)), v_base_offset, v_ht_cell, v_factor = _mm512_set1_epi64(
      ht->hash_mask), v_shift = _mm512_set1_epi64(ht->skip_bits), v_cell_hash, v_neg_one512 = _mm512_set1_epi64(-1), v_zero512 = _mm512_set1_epi64(0), v_write_index =
      _mm512_set1_epi64(0), v_ht_addr = _mm512_set1_epi64((uint64_t) ht->buckets), v_word_size = _mm512_set1_epi64(WORDSIZE), v_tuple_size = _mm512_set1_epi64(sizeof(tuple_t)),
      v_bucket_size = _mm512_set1_epi64(sizeof(bucket_t)), v_next_off = _mm512_set1_epi64(offsetof(bucket_t, next)), v_right_payload, v_payload_off = _mm512_set1_epi64(24), v_addr,
      v_all_ones = _mm512_set1_epi64(-1), v_conflict, v_one = _mm512_set1_epi64(1), v_new_bucket;
  __m256i v256_one = _mm256_set1_epi32(1);
  tuple_t *join_res = NULL;
  uint64_t *pos = NULL, *new_bucket = (uint64_t*) &v_new_bucket;
  bucket_t * bucket;
  int tail_add = 0;
  __attribute__((aligned(64)))               uint64_t cur_offset = 0, base_off[16], *ht_pos;
  for (int i = 0; i <= VECTOR_SCALE; ++i) {
    base_off[i] = i * sizeof(tuple_t);
    mask[i] = (1 << i) - 1;
  }
  v_base_offset = _mm512_load_epi64(base_off);
  __attribute__((aligned(64)))               StateSIMD state[1];
  // init # of the state
  for (int i = 0; i < 1; ++i) {
    state[i].stage = 1;
    state[i].m_have_tuple = 0;
    state[i].ht_off = _mm512_set1_epi64(0);
    state[i].payload = _mm512_set1_epi64(0);
    state[i].key = _mm512_set1_epi64(0);
  }
  k = 0;
  for (uint64_t cur = 0; cur < rel->num_tuples || state[k].m_have_tuple;) {

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
    // hash the cell values
    v_cell_hash = _mm512_and_epi64(state[k].key, v_factor);
    v_cell_hash = _mm512_srlv_epi64(v_cell_hash, v_shift);
    v_cell_hash = _mm512_mullo_epi64(v_cell_hash, v_bucket_size);
    state[k].ht_off = _mm512_mask_add_epi64(state[k].ht_off, m_new_cells, v_cell_hash, v_ht_addr);

    /////////////////// random access
    // check valid bucket
    v_ht_cell = _mm512_mask_i64gather_epi64(v_neg_one512, state[k].m_have_tuple, state[k].ht_off, 0, 1);
    // inset new nodes
    m_to_insert = _mm512_cmpeq_epi64_mask(v_ht_cell, v_zero512);
    m_to_insert = _mm512_kand(m_to_insert, state[k].m_have_tuple);

    v_addr = _mm512_mask_blend_epi64(state[k].m_have_tuple, v_all_ones, state[k].ht_off);
    v_conflict = _mm512_conflict_epi64(v_addr);
    m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
    m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
    if (m_no_conflict) {
      // write the key , payload, count, next to the nodes
      _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].key))), state[k].key, 1);
      _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].payload))), v_one, 1);
      _mm512_mask_i64scatter_epi32(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,count))), v256_one, 1);
      _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_addr,_mm512_set1_epi64(offsetof(bucket_t,next))), v_zero512, 1);

      state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
      found += _mm_popcnt_u32(m_no_conflict);
    }

    v_ht_cell = _mm512_mask_i64gather_epi64(v_neg_one512, state[k].m_have_tuple, _mm512_add_epi64(state[k].ht_off, v_tuple_size), 0, 1);  // note the offset of the tuple in %bucket_t%

    ///// step 4: compare;
    m_match = _mm512_cmpeq_epi64_mask(state[k].key, v_ht_cell);
    m_match = _mm512_kand(m_match, state[k].m_have_tuple);

    /// update the aggregators
    v_addr = _mm512_mask_blend_epi64(state[k].m_have_tuple, v_all_ones, state[k].ht_off);
    v_conflict = _mm512_conflict_epi64(v_addr);
    m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
    m_no_conflict = _mm512_kand(m_no_conflict, m_match);

    // gather and scatter payloads
    v_right_payload = _mm512_mask_i64gather_epi64(v_neg_one512, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_payload_off), 0, 1);
    v_right_payload = _mm512_add_epi64(v_right_payload, v_one);
    _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_payload_off), v_right_payload, 1);

    state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
    m_match = _mm512_kandn(m_no_conflict, m_match);

    // step 7: NOT found, then insert
    v_addr = _mm512_mask_i64gather_epi64(v_all_ones, _mm512_kandn(m_match, state[k].m_have_tuple), _mm512_add_epi64(state[k].ht_off, v_next_off), 0, 1);
    m_to_insert = _mm512_kand(_mm512_kandn(m_match, state[k].m_have_tuple), _mm512_cmpeq_epi64_mask(v_addr, v_zero512));
    v_addr = _mm512_mask_blend_epi64(_mm512_kandn(m_match, state[k].m_have_tuple), v_all_ones, state[k].ht_off);
    v_conflict = _mm512_conflict_epi64(v_addr);
    m_no_conflict = _mm512_testn_epi64_mask(v_conflict, v_all_ones);
    m_no_conflict = _mm512_kand(m_no_conflict, m_to_insert);
    if (m_no_conflict) {
      for (int i = 0; i < VECTOR_SCALE; ++i) {
        new_bucket[i] = 0;
        if (m_no_conflict & (1 << i)) {
          get_new_bucket(&bucket, overflowbuf);
          new_bucket[i] = bucket;
        }
      }
      _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].key))), state[k].key, 1);
      _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,tuples[0].payload))), v_one, 1);
      _mm512_mask_i64scatter_epi32(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,count))), v256_one, 1);
      _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(v_new_bucket,_mm512_set1_epi64(offsetof(bucket_t,next))), v_zero512, 1);
      _mm512_mask_i64scatter_epi64(0, m_no_conflict, _mm512_add_epi64(state[k].ht_off, v_next_off), v_new_bucket, 1);

      found += _mm_popcnt_u32(m_no_conflict);
      state[k].m_have_tuple = _mm512_kandn(m_no_conflict, state[k].m_have_tuple);
      tail_add += _mm_popcnt_u32(m_no_conflict);
    }

    v_addr = _mm512_mask_i64gather_epi64(v_all_ones, state[k].m_have_tuple, _mm512_add_epi64(state[k].ht_off, v_next_off), 0, 1);
    // the remaining matches, DO NOT get next
    state[k].ht_off = _mm512_mask_blend_epi64(m_match, v_addr, state[k].ht_off);

  }

  return found;
}

static void morse_driven(void*param, AGGFun fun, bucket_buffer_t **overflowbuf) {
  arg_t *args = (arg_t *) param;
  uint64_t base = 0, num = 0;
  args->num_results = 0;
  relation_t relS;
  relS.tuples = args->relS.tuples;
  relS.num_tuples = 0;
  while (1) {
    lock(&g_lock_morse);
    base = global_curse;
    global_curse += global_morse_size;
    unlock(&g_lock_morse);
    if (base >= global_upper) {
      break;
    }
    num = (global_upper - base) < global_morse_size ? (global_upper - base) : global_morse_size;
    relS.tuples = args->relS.tuples + base;
    relS.num_tuples = num;
    args->num_results += fun(args->ht, &relS, overflowbuf);
  }
}

map<int64_t,int64_t>len2num;
void search_ht(hashtable_t *ht){
  int64_t len=0;
  for(int64_t i=0;i<ht->num_buckets;++i){
    len=0;
    for(auto it= ht->buckets+i;it;it=it->next){
      len++;
    }
    len2num[len]++;
  }
}

void *agg_thread(void *param) {
  int rv;
  arg_t *args = (arg_t *) param;
  struct timeval t1, t2;
  int deltaT = 0, thread_num = 0;
  bucket_buffer_t *overflowbuf;
  hashtable_t *ht;
  uint64_t nbuckets = (args->relR.num_tuples / BUCKET_SIZE);
  if (args->tid == 0) {
    strcpy(pfun[5].fun_name, "IMV");
    strcpy(pfun[4].fun_name, "AMAC");
    strcpy(pfun[3].fun_name, "FVA");
    strcpy(pfun[2].fun_name, "DVA");
    strcpy(pfun[1].fun_name, "SIMD");
    strcpy(pfun[0].fun_name, "Naive");

    pfun[5].fun_ptr = agg_imv;
    pfun[4].fun_ptr = agg_amac;
    pfun[3].fun_ptr = agg_FVA;
    pfun[2].fun_ptr = agg_DVA;
    pfun[1].fun_ptr = agg_SIMD;
    pfun[0].fun_ptr = agg_raw;

    pf_num = 6;
  }
  BARRIER_ARRIVE(args->barrier, rv);

  for (int fid = 0; fid < pf_num; ++fid) {
    for (int rp = 0; rp < REPEAT_PROBE; ++rp) {
      init_bucket_buffer(&overflowbuf);
      allocate_hashtable(&ht, nbuckets);
      BARRIER_ARRIVE(args->barrier, rv);
      gettimeofday(&t1, NULL);

#if MORSE_SIZE
      args->ht = ht;
      morse_driven(param, pfun[fid].fun_ptr, &overflowbuf);
#else
      args->num_results = pfun[fid].fun_ptr(ht, &args->relS, &overflowbuf);
#endif
      lock(&g_lock);
#if DIVIDE
      total_num += args->num_results;
#elif MORSE_SIZE
      total_num += args->num_results;
#else
      total_num = args->num_results;
#endif
#if PRINT_HT
      search_ht(ht);
#endif
      unlock(&g_lock);
      BARRIER_ARRIVE(args->barrier, rv);
      if (args->tid == 0) {
        gettimeofday(&t2, NULL);
        printf("total result num = %lld\t", total_num);
        deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
        printf("---- %5s AGG costs time (ms) = %10.4lf\n", pfun[fid].fun_name, deltaT * 1.0 / 1000);
        total_num = 0;
        global_curse = 0;
        for(auto iter:ctimes_first){
          cout<<"first cnum = "<<iter.first<<" , times = "<<iter.second<<endl;
        }
        for(auto iter:ctimes_match){
          cout<<"match cnum = "<<iter.first<<" , times = "<<iter.second<<endl;
        }
        for(auto iter:ctimes_tail){
          cout<<"tail  cnum = "<<iter.first<<" , times = "<<iter.second<<endl;
        }
        for(auto iter:len2num){
          cout<<"ht  cnum = "<<iter.first<<" , times = "<<iter.second<<endl;
        }
        len2num.clear();
        ctimes_first.clear();
        ctimes_match.clear();
        ctimes_tail.clear();
      }
      destroy_hashtable(ht);
      free_bucket_buffer(overflowbuf);
    }
  }
  return nullptr;
}

result_t *AGG(relation_t *relR, relation_t *relS, int nthreads) {
  hashtable_t *ht;
  int64_t result = 0;
  int32_t numR, numS, numRthr, numSthr; /* total and per thread num */
  int i, rv;
  cpu_set_t set;
  arg_t args[nthreads];
  pthread_t tid[nthreads];
  pthread_attr_t attr;
  pthread_barrier_t barrier;

  result_t *joinresult = 0;
  joinresult = (result_t *) malloc(sizeof(result_t));

#ifdef JOIN_RESULT_MATERIALIZE
  joinresult->resultlist = (threadresult_t *) alloc_aligned(sizeof(threadresult_t) * nthreads);
#endif

#if USE_TBB && 0
  pthread_attr_init(&attr);
  for (i = 0; i < nthreads; i++) {
    int cpu_idx = get_cpu_id(i);

    DEBUGMSG(1, "Assigning thread-%d to CPU-%d\n", i, cpu_idx);
#if AFFINITY
    CPU_ZERO(&set);
    CPU_SET(cpu_idx, &set);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
#endif
  }

  tbb_run(relR, relS, nthreads);
  joinresult->totalresults = result;
  joinresult->nthreads = nthreads;
  return joinresult;
#endif
  uint32_t nbuckets = (relR->num_tuples / BUCKET_SIZE);
  allocate_hashtable(&ht, nbuckets);

  numR = relR->num_tuples;
  numS = relS->num_tuples;
  numRthr = numR / nthreads;
  numSthr = numS / nthreads;

  rv = pthread_barrier_init(&barrier, NULL, nthreads);
  if (rv != 0) {
    printf("Couldn't create the barrier\n");
    exit(EXIT_FAILURE);
  }
  global_curse = 0;
  global_upper = relS->num_tuples;
  if(nthreads==1){
    global_morse_size= relS->num_tuples;
  }else{
    global_morse_size = MORSE_SIZE;
  }
  pthread_attr_init(&attr);
  for (i = 0; i < nthreads; i++) {
    int cpu_idx = get_cpu_id(i);

#if AFFINITY
    DEBUGMSG(1, "Assigning thread-%d to CPU-%d\n", i, cpu_idx);
    CPU_ZERO(&set);
    CPU_SET(cpu_idx, &set);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
#endif
    args[i].tid = i;
    args[i].ht = ht;
    args[i].barrier = &barrier;
#if DIVIDE
    /* assing part of the relR for next thread */
    args[i].relR.num_tuples = (i == (nthreads - 1)) ? numR : numRthr;
    args[i].relR.tuples = relR->tuples + numRthr * i;
    numR -= numRthr;
#else
    args[i].relR.num_tuples = relR->num_tuples;
    args[i].relR.tuples = relR->tuples;
#endif
#if DIVIDE
    /* assing part of the relS for next thread */
    args[i].relS.num_tuples = (i == (nthreads - 1)) ? numS : numSthr;
    args[i].relS.tuples = relS->tuples + numSthr * i;
    numS -= numSthr;
#else
    args[i].relS.num_tuples = relS->num_tuples;
    args[i].relS.tuples = relS->tuples;
#endif
    args[i].threadresult = &(joinresult->resultlist[i]);

    rv = pthread_create(&tid[i], &attr, agg_thread, (void *) &args[i]);
    if (rv) {
      printf("ERROR; return code from pthread_create() is %d\n", rv);
      exit(-1);
    }
  }

  for (i = 0; i < nthreads; i++) {
    pthread_join(tid[i], NULL);
    /* sum up results */
#if DIVIDE
    result += args[i].num_results;
#else
    result = args[i].num_results;
#endif
  }
  joinresult->totalresults = result;
  joinresult->nthreads = nthreads;

  destroy_hashtable(ht);

  return joinresult;
}
