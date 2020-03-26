#include "rof/SimdFilter.hpp"
size_t simd_filter_q11_build_date(size_t& begin, size_t end, Database& db, uint64_t* pos_buff) {
  size_t found = 0;
  __mmask8 m_valid = -1, m_eval;
  __m512i v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), v_offset;
  ///////////////////
  //D_YEAR = 1993
  // int => 32 bits
  auto& d = db["date"];
  auto d_year = d["d_year"].data<types::Integer>();
  __m256i v256_const = _mm256_set1_epi32(1993), v256_col;
  //////////////////////
  for (size_t & cur = begin; cur < end;) {
    v_offset = _mm512_add_epi64(_mm512_set1_epi64(cur), v_base_offset);
    if (cur + VECTORSIZE >= end) {
      m_valid = (m_valid >> (cur + VECTORSIZE - end));
    }
    /////////////////
    v256_col = _mm256_maskz_loadu_epi32(m_valid, (d_year + cur));
    m_eval = _mm256_cmpeq_epu32_mask(v256_const, v256_col);
    m_eval = _mm512_kand(m_eval, m_valid);
    //////////////////
    cur += VECTORSIZE;
    _mm512_mask_compressstoreu_epi64(pos_buff + found, m_eval, v_offset);
    found += _mm_popcnt_u32((m_eval));
    if(found + VECTORSIZE >= ROF_VECTOR_SIZE) {
      return found;
    }
  }
  return found;
}

size_t simd_filter_q11_probe(size_t& begin, size_t end, Database& db, uint64_t* pos_buff) {
  size_t found = 0;
  __mmask8 m_valid = -1, m_eval;
  __m512i v_base_offset = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), v_offset;
  ///////////////////
  // LO_DISCOUNT BETWEEN 1 AND 3 AND LO_QUANTITY < 25;
  // Numeric => 64 bits,  int => 32 bits
  auto& lo = db["lineorder"];
  auto lo_quantity = lo["lo_quantity"].data<types::Integer>();
  auto lo_discount = lo["lo_discount"].data<types::Numeric<18, 2>>();

  __m256i v256_25 = _mm256_set1_epi32(25), v256_col;
  __m512i v_col, v_100 = _mm512_set1_epi64(100), v_300 = _mm512_set1_epi64(300);
  //////////////////////
  for (size_t & cur = begin; cur < end;) {
    v_offset = _mm512_add_epi64(_mm512_set1_epi64(cur), v_base_offset);
    if (cur + VECTORSIZE >= end) {
      m_valid = (m_valid >> (cur + VECTORSIZE - end));
    }
    m_eval = m_valid;
    /////////////////
    // LO_DISCOUNT BETWEEN 1 AND 3
    v_col = _mm512_maskz_loadu_epi64(m_valid,(lo_discount+ cur));
    m_eval = _mm512_kand( _mm512_cmpge_epu64_mask(v_300, v_col),_mm512_cmpge_epu64_mask(v_col,v_100));
    // LO_QUANTITY < 25
    v256_col = _mm256_maskz_loadu_epi32(m_valid, (lo_quantity + cur));
    m_eval =_mm512_kand(m_eval, _mm256_cmpgt_epu32_mask(v256_25, v256_col));
    m_eval = _mm512_kand(m_eval, m_valid);
    //////////////////
    cur += VECTORSIZE;
    _mm512_mask_compressstoreu_epi64(pos_buff + found, m_eval, v_offset);
    found += _mm_popcnt_u32((m_eval));
    if(found + VECTORSIZE >= ROF_VECTOR_SIZE) {
      return found;
    }
  }
  return found;

}
