#include "rof/AmacProbe.hpp"
size_t amac_probe_q11(size_t begin, size_t end, Database& db, runtime::Hashmap* hash_table, uint64_t & results, uint64_t* pos_buff) {
  size_t found = 0, cur = begin;
  int k = 0, done = 0, keyOff = sizeof(runtime::Hashmap::EntryHeader), buildkey;
  Hashjoin::AMACState amac_state[stateNum];
  int probeKey;
  hash_t probeHash;
  // initialization
  for (int i = 0; i < stateNum; ++i) {
    amac_state[i].stage = 1;
  }
  auto& lo = db["lineorder"];
  auto lo_orderdate = lo["lo_orderdate"].data<types::Integer>();
  auto lo_quantity = lo["lo_quantity"].data<types::Integer>();
  auto lo_discount = lo["lo_discount"].data<types::Numeric<18, 2>>();
  auto lo_extendedprice = lo["lo_extendedprice"].data<types::Numeric<18, 2>>();

  while (done < stateNum) {
    k = (k >= stateNum) ? 0 : k;
    switch (amac_state[k].stage) {
      case 1: {
        if (cur >= end) {
          ++done;
          amac_state[k].stage = 3;
          //   std::cout<<"amac done one "<<probe_num<<" , "<<nextProbe<<std::endl;
          break;
        }
#if SEQ_PREFETCH
//        _mm_prefetch((char*)(lo_orderdate+cur)+PDIS,_MM_HINT_T0);
//        _mm_prefetch(((char*)(lo_orderdate+cur)+PDIS+64),_MM_HINT_T0);
#endif
        if (pos_buff) {
          probeKey = *(int*) (lo_orderdate + pos_buff[cur]);
          amac_state[k].tuple_id = pos_buff[cur];
        } else {
          probeKey = *(int*) (lo_orderdate + cur);
          amac_state[k].tuple_id = cur;
        }
        probeHash = (runtime::MurMurHash()(probeKey, primitives::seed));
        ++cur;
        amac_state[k].probeKey = probeKey;
        amac_state[k].probeHash = probeHash;
        hash_table->PrefetchEntry(probeHash);
        amac_state[k].stage = 2;
      }
        break;
      case 2: {
        amac_state[k].buildMatch = hash_table->find_chain_tagged(amac_state[k].probeHash);
        if (nullptr == amac_state[k].buildMatch) {
          amac_state[k].stage = 1;
          --k;
          break;
        } else {
          _mm_prefetch((char * )(amac_state[k].buildMatch), _MM_HINT_T0);
          _mm_prefetch((char * )(amac_state[k].buildMatch) + 64, _MM_HINT_T0);
          amac_state[k].stage = 0;
        }
      }
        break;
      case 0: {
        auto entry = amac_state[k].buildMatch;

        buildkey = *((addBytes((reinterpret_cast<int*>(entry)), keyOff)));
        if ((buildkey == amac_state[k].probeKey)) {
#if WRITE_SEQ_PREFETCH
//          _mm_prefetch((char * )(output_build+pos)+PDIS, _MM_HINT_T0);
//          _mm_prefetch((char * )(output_probe+pos)+PDIS + 64, _MM_HINT_T0);
#endif
          results += lo_extendedprice[amac_state[k].tuple_id].value * lo_discount[amac_state[k].tuple_id].value;
          ++found;
        }
        entry = entry->next;
        if (nullptr == entry) {
          amac_state[k].stage = 1;
          --k;
        } else {
          amac_state[k].buildMatch = entry;
          _mm_prefetch((char * )(entry), _MM_HINT_T0);
          _mm_prefetch((char * )(entry) + 64, _MM_HINT_T0);
        }

      }
        break;
    }
    ++k;
  }
  return found;
}
