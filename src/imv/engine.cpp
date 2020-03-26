#pragma once
#include <unistd.h>

#include "benchmarks/tpch/Queries.hpp"
#include "common/runtime/Hash.hpp"
#include "common/runtime/Types.hpp"
#include "hyper/GroupBy.hpp"
#include "hyper/ParallelHelper.hpp"
#include "tbb/tbb.h"
#include "vectorwise/Operations.hpp"
#include "vectorwise/Operators.hpp"
#include "vectorwise/Primitives.hpp"
#include "vectorwise/QueryBuilder.hpp"
#include "vectorwise/VectorAllocator.hpp"
#include <iostream>
#include "profile.hpp"
#include "common/runtime/Import.hpp"
#include <unordered_set>
#include "imv/HashProbe.hpp"
#include <vector>
#include "imv/Pipeline.hpp"
#include "imv/HashBuild.hpp"
#include "imv/HashAgg.hpp"
#include "imv/PipelineBuild.hpp"

using namespace types;
using namespace runtime;
using namespace std;

using vectorwise::primitives::hash_t;
#define PrintResults 1
#define DEBUG 0
/*
 * select count(*) from orders, lineitem where o_orderkey = l_orderkey and
 */
static int repetitions = 0;
static uint64_t ht_date_size = 1024;

auto vectorJoinFun = &vectorwise::Hashjoin::joinAMAC;
auto pipelineFun = &filter_probe_simd_imv;
auto aggFun = &agg_raw;
auto buildFun = &build_raw;

auto constrant_o_orderdate = types::Date::castString("1996-01-01");
types::Numeric<12, 2> constrant_l_quantity = types::Numeric<12, 2>(types::Integer(CONSTRANT_L_QUAN));

size_t nrTuples(Database& db, std::vector<std::string> tables) {
  size_t sum = 0;
  for (auto& table : tables)
    sum += db[table].nrTuples;
  return sum;
}
static PipelineTimer ptimer;
bool Qa_compilation(Database& db, size_t nrThreads) {
  ptimer.reset();
  ptimer.log_time("start");

  auto resources = initQuery(nrThreads);
  auto c2 = types::Numeric<12, 2>::castString("0.07");
  auto c3 = types::Integer(24);

  auto& ord = db["orders"];
  auto& li = db["lineitem"];
  auto o_orderkey = ord["o_orderkey"].data<types::Integer>();
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
  auto o_orderdate = ord["o_orderdate"].data<types::Date>();
  auto l_quantity_col = li["l_quantity"].data<types::Numeric<12, 2>>();

  using hash = runtime::MurMurHash;
  using range = tbb::blocked_range<size_t>;
  const auto add = [](const size_t& a, const size_t& b) {return a + b;};

// build a hash table from [orders]
  Hashset<types::Integer, hash> ht1;
  tbb::enumerable_thread_specific<runtime::Stack<decltype(ht1)::Entry>> entries1;
  size_t tuples = ord.nrTuples;

  auto found1 = tbb::parallel_reduce(range(0, tuples, morselSize), 0, [&](const tbb::blocked_range<size_t>& r, const size_t& f) {
    auto found = f;
    auto& entries = entries1.local();
    for (size_t i = r.begin(), end = r.end(); i != end; ++i) {
      if(o_orderdate[i] < constrant_o_orderdate) {
        entries.emplace_back(ht1.hash(o_orderkey[i]), o_orderkey[i]);
        found++;
      }
    }
    return found;
  },
                                     add);
  ht_date_size = found1;
  ht1.setSize(found1);
  parallel_insert(entries1, ht1);
  ptimer.log_time("build");
#if DEBUG
  cout << "Build hash table tuples num = " << found1 << endl;
  ht1.printStaTag();
#endif
// look up the hash table 1
  auto found2 = tbb::parallel_reduce(range(0, li.nrTuples, morselSize), 0, [&](const tbb::blocked_range<size_t>& r, const size_t& f) {
    auto found = f;
    for (size_t i = r.begin(), end = r.end(); i != end; ++i)
    if (l_quantity_col[i]<constrant_l_quantity&& ht1.contains(l_orderkey[i])) {
      found++;
    }
    return found;
  },
                                     add);
  ptimer.log_time("probe");
#if PrintResults
  cout << "Qa hyper results num = " << found2 << endl;
#endif
  leaveQuery(nrThreads);
  return true;
}
bool Qa_pipeline(Database& db, size_t nrThreads) {

  auto resources = initQuery(nrThreads);
  auto c2 = types::Numeric<12, 2>::castString("0.07");
  auto c3 = types::Integer(24);

  auto& ord = db["orders"];
  auto& li = db["lineitem"];
  auto& part = db["part"];
  auto p_partkey = part["p_partkey"].data<types::Integer>();

  auto o_orderkey = ord["o_orderkey"].data<types::Integer>();
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
  auto o_orderdate = ord["o_orderdate"].data<types::Date>();
  auto l_quantity_col = li["l_quantity"].data<types::Numeric<12, 2>>();

  using hash = runtime::MurMurHash;
  using range = tbb::blocked_range<size_t>;
  const auto add = [](const size_t& a, const size_t& b) {return a + b;};

// build a hash table from [orders]
  Hashset<types::Integer, hash> ht1;
  tbb::enumerable_thread_specific<runtime::Stack<decltype(ht1)::Entry>> entries1;
  size_t tuples = ord.nrTuples;

  auto found1 = tbb::parallel_reduce(range(0, tuples, morselSize), 0, [&](const tbb::blocked_range<size_t>& r, const size_t& f) {
    auto found = f;
    auto& entries = entries1.local();
    for (size_t i = r.begin(), end = r.end(); i != end; ++i) {
      if(o_orderdate[i] < constrant_o_orderdate) {
        entries.emplace_back(ht1.hash(o_orderkey[i]), o_orderkey[i]);
        found++;
      }
    }
    return found;
  },
                                     add);

  ht1.setSize(found1);
  parallel_insert(entries1, ht1);
#if DEBUG
  cout << "Build hash table tuples num = " << found1 << endl;
  ht1.printStaTag();
#endif

  vector<pair<string, decltype(pipelineFun)> > compilerName2fun;

//  compilerName2fun.push_back(make_pair("filter_probe_scalar", filter_probe_scalar));
  compilerName2fun.push_back(make_pair("filter_probe_imv1", filter_probe_imv1));
  compilerName2fun.push_back(make_pair("filter_probe_imv2", filter_probe_imv2));
  compilerName2fun.push_back(make_pair("filter_probe_imv", filter_probe_imv));
  compilerName2fun.push_back(make_pair("filter_probe_simd_imv", filter_probe_simd_imv));
  compilerName2fun.push_back(make_pair("filter_probe_simd_gp", filter_probe_simd_gp));
  compilerName2fun.push_back(make_pair("filter_probe_simd_amac", filter_probe_simd_amac));
  tbb::enumerable_thread_specific<vector<uint32_t>> probe_offset;
  tbb::enumerable_thread_specific<vector<void*>> build_addr;
  tbb::enumerable_thread_specific<vector<uint64_t>> rof_buffer;

  PerfEvents event;
  uint64_t found2 = 0;
  for (auto name2fun : compilerName2fun) {
    auto pipelineFun = name2fun.second;
    event.timeAndProfile(name2fun.first, li.nrTuples, [&]() {
      found2 = tbb::parallel_reduce(range(0, li.nrTuples, morselSize), 0, [&](const tbb::blocked_range<size_t>& r, const size_t& f) {
            auto found = f;
            auto rof_buff = rof_buffer.local();
            rof_buff.clear();
            rof_buff.resize(ROF_VECTOR_SIZE);
            auto probe_off = probe_offset.local();
            probe_off.clear();
            probe_off.resize(morselSize);
            auto build_add = build_addr.local();
            build_add.clear();
            build_add.resize(morselSize);

            found+=pipelineFun(r.begin(),r.end(),db,&ht1,&build_add[0],&probe_off[0],&rof_buff[0]);

            return found;
          },
          add);
    },
                         repetitions);
#if PrintResults
    cout << "Qa results num :" << found2 << endl;
#endif
  }

  leaveQuery(nrThreads);
  return true;
}
bool Qa_imv(Database& db, size_t nrThreads) {
  ptimer.reset();
  ptimer.log_time("start");
  auto resources = initQuery(nrThreads);
  auto& ord = db["orders"];
  auto& li = db["lineitem"];
  auto o_orderkey = ord["o_orderkey"].data<types::Integer>();
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
  auto o_orderdate = ord["o_orderdate"].data<types::Date>();
  auto l_quantity_col = li["l_quantity"].data<types::Numeric<12, 2>>();

  using hash = runtime::MurMurHash;
  using range = tbb::blocked_range<size_t>;
  const auto add = [](const size_t& a, const size_t& b) {return a + b;};

// build a hash table from [orders]
  Hashset<types::Integer, hash> ht1;
  tbb::enumerable_thread_specific<runtime::Stack<decltype(ht1)::Entry>> entries1;
  size_t tuples = ord.nrTuples;
  tbb::enumerable_thread_specific<vector<uint64_t>> position;
  auto entry_size = sizeof(decltype(ht1)::Entry);
  ht1.setSize(ht_date_size);

  auto found1 = tbb::parallel_reduce(range(0, tuples, morselSize), 0, [&](const tbb::blocked_range<size_t>& r, const size_t& f) {
    auto found = f;
    auto pos_buff = position.local();
    pos_buff.clear();
    pos_buff.resize(ROF_VECTOR_SIZE);
    found +=build_pipeline_imv_qa(r.begin(),r.end(),db,&ht1,&this_worker->allocator,entry_size,constrant_o_orderdate);

    return found;
  },
                                     add);
  ptimer.log_time("build");
#if DEBUG
  cout << "Build hash table tuples num = " << found1 << endl;
  ht1.printStaTag();
#endif

  tbb::enumerable_thread_specific<vector<uint32_t>> probe_offset;
  tbb::enumerable_thread_specific<vector<void*>> build_addr;
  tbb::enumerable_thread_specific<vector<uint64_t>> rof_buffer;

  uint64_t found2 = 0;
  found2 = tbb::parallel_reduce(range(0, li.nrTuples, morselSize), 0, [&](const tbb::blocked_range<size_t>& r, const size_t& f) {
    auto found = f;
    auto rof_buff = rof_buffer.local();
    rof_buff.clear();
    rof_buff.resize(ROF_VECTOR_SIZE);
    auto probe_off = probe_offset.local();
    probe_off.clear();
    probe_off.resize(morselSize);
    auto build_add = build_addr.local();
    build_add.clear();
    build_add.resize(morselSize);

    found+=filter_probe_imv(r.begin(),r.end(),db,&ht1,&build_add[0],&probe_off[0],&rof_buff[0]);

    return found;
  },
                                add);
  ptimer.log_time("probe");

#if PrintResults
  cout << "Qa results num :" << found2 << endl;
#endif

  leaveQuery(nrThreads);
  return true;
}
bool Qa_rof(Database& db, size_t nrThreads) {
  ptimer.reset();
  ptimer.log_time("start");
  auto resources = initQuery(nrThreads);
  auto& ord = db["orders"];
  auto& li = db["lineitem"];
  auto o_orderkey = ord["o_orderkey"].data<types::Integer>();
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
  auto o_orderdate = ord["o_orderdate"].data<types::Date>();
  auto l_quantity_col = li["l_quantity"].data<types::Numeric<12, 2>>();

  using hash = runtime::MurMurHash;
  using range = tbb::blocked_range<size_t>;
  const auto add = [](const size_t& a, const size_t& b) {return a + b;};

// build a hash table from [orders]
  Hashset<types::Integer, hash> ht1;
  tbb::enumerable_thread_specific<runtime::Stack<decltype(ht1)::Entry>> entries1;
  size_t tuples = ord.nrTuples;
  tbb::enumerable_thread_specific<vector<uint64_t>> position;
  auto entry_size = sizeof(decltype(ht1)::Entry);
  ht1.setSize(ht_date_size);

  auto found1 = tbb::parallel_reduce(range(0, tuples, morselSize), 0, [&](const tbb::blocked_range<size_t>& r, const size_t& f) {
    auto found = f;
    auto pos_buff = position.local();
    pos_buff.clear();
    pos_buff.resize(ROF_VECTOR_SIZE);
    for (size_t i = r.begin(), size = 0; i < r.end(); ) {
      size = simd_filter_qa_build(i,r.end(),db,&pos_buff[0],constrant_o_orderdate);
      found += build_gp_qa(0,size,db,&ht1,&this_worker->allocator,entry_size, &pos_buff[0]);
    }
    return found;
  },
                                     add);
  ptimer.log_time("build");

#if DEBUG
  cout << "Build hash table tuples num = " << found1 << endl;
  ht1.printStaTag();
#endif

  tbb::enumerable_thread_specific<vector<uint32_t>> probe_offset;
  tbb::enumerable_thread_specific<vector<void*>> build_addr;
  tbb::enumerable_thread_specific<vector<uint64_t>> rof_buffer;

  uint64_t found2 = 0;
  found2 = tbb::parallel_reduce(range(0, li.nrTuples, morselSize), 0, [&](const tbb::blocked_range<size_t>& r, const size_t& f) {
    auto found = f;
    auto rof_buff = rof_buffer.local();
    rof_buff.clear();
    rof_buff.resize(ROF_VECTOR_SIZE);
    auto probe_off = probe_offset.local();
    probe_off.clear();
    probe_off.resize(morselSize);
    auto build_add = build_addr.local();
    build_add.clear();
    build_add.resize(morselSize);

    found+=filter_probe_simd_gp(r.begin(),r.end(),db,&ht1,&build_add[0],&probe_off[0],&rof_buff[0]);

    return found;
  },
                                add);
  ptimer.log_time("probe");

#if PrintResults
  cout << "Qa results num :" << found2 << endl;
#endif

  leaveQuery(nrThreads);
  return true;
}
bool Qa_rof_imv(Database& db, size_t nrThreads) {
  ptimer.reset();
  ptimer.log_time("start");
  auto resources = initQuery(nrThreads);
  auto& ord = db["orders"];
  auto& li = db["lineitem"];
  auto o_orderkey = ord["o_orderkey"].data<types::Integer>();
  auto l_orderkey = li["l_orderkey"].data<types::Integer>();
  auto o_orderdate = ord["o_orderdate"].data<types::Date>();
  auto l_quantity_col = li["l_quantity"].data<types::Numeric<12, 2>>();

  using hash = runtime::MurMurHash;
  using range = tbb::blocked_range<size_t>;
  const auto add = [](const size_t& a, const size_t& b) {return a + b;};

// build a hash table from [orders]
  Hashset<types::Integer, hash> ht1;
  tbb::enumerable_thread_specific<runtime::Stack<decltype(ht1)::Entry>> entries1;
  size_t tuples = ord.nrTuples;
  tbb::enumerable_thread_specific<vector<uint64_t>> position;
  auto entry_size = sizeof(decltype(ht1)::Entry);
  ht1.setSize(ht_date_size);

  auto found1 = tbb::parallel_reduce(range(0, tuples, morselSize), 0, [&](const tbb::blocked_range<size_t>& r, const size_t& f) {
    auto found = f;
    auto pos_buff = position.local();
    pos_buff.clear();
    pos_buff.resize(ROF_VECTOR_SIZE);
    for (size_t i = r.begin(), size = 0; i < r.end(); ) {
      size = simd_filter_qa_build(i,r.end(),db,&pos_buff[0],constrant_o_orderdate);
      found += build_imv_qa(0,size,db,&ht1,&this_worker->allocator,entry_size, &pos_buff[0]);
    }
    return found;
  },
                                     add);
  ptimer.log_time("build");

#if DEBUG
  cout << "Build hash table tuples num = " << found1 << endl;
  ht1.printStaTag();
#endif

  tbb::enumerable_thread_specific<vector<uint32_t>> probe_offset;
  tbb::enumerable_thread_specific<vector<void*>> build_addr;
  tbb::enumerable_thread_specific<vector<uint64_t>> rof_buffer;

  uint64_t found2 = 0;
  found2 = tbb::parallel_reduce(range(0, li.nrTuples, morselSize), 0, [&](const tbb::blocked_range<size_t>& r, const size_t& f) {
    auto found = f;
    auto rof_buff = rof_buffer.local();
    rof_buff.clear();
    rof_buff.resize(ROF_VECTOR_SIZE);
    auto probe_off = probe_offset.local();
    probe_off.clear();
    probe_off.resize(morselSize);
    auto build_add = build_addr.local();
    build_add.clear();
    build_add.resize(morselSize);

    found+=filter_probe_simd_imv(r.begin(),r.end(),db,&ht1,&build_add[0],&probe_off[0],&rof_buff[0]);

    return found;
  },
                                add);
  ptimer.log_time("probe");

#if PrintResults
  cout << "Qa results num :" << found2 << endl;
#endif

  leaveQuery(nrThreads);
  return true;
}

std::unique_ptr<Q3Builder::Q3> Q3Builder::getQuery() {
  using namespace vectorwise;
  auto result = Result();
  previous = result.resultWriter.shared.result->participate();
  auto r = make_unique<Q3>();
// JOIN SEL

  auto order = Scan("orders");
//  o_orderdate < date '"1996-01-01"
  Select(Expression().addOp(BF(primitives::sel_less_Date_col_Date_val),  //
                            Buffer(sel_order, sizeof(pos_t)),  //
                            Column(order, "o_orderdate"),  //
                            Value(&constrant_o_orderdate)));
///////////NOTE the order
  auto lineitem = Scan("lineitem");

// l_quantity < 24
  Select((Expression()  //
      .addOp(BF(vectorwise::primitives::sel_less_int64_t_col_int64_t_val),  //
             Buffer(sel_cust, sizeof(pos_t)),  //
             Column(lineitem, "l_quantity"),  //
             Value(&constrant_l_quantity))));

  HashJoin(Buffer(cust_ord, sizeof(pos_t)), vectorJoinFun)  //
      .setProbeSelVector(Buffer(sel_cust), vectorJoinFun).addBuildKey(Column(order, "o_orderkey"), Buffer(sel_order), conf.hash_sel_int32_t_col(),
                                                                      primitives::scatter_sel_int32_t_col)  //
      .addProbeKey(Column(lineitem, "l_orderkey"), Buffer(sel_cust), conf.hash_sel_int32_t_col(), primitives::keys_equal_int32_t_col).setHTsize(ht_date_size);

// count(*)
  FixedAggregation(Expression().addOp(primitives::aggr_static_count_star, Value(&r->count)));

//   result.addValue("count", Buffer(cust_ord))
//       .finalize();

  r->rootOp = popOperator();
  return r;
}
bool join_vectorwise(Database& db, size_t nrThreads, size_t vectorSize) {
  using namespace vectorwise;
  WorkerGroup workers(nrThreads);
  vectorwise::SharedStateManager shared;
  std::unique_ptr<runtime::Query> result;
  std::atomic<int64_t> aggr;
  aggr = 0;
  workers.run([&]() {
    Q3Builder builder(db, shared, vectorSize);
    auto query = builder.getQuery();
    /* auto found = */
    auto n_ = query->rootOp->next();
    if (n_) {
      aggr.fetch_add(query->count);
    }
#if PrintResults
              auto leader = barrier();
              if (leader) {
                cout<<"vectorwise join result: "<< aggr.load()<<endl;
              }
#endif
            });
  for (int i = 1; i < workers.pipeline_cost_time.size(); ++i) {
    double run_time = workers.pipeline_cost_time[i].second - workers.pipeline_cost_time[i - 1].second;
    cout << workers.pipeline_cost_time[i].first << " cost time = " << run_time * 1e3 << std::endl;
  }

  return true;
}

void test_vectorwise_sel_probe(Database& db, size_t nrThreads, PerfEvents &e) {
  size_t vectorSize = 1024;
  vector<pair<string, decltype(vectorJoinFun)> > vectorName2fun;
  vectorName2fun.push_back(make_pair("joinSelSIMD", &vectorwise::Hashjoin::joinSelSIMD));
  vectorName2fun.push_back(make_pair("joinSelParallel", &vectorwise::Hashjoin::joinSelParallel));
  vectorName2fun.push_back(make_pair("joinIMV", &vectorwise::Hashjoin::joinSelIMV));

  for (auto name2fun : vectorName2fun) {
    vectorJoinFun = name2fun.second;
    e.timeAndProfile(name2fun.first, nrTuples(db, { "orders", "lineitem" }), [&]() {
      join_vectorwise(db,nrThreads,vectorSize);
    },
                     repetitions);
  }
}
int main(int argc, char* argv[]) {
  if (argc <= 2) {
    std::cerr << "Usage: ./" << argv[0] << "<number of repetitions> <path to tpch dir> [nrThreads = all] \n "
              " EnvVars: [vectorSize = 1024] [SIMDhash = 0] [SIMDjoin = 0] "
              "[SIMDsel = 0]";
    exit(1);
  }

  Database tpch;
// load tpch data
  importTPCH(argv[2], tpch);
  PerfEvents e;

// run queries
  repetitions = atoi(argv[1]);
  size_t nrThreads = std::thread::hardware_concurrency();
  size_t vectorSize = 1024;
  bool clearCaches = false;
  if (argc > 3)
    nrThreads = atoi(argv[3]);
  tbb::task_scheduler_init scheduler(nrThreads);
 ptimer.reset();
  e.timeAndProfile("compilation", nrTuples(tpch, { "orders", "lineitem" }), [&]() {
    Qa_compilation(tpch, nrThreads);
  },
                   repetitions);
  ptimer.print();
  e.timeAndProfile("imv", nrTuples(tpch, { "orders", "lineitem" }), [&]() {
    Qa_imv(tpch, nrThreads);
  },
                   repetitions);
  ptimer.print();
  e.timeAndProfile("rof", nrTuples(tpch, { "orders", "lineitem" }), [&]() {
    Qa_rof(tpch, nrThreads);
  },
                   repetitions);
  ptimer.print();
  e.timeAndProfile("rof-imv", nrTuples(tpch, { "orders", "lineitem" }), [&]() {
    Qa_imv(tpch, nrThreads);
  },
                   repetitions);
  ptimer.print();
  test_vectorwise_sel_probe(tpch, nrThreads, e);
  scheduler.terminate();
  return 0;
}
