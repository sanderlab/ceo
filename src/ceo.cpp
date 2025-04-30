#include <iostream>
#include <cmath>
#include <numeric>
#include <future>
#include <cstdlib>
#include <fmt/core.h>
#include <Eigen/Dense>
#include <cnpy.h>
#include "ceo.h"

namespace ceo {
    CEO::CEO(cnpy::npz_t & npz, int num_threads = 1, bool ignore1 = false) :
    m_num_threads {num_threads},
    m_ignore1 {ignore1},
    m_thread_pool {m_num_threads},
    m_N {npz[NPZ_onehot].shape[0]}, 
    m_D {npz[NPZ_onehot].shape[1]},
    m_L {npz[NPZ_onehot].shape[2]}
    {
        m_logfactorial_cache.resize(m_N + 1);
        m_log_cache.resize(m_N + 1);
        for (size_type j = 0; j <= m_N; ++j) {
            m_logfactorial_cache[j] = lgamma(1 + static_cast<entropy_type>(j));
            m_log_cache[j] = log(static_cast<entropy_type>(j));
        }

        Eigen::Map<FloatingPointArrayX> map_A(npz[NPZ_A].data<entropy_type>(), npz[NPZ_A].shape[0]);
        m_A.resize(map_A.size());
        m_A = map_A;

        Eigen::Map<IntegerArrayXX> map_N_iak(npz[NPZ_onehot].data<size_type>(), m_L, m_D * m_N);
        m_N_iak.resize(m_L, m_D * m_N);
        m_N_iak = map_N_iak;

        m_N_ia.resize(m_L, m_D);
        m_N_ia.setConstant(0);
        for (size_type k = 0; k < m_N; ++k) {
            for (size_type a = 0; a < m_D; ++a) {
                m_N_ia(Eigen::all, a) += m_N_iak(Eigen::all, a + k * m_D);
            }
        }

        // Stilde_km for all possible cluster sizes j = 1 .. m_N
        m_Stilde_km_cache.resize(m_N + 1);
        for (size_type j = 1; j <= m_N; ++j) {
            m_Stilde_km_cache[j] = 0.;
            for (size_type a = 0; a < m_D; ++a) {
                for (size_type i = 0; i < m_L; ++i) {
                    m_Stilde_km_cache[j] -= lgamma(1. + (entropy_type)j * (entropy_type)m_N_ia(i, a) / (entropy_type)m_N);
                }
            }
        }

        if (m_ignore1) {
            m_Stilde_km_cache[1] = 0.;
        }

        entropy_type dS0_0 = m_N * (-m_Stilde_km_cache[1]);
        std::cout << m_N << " clusters: dS0 = " << dS0_0 << std::endl;

        m_dS_km_cache.resize(m_N * (m_N - 1) / 2);
        initialize_dS_km_cache_mt();
        // initialize_dS_km_cache();

        m_result_merge_km.resize(2, m_A.size() * (m_N - 1));
        m_result_dS0.resize(m_A.size() * (m_N - 1));
        m_result_dS_km.resize(m_A.size() * (m_N - 1));
        m_result_dQ_km.resize(m_A.size() * (m_N - 1));
    }

    auto CEO::km_cache_index(std::size_t cache_index) {
        size_type const m = static_cast<size_type>((1 + sqrt(8 * cache_index + 1)) / 2);
        size_type const k = cache_index - m * (m - 1) / 2;
        return std::make_pair(k, m);
    }

    entropy_type CEO::real_dS_km(size_type k, size_type m, IntegerArrayXX const & N_iak, IntegerArrayX const & N_k) {
        entropy_type S = 0.;        
        for (size_type a = 0; a < m_D; ++a) {
            for (size_type i = 0; i < m_L; ++i) {
                S -= m_logfactorial_cache[N_iak(i, a + k * m_D) + N_iak(i, a + m * m_D)];
            }
        }
        entropy_type Stilde = m_Stilde_km_cache[N_k(k) + N_k(m)];
        entropy_type result = S - Stilde;

        // m_logger.limit("dS_km.txt", 10);
        // auto & ofs = m_logger("dS_km.txt");
        // ofs << fmt::format("{}_{} S={:.17g} N_k+N_m={} Stilde={:.17g} dS_km={:.17g}\n", k+1, m+1, S, N_k(k) + N_k(m), Stilde, result);
        return result;
    }

    void CEO::initialize_dS_km_cache() {
        IntegerArrayX const N_k {IntegerArrayX::Ones(m_N)};
        std::size_t j = 0;
        for (size_type m = 1; m < m_N; ++m) {
            for (size_type k = 0; k < m; ++k) {
                m_dS_km_cache[j++] = real_dS_km(k, m, m_N_iak, N_k);
            }
        }

        // for (size_type k = 0; k < m_N - 1; ++k) {
        //     for (size_type m = k + 1; m < m_N; ++m) {
                // if (dS_km_cache_core_index(k, m) < 10)
                //     std::cout << k + 1 << " " << m + 1 << " " << real_dS_km(k, m) << std::endl;
        //     }
        // }
    }

    void CEO::initialize_dS_km_cache_worker(std::size_t chunk_begin, std::size_t chunk_end, IntegerArrayX const & N_k) {
        auto [chunk_begin_k, chunk_begin_m] = km_cache_index(chunk_begin);
        auto [chunk_end_k, chunk_end_m] = km_cache_index(chunk_end);
        std::size_t j = chunk_begin;

        if (chunk_begin_m == chunk_end_m) {
            for (size_type k = chunk_begin_k; k <= chunk_end_k; ++k) {
                m_dS_km_cache[j++] = real_dS_km(k, chunk_begin_m, m_N_iak, N_k);
            }
            return;
        }

        for (size_type k = chunk_begin_k; k < chunk_begin_m; ++k) {
            m_dS_km_cache[j++] = real_dS_km(k, chunk_begin_m, m_N_iak, N_k);
        }

        for (size_type m = chunk_begin_m + 1; m < chunk_end_m; ++m) {
            for (size_type k = 0; k < m; ++k) {
                m_dS_km_cache[j++] = real_dS_km(k, m, m_N_iak, N_k);
            }
        }

        for (size_type k = 0; k <= chunk_end_k; ++k) {
            m_dS_km_cache[j++] = real_dS_km(k, chunk_end_m, m_N_iak, N_k);
        }
    }

    void CEO::initialize_dS_km_cache_mt() {
        IntegerArrayX const N_k {IntegerArrayX::Ones(m_N)};
        std::size_t const cache_size = m_N * (m_N - 1) / 2;
        const std::size_t q = cache_size / m_num_threads;
        const std::size_t r = cache_size % m_num_threads;
        std::size_t chunk_begin = 0; 
        std::size_t num_workers = 0;
        std::vector<std::future<void>> threadhandles(m_num_threads);
        for (std::size_t j = 0; j < m_num_threads; ++j) {
            std::size_t chunk_size = (j < r) ? (q + 1) : q;
            if (chunk_size == 0) break;
            ++num_workers;            
            std::size_t chunk_end = chunk_begin + chunk_size - 1;
            // std::cout << "initialize_dS_km_cache_mt: chunk " << j << ": " << chunk_begin << "-" << chunk_end << std::endl;
            threadhandles[j] = m_thread_pool.push([this, chunk_begin, chunk_end, & N_k](int thread_id) {
                this->initialize_dS_km_cache_worker(chunk_begin, chunk_end, N_k);
            });
            chunk_begin += chunk_size;
        }

        for (std::size_t j = 0; j < num_workers; ++j) {
            threadhandles[j].wait();
        }
    }

    void CEO::grid_search_worker(std::size_t const chunk_begin_point, std::size_t const chunk_end_point) {
        entropy_type const inf = std::numeric_limits<entropy_type>::infinity();
        // one more entry at the end (m_N + 1) to serve as temporary storage for the newly-merged cluster
        IntegerArrayXX N_iak(m_L, m_D * (m_N + 1));
        IntegerArrayX N_k(m_N + 1);
        FloatingPointArrayX dS_k(m_N + 1);
        FloatingPointArrayX dS_km_cache(m_N * (m_N - 1) / 2);
        // FloatingPointArrayX dQ_km_cache(m_N * (m_N - 1) / 2);
        std::size_t result_index = chunk_begin_point * (m_N - 1);

        for (std::size_t i = chunk_begin_point; i <= chunk_end_point; ++i) {
            entropy_type A = m_A(i);

            N_iak(Eigen::all, Eigen::seqN(0, m_D * m_N)) = m_N_iak;
            N_k.setConstant(size_type(1));
            dS_k.setConstant(-m_Stilde_km_cache[1]);
            dS_km_cache = m_dS_km_cache;

            // calculate dQ_km_cache
            // dQ_km_cache = A * dS_km_cache + (1 - A) * m_L * m_logfactorial_cache[2];
            // use the above one-liner instead of the following nested loops
            // std::size_t j = 0;
            // for (size_type m = 1; m < m_N; ++m) {
            //     for (size_type k = 0; k < m; ++k) {
            // // for (size_type k = 0; k < m_N - step - 1; ++k) {
            // //     for (size_type m = k + 1; m < m_N - step; ++m) {
            //         dQ_km_cache(j) = A * dS_km_cache(j) + (1 - A) * m_L * m_logfactorial_cache[2]; // m_logfactorial_cache[N_k(k) + N_k(m)]
            //         ++j;
            //     }
            // }

            for (size_type step = 0; step < m_N - 1; ++step) {
                size_type best_k, best_m;
                entropy_type best_dQ_km = inf;
                std::size_t j = 0;
                for (size_type m = 1; m < m_N - step; ++m) {
                    for (size_type k = 0; k < m; ++k) {
                // for (size_type k = 0; k < m_N - step - 1; ++k) {
                //     for (size_type m = k + 1; m < m_N - step; ++m, ++j) {
                        // version 1
                        // entropy_type const tmp_dQ_km = A * dS_km_cache(j) / m_L + (1. - A) * m_logfactorial_cache[N_k(k) + N_k(m)];
                        
                        // version 2
                        entropy_type const tmp_dQ_km = (A * dS_km_cache(j) / m_L + (1. - A) * m_logfactorial_cache[N_k(k) + N_k(m)]) / (N_k(k) + N_k(m)); // normalize by new cluster size. Is it ok?
                        
                        // version 3
                        // entropy_type const tmp_dQ_km = A * dS_km_cache(j) / m_L / (N_k(k) + N_k(m)) + (1. - A) * m_log_cache[N_k(k) + N_k(m)];
                        
                        // cache dQ_km
                        // entropy_type const tmp_dQ_km = dQ_km_cache(j);

                        if (tmp_dQ_km < best_dQ_km) {
                            dS_k(m_N) = dS_km_cache(j);
                            best_k = k;
                            best_m = m;
                            best_dQ_km = tmp_dQ_km;
                        }

                        ++j;
                    }
                }

                // std::size_t j = 0;
                // entropy_type best_dQ_km = dQ_km_cache(Eigen::seqN(0, (m_N - step) * (m_N - step - 1) / 2)).minCoeff(&j);
                // dS_new_cluster = dS_km_cache(j);
                // best_m = static_cast<size_type>((1 + sqrt(8 * j + 1)) / 2);
                // best_k = j - best_m * (best_m - 1) / 2;

                // if ((198 <= step) && (step <= 200)) {
                //     std::size_t j = 0;
                //     for (size_type m = 1; m < m_N - step; ++m) {
                //         for (size_type k = 0; k < m; ++k) {
                //             dS_km_cache_ofs << fmt::format("A={:.3f} step={} {}_{}: {}_{} N_k+N_m={} dQ_km={:.17g} dS_km={:.17g}\n", A, step+1, k+1, m+1, labels(k)+1, labels(m)+1, N_k(k) + N_k(m), tmp_dQ_km, dS_km_cache(j));
                //             ++j;
                //         }
                //     }
                // }

                N_k(m_N) = N_k(best_k) + N_k(best_m);
                N_iak(Eigen::all, Eigen::seqN(m_N * m_D, m_D)) = N_iak(Eigen::all, Eigen::seqN(best_k * m_D, m_D)) + N_iak(Eigen::all, Eigen::seqN(best_m * m_D, m_D));

                remove_paired_cache_items_km(dS_km_cache, m_N - step, best_k, best_m);
                // remove_paired_cache_items_km(dQ_km_cache, m_N - step, best_k, best_m);
                remove_linear_cache_items_km(N_k, m_N - step, best_k, best_m);
                remove_linear_cache_items_km(dS_k, m_N - step, best_k, best_m);
                remove_N_iak_items_km(N_iak, m_N - step, best_k, best_m);

                // remove_linear_cache_items_km(labels, m_N - step, best_k, best_m);

                size_type new_cluster = m_N - step - 2;
                dS_k(new_cluster) = dS_k(m_N);
                N_k(new_cluster) = N_k(m_N);
                N_iak(Eigen::all, Eigen::seqN(new_cluster * m_D, m_D)) = N_iak(Eigen::all, Eigen::seqN(m_N * m_D, m_D));
                
                std::size_t base = cache_index_km(0, new_cluster);
                for (size_type j = 0; j < new_cluster; ++j) {
                    dS_km_cache(base + j) = real_dS_km(j, new_cluster, N_iak, N_k);
                }
                // for (size_type j = 0; j < new_cluster; ++j) {
                //     dQ_km_cache(base + j) = A * dS_km_cache(base + j) / m_L + (1 - A) * m_logfactorial_cache[N_k(j) + N_k(new_cluster)];
                // }

                entropy_type tmp_dS0 = dS_k(Eigen::seq(0, new_cluster)).sum();
                // entropy_type tmp_dS0 = 0;
                // for (size_type j = 0; j <= new_cluster; j++) {
                //     tmp_dS0 += dS_k[j];
                // }

                m_result_dS0[result_index] = tmp_dS0;
                m_result_dQ_km[result_index] = best_dQ_km;
                m_result_dS_km[result_index] = dS_k(new_cluster);
                m_result_merge_km(0, result_index) = best_k;
                m_result_merge_km(1, result_index) = best_m;
                
                ++result_index;
            }
        }
    }

    template<typename T>
    void CEO::remove_linear_cache_items_km(Eigen::Array<T, Eigen::Dynamic, 1> & cache, std::size_t num_clusters, size_type k, size_type m) {
        if (k == num_clusters - 2) return;
        if (m > k + 1) { // m != k + 1
            cache(Eigen::seq(k, m - 2)) = cache(Eigen::seq(k + 1, m - 1));
        }
        cache(Eigen::seq(m - 1, num_clusters - 3)) = cache(Eigen::seq(m + 1, num_clusters - 1));
    }

    void CEO::remove_N_iak_items_km(IntegerArrayXX & N_iak, std::size_t num_clusters, size_type k, size_type m) {
        if (k == num_clusters - 2) return;
        if (m > k + 1) { // m != k + 1
            N_iak(Eigen::all, Eigen::seq(k * m_D, ((m - 2) + 1) * m_D - 1)) = N_iak(Eigen::all, Eigen::seq((k + 1) * m_D, ((m - 1) + 1) * m_D - 1));
        }
        N_iak(Eigen::all, Eigen::seq((m - 1) * m_D, ((num_clusters - 3) + 1) * m_D - 1)) = N_iak(Eigen::all, Eigen::seq((m + 1) * m_D, ((num_clusters - 1) + 1) * m_D - 1));
    }

    void CEO::remove_paired_cache_items_km(FloatingPointArrayX & cache, std::size_t num_clusters, size_type k, size_type m) {
        std::size_t dst, src_block_begin, src_block_end, j;
        if (k == 0) {
            dst = 0;
            src_block_begin = 2;
            j = 3;
            src_block_end = 2;
        } else {
            dst = cache_index_km(0, k);
            src_block_begin = dst + k; // == cache_index_km(0, k + 1)
            j = k + 1;
            src_block_end = src_block_begin + k - 1; // == cache_index_km(k, k + 1) - 1
        }
        // print(f'dst={dst} src_block_begin={src_block_begin} src_block_end={src_block_end} j={j}')

        while (j < m) {
            // print(f'(a)copy: ({dst}..{dst + src_block_end - src_block_begin}) <- ({src_block_begin}..{src_block_end})', end=' ')
            cache(Eigen::seq(dst, dst + src_block_end - src_block_begin)) = cache(Eigen::seq(src_block_begin, src_block_end));
            dst += src_block_end - src_block_begin + 1;

            src_block_begin = src_block_end + 2;
            src_block_end += j;
            ++j;
            // print(f'new: dst={dst} src_block_begin={src_block_begin} src_block_end={src_block_end} j={j}')
            // for i in cache: print('{:02d} '.format(i), end='')
            // print()
        }

        // if m == 1, src_block_end = cache_index_km(0, m) - 1 will be -1 and underflowed because src_block_end is unsigned
        if (m > 1) {
            src_block_end = cache_index_km(0, m) - 1;
            if (src_block_begin <= src_block_end) {
                // print(f'(b)copy: ({dst}..{dst + src_block_end - src_block_begin}) <- ({src_block_begin}..{src_block_end})', end=' ')
                cache(Eigen::seq(dst, dst + src_block_end - src_block_begin)) = cache(Eigen::seq(src_block_begin, src_block_end));
                dst += src_block_end - src_block_begin + 1;
                // print(f'new: dst={dst} src_block_begin={src_block_begin} src_block_end={src_block_end} j={j}')
                // for i in cache: print('{:02d} '.format(i), end='')
                // print()
            }

            j = m + 1;
            src_block_begin = src_block_end + j; // == cache_index_km(0, m + 1)
            src_block_end = src_block_begin + k - 1; // == cache_index_km(k, m + 1) - 1
            // print(f'(k)new: dst={dst} src_block_begin={src_block_begin} src_block_end={src_block_end} j={j}')
        } else { // m == 1
            j = m + 1;
            src_block_begin = -1 + j; // == cache_index_km(0, m + 1)
            src_block_end = src_block_begin + k - 1; // == cache_index_km(k, m + 1) - 1
            // print(f'(k)new: dst={dst} src_block_begin={src_block_begin} src_block_end={src_block_end} j={j}')
        }

        while (j < num_clusters) {
            if (src_block_begin <= src_block_end) {
                // print(f'(k)copy: ({dst}..{dst + src_block_end - src_block_begin}) <- ({src_block_begin}..{src_block_end})')
                cache(Eigen::seq(dst, dst + src_block_end - src_block_begin)) = cache(Eigen::seq(src_block_begin, src_block_end));
                dst += src_block_end - src_block_begin + 1;
                // for i in cache: print('{:02d} '.format(i), end='')
                // print()
            }

            src_block_begin = src_block_end + 2;
            src_block_end += m - k; // == cache_index_km(m, m + 1) - 1
            // print(f'(m)new: dst={dst} src_block_begin={src_block_begin} src_block_end={src_block_end} j={j}')
            if (src_block_begin <= src_block_end) {
                // print(f'(m)copy: ({dst}..{dst + src_block_end - src_block_begin}) <- ({src_block_begin}..{src_block_end})')
                cache(Eigen::seq(dst, dst + src_block_end - src_block_begin)) = cache(Eigen::seq(src_block_begin, src_block_end));
                dst += src_block_end - src_block_begin + 1;
                // for i in cache: print('{:02d} '.format(i), end='')
                // print()
            }

            src_block_begin = src_block_end + 2;
            src_block_end += j - m + k;
            ++j;
            // print(f'(k)new: dst={dst} src_block_begin={src_block_begin} src_block_end={src_block_end} j={j}')
        }

        src_block_end = cache_index_km(0, num_clusters) - 1;
        if (src_block_begin <= src_block_end) {
            // print(f'(f)copy: ({dst}..{dst + src_block_end - src_block_begin}) <- ({src_block_begin}..{src_block_end})')
            cache(Eigen::seq(dst, dst + src_block_end - src_block_begin)) = cache(Eigen::seq(src_block_begin, src_block_end));
            dst += src_block_end - src_block_begin + 1;
            // for i in cache: print('{:02d} '.format(i), end='')
            // print()
        }

        // print(cache_index_km(0, cache_N - 2) - 1)
    }

    void CEO::grid_search() {
        const std::size_t q = m_A.size() / m_num_threads;
        const std::size_t r = m_A.size() % m_num_threads;
        std::size_t chunk_begin_point = 0; 
        std::size_t num_workers = 0;
        std::vector<std::future<void>> threadhandles(m_num_threads);
        for (std::size_t j = 0; j < m_num_threads; ++j) {
            std::size_t chunk_size = (j < r) ? (q + 1) : q;
            if (chunk_size == 0) break;
            ++num_workers;            
            std::size_t chunk_end_point = chunk_begin_point + chunk_size - 1;
            threadhandles[j] = m_thread_pool.push([this, chunk_begin_point, chunk_end_point](int thread_id) {
                this->grid_search_worker(chunk_begin_point, chunk_end_point);
            });
            chunk_begin_point += chunk_size;
        }

        for (std::size_t j = 0; j < num_workers; ++j) {
            threadhandles[j].wait();
        }
    }

    void CEO::save_npz(std::string npz_path) const {
        cnpy::npz_save(npz_path, NPZ_N_ia,                m_N_ia.data(),             {m_D, m_L},                               "w");
        cnpy::npz_save(npz_path, NPZ_Stilde_km_cache,     &m_Stilde_km_cache[0],     {m_Stilde_km_cache.size()},               "a");
        cnpy::npz_save(npz_path, NPZ_dS0,                 &m_result_dS0[0],          {m_result_dS0.size()},                    "a");
        cnpy::npz_save(npz_path, NPZ_dS_km,               &m_result_dS_km[0],        {m_result_dS_km.size()},                  "a");
        cnpy::npz_save(npz_path, NPZ_dQ_km,               &m_result_dQ_km[0],        {m_result_dQ_km.size()},                  "a");
        cnpy::npz_save(npz_path, NPZ_merge_km,            m_result_merge_km.data(),  {(std::size_t)m_A.size() * (m_N - 1), 2}, "a");
    }

}

