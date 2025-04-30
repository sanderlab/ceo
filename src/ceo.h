#pragma once

#include <Eigen/Dense>
#include <cnpy.h>
#include "ctpl/ctpl.h"

namespace ceo {
    const std::string NPZ_onehot {"onehot"};
    const std::string NPZ_A {"A"};
    const std::string NPZ_N_ia {"N_ia"};
    const std::string NPZ_Stilde_km_cache {"Stilde_km_cache"};
    const std::string NPZ_dS0 {"dS0"};
    const std::string NPZ_dS_km {"dS_km"};
    const std::string NPZ_dQ_km {"dQ_km"};
    const std::string NPZ_merge_km {"merge_km"};

    using size_type = uint16_t;
    using entropy_type = double;

    class CEO {
    private:
        using IntegerArrayX = Eigen::Array<size_type, Eigen::Dynamic, 1>;
        using IntegerArrayXX = Eigen::Array<size_type, Eigen::Dynamic, Eigen::Dynamic>;
        using FloatingPointArrayX = Eigen::Array<entropy_type, Eigen::Dynamic, 1>;
        using FloatingPointArrayXX = Eigen::Array<entropy_type, Eigen::Dynamic, Eigen::Dynamic>;
        using merge_ops_type = Eigen::Array<size_type, 2, Eigen::Dynamic>;

        int m_num_threads;
        bool m_ignore1;
        ctpl::thread_pool m_thread_pool;

        std::size_t m_N;
        std::size_t m_D;
        std::size_t m_L;
        IntegerArrayXX m_N_iak;
        IntegerArrayXX m_N_ia;
        FloatingPointArrayX m_A;

        merge_ops_type m_result_merge_km;
        std::vector<entropy_type> m_result_dS0;
        std::vector<entropy_type> m_result_dS_km;
        std::vector<entropy_type> m_result_dQ_km;

        // cache-related
        std::vector<entropy_type> m_logfactorial_cache;
        std::vector<entropy_type> m_log_cache;
        FloatingPointArrayX m_dS_km_cache;
        std::vector<entropy_type> m_Stilde_km_cache;

    private:
        inline auto cache_index_km(size_type k, size_type m) { // use auto to avoid overflow
            return (m - 1) * m / 2 + k;
        }

        auto km_cache_index(std::size_t cache_index);
        entropy_type real_dS_km(size_type k, size_type m, IntegerArrayXX const & N_iak, IntegerArrayX const & N_k);
        void initialize_dS_km_cache();
        void initialize_dS_km_cache_worker(std::size_t chunk_begin, std::size_t chunk_end, IntegerArrayX const & N_k);
        void initialize_dS_km_cache_mt();
        void grid_search_worker(std::size_t const chunk_begin_point, std::size_t const chunk_end_point);
        
        template<typename T>
        void remove_linear_cache_items_km(Eigen::Array<T, Eigen::Dynamic, 1> & cache, std::size_t num_clusters, size_type k, size_type m);
        
        void remove_N_iak_items_km(IntegerArrayXX & N_iak, std::size_t num_clusters, size_type k, size_type m);
        void remove_paired_cache_items_km(FloatingPointArrayX & cache, std::size_t num_clusters, size_type k, size_type m);

    public:
        CEO() = delete;
        CEO(cnpy::npz_t & npz, int num_threads, bool ignore1);
        void grid_search();
        void save_npz(std::string npz_path) const;
    };
}
