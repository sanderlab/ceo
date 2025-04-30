#include <iostream>
#include <filesystem>
#include <fmt/core.h>
#include <thread>
#include <boost/program_options.hpp>
#include <cnpy.h>

#include "ceo.h"

namespace po = boost::program_options;
namespace fs = std::filesystem;

int main(int argc, char **argv) {
    fs::path base_dir(fs::current_path());
    fs::path output_dir(fs::current_path());
    fs::path in_npz_path;
    fs::path out_npz_path;
    std::size_t num_threads = 1;
    bool ignore1 = false;

    po::options_description options("Options");
    options.add_options()
        ("help", "print this help message")
        ("in-npz", po::value<std::string>(), "input file")
        ("out-npz", po::value<std::string>(), "output file")
        ("threads", po::value<int>()->default_value(1), "number of threads (default is 1)")
        ("ignore1", "use 0 for Stilde for single-sequence clusters");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << options << "\n";
        return 0;
    }

    if (vm.count("in-npz")) {
        in_npz_path = vm["in-npz"].as<std::string>();
        if (in_npz_path.is_relative()) {
            in_npz_path = (base_dir / in_npz_path).lexically_normal();
        }
    } else {
        throw std::invalid_argument("Missing input file.");
    }

    if (vm.count("out-npz")) {
        out_npz_path = vm["out-npz"].as<std::string>();
        if (out_npz_path.is_relative()) {
            out_npz_path = (base_dir / out_npz_path).lexically_normal();
        }
        auto output_dir = out_npz_path.parent_path();
        if (!fs::is_directory(output_dir)) {
            fs::create_directories(output_dir);
        }
    } else {
        throw std::invalid_argument("Missing output file.");
    }

    if (vm.count("threads")) {
        num_threads = vm["threads"].as<int>();
    }

    const auto processor_count = std::thread::hardware_concurrency();
    if ((num_threads == 0) || (num_threads > processor_count)) {
        num_threads = processor_count;
    }

    if (vm.count("ignore1")) {
        ignore1 = true;
        std::cout << "Use 0 for Stilde for single-sequence clusters." << std::endl;
    }

    cnpy::npz_t npz = cnpy::npz_load(in_npz_path);
    ceo::CEO ceo(npz, num_threads, ignore1);
    ceo.grid_search();
    ceo.save_npz(out_npz_path);

    return 0;
}
