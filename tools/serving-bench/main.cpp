#include "harness.h"

#include "llama.h"

#include <cstdio>

int main(int argc, char ** argv) {
    serving_bench::harness_config cfg;
    const auto pr = serving_bench::parse_cli(argc, argv, cfg);
    if (pr == serving_bench::parse_result::help) {
        return 0;
    }
    if (pr == serving_bench::parse_result::error) {
        return 1;
    }

    llama_backend_init();

    std::fprintf(stderr, "[serving-bench v0.0 stub]\n");
    serving_bench::print_config(stderr, cfg);

    llama_backend_free();
    return 0;
}
