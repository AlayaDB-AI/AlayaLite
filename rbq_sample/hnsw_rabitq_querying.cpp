#include <cstdint>
#include <iostream>
#include <vector>

#include "index/graph/rbq_plus_hnsw/rbq_plus_hnsw.hpp"
#include "utils/rbq_utils/io.hpp"
#include "utils/rbq_utils/stopw.hpp"

std::vector<size_t> efs = {10,  20,  40,  50,  60,  80,  100, 150,  170,  190, 200,
                           250, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000};

size_t test_round = 3;
size_t topk = 10;

using data_type = alaya::RowMajorArray<float>;
using gt_type = alaya::RowMajorArray<uint32_t>;

int main(int argc, char* argv[]) { //NOLINT
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <arg1> <arg2> <arg3> \n"
                  << "arg1: path for index \n"
                  << "arg2: path for query file, format .fvecs\n"
                  << "arg3: path for groundtruth file format .ivecs\n";
        exit(1);
    }

    char* index_file = argv[1];
    char* query_file = argv[2];
    char* gt_file = argv[3];

    data_type query;
    gt_type gt;
    alaya::load_vecs<float, data_type>(query_file, query);
    alaya::load_vecs<uint32_t, gt_type>(gt_file, gt);
    size_t nq = query.rows();
    size_t total_count = nq * topk;

    alaya::RBQ_HNSW<> hnsw;
    // alaya::MetricType metric_type = alaya::MetricType::L2;
    // if (argc > 4) {
    //     std::string metric_str(argv[4]);
    //     if (metric_str == "ip" || metric_str == "IP") {
    //         metric_type = alaya::MetricType::IP;
    //     }
    // }
    // if (metric_type == alaya::MetricType::IP) {
    //     std::cout << "Metric Type: IP\n";
    // } else if (metric_type == alaya::MetricType::L2) {
    //     std::cout << "Metric Type: L2\n";
    // }
    hnsw.load(index_file);

    alaya::StopW stopw;
    auto nefs = efs;
    size_t length = nefs.size();

    std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(length));
    std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(length));

    std::cout << "search start >.....\n";
    std::cout << "length:"<<length<<"  test_round: "<<test_round<<"\n";

    for (size_t i_probe = 0; i_probe < length; ++i_probe) {
        for (size_t r = 0; r < test_round; r++) {
            size_t ef = nefs[i_probe];
            size_t total_correct = 0;
            float total_time = 0;

            std::cout<<"query_num: "<<nq<<" topk: "<<topk<<" ef: "<<ef<<"\n";

            auto start = std::chrono::high_resolution_clock::now();

            std::vector<std::vector<std::pair<float,uint32_t>>> res =
                hnsw.search(query.data(), nq, topk, ef, 1);

            auto end = std::chrono::high_resolution_clock::now();
            float elapsed_us =
                std::chrono::duration<float, std::micro>(end - start).count();

            total_time += elapsed_us;

            for (size_t i = 0; i < nq; i++) {
                for (size_t j = 0; j < topk; j++) {
                    for (size_t k = 0; k < topk; k++) {
                        if (gt(i, k) == res[i][j].second) {
                            total_correct++;
                            break;
                        }
                    }
                }
            }

            float qps = static_cast<float>(nq) / ((total_time) / 1e6F);

            float recall =
                static_cast<float>(total_correct) / static_cast<float>(total_count);

            all_qps[r][i_probe] = qps;
            all_recall[r][i_probe] = recall;
        }
    }

    std::cerr<<"ready to calculate avg...\n";
    auto avg_qps = alaya::horizontal_avg(all_qps);
    auto avg_recall = alaya::horizontal_avg(all_recall);

    std::cout << "EF\tQPS\tRecall\t"

                 "\n";
    for (size_t i = 0; i < avg_qps.size(); ++i) {
        std::cout << efs[i] << '\t' << avg_qps[i] << '\t' << avg_recall[i] << '\t' << '\n';
    }

}