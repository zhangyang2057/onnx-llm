//
//  cli_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "llm.hpp"
#include <fstream>
#include <stdlib.h>
#include <sys/stat.h>
#include <dirent.h>

void benchmark(Llm* llm, std::string prompt_file) {
    std::cout << "prompt file is " << prompt_file << std::endl;
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        // prompt start with '#' will be ignored
        if (prompt.substr(0, 1) == "#") {
            continue;
        }
        std::string::size_type pos = 0;
        while ((pos = prompt.find("\\n", pos)) != std::string::npos) {
            prompt.replace(pos, 2, "\n");
            pos += 1;
        }
        prompts.push_back(prompt);
    }
    int prompt_len = 0;
    int decode_len = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    for (int i = 0; i < prompts.size(); i++) {
        llm->response(prompts[i]);
        prompt_len += llm->prompt_len_;
        decode_len += llm->gen_seq_len_;
        prefill_time += llm->prefill_us_;
        decode_time += llm->decode_us_;
    }
    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;
    printf("\n#################################\n");
    printf("prompt tokens num  = %d\n", prompt_len);
    printf("decode tokens num  = %d\n", decode_len);
    printf("prefill time = %.2f s\n", prefill_s);
    printf(" decode time = %.2f s\n", decode_s);
    printf("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);
    printf(" decode speed = %.2f tok/s\n", decode_len / decode_s);
    printf("##################################\n");
}

std::vector<std::string> get_input_list(const char *dataset_path) {
    std::vector<std::string> v;
    v.reserve(10000);

    DIR *d;
    struct dirent *dir;
    if ((d = opendir(dataset_path)) != NULL)
    {
        std::string base(dataset_path);
        while ((dir = readdir(d)) != NULL)
        {
            struct stat statbuf;
            std::string file_name = base + "/" + std::string(dir->d_name);
            stat(file_name.c_str(), &statbuf);
            if (S_ISREG(statbuf.st_mode))
            {
                v.push_back(file_name);
            }
        }
        closedir(d);
    }
    else
    {
        printf("Unable to open %s\n", dataset_path);
    }

    return v;
}


void evaluate(Llm* llm, const std::string &dataset_path) {
    std::cout << "dataset_path = " << dataset_path << std::endl;

    // read dataset directory
    auto file_list = get_input_list(dataset_path.c_str());
    std::sort(file_list.begin(), file_list.end());
    auto size = file_list.size();
    size = 1020;
    for (size_t i = 0; i < size; i++)
    {
        llm->response(i, file_list[i]);
    }
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " model_dir mode <prompt.txt | dataset_path>" << std::endl;
        return 0;
    }
    std::string model_dir = argv[1];
    std::cout << "model path is " << model_dir << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(model_dir));
    llm->load();
    if (strcmp(argv[2], "chat") == 0) {
        llm->chat();
    } else if (strcmp(argv[2], "benchmark") == 0){
        std::string prompt_file = argv[3];
        benchmark(llm.get(), prompt_file);
    } else {
        std::string dataset_path = argv[3];
        evaluate(llm.get(), dataset_path);
    }

    return 0;
}
