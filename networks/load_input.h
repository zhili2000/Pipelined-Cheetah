#include <vector>
#include <iostream>
#include "library_fixed.h"

std::vector<uint64_t*> loadInput(int im_batch_size) {
    std::vector<uint64_t*> images;

    if (party == CLIENT) {
      std::vector<std::string> filenames;
      std::string line;

      // Read all filenames from stdin
      while (std::getline(std::cin, line)) {
          filenames.push_back(line);
      }

      uint64_t __tmp_in_tmp0;

      for (const std::string& filename : filenames) {
          std::ifstream file(filename, std::ios::binary);
          if (!file) {
              std::cerr << "Failed to open file: " << filename << std::endl;
              continue;
          }

          uint64_t* tmp0 = make_array<uint64_t>(1, 227, 227, 3);
          std::cerr << "Loading input from " << filename << "..." << std::endl;

          for (uint64_t i0 = 0; i0 < 1; i0++) {
              for (uint64_t i1 = 0; i1 < 227; i1++) {
                  for (uint64_t i2 = 0; i2 < 227; i2++) {
                      for (uint64_t i3 = 0; i3 < 3; i3++) {
                          file >> __tmp_in_tmp0;
                          Arr4DIdxRowM(tmp0, 1, 227, 227, 3, i0, i1, i2, i3) = 1;
                      }
                  }
              }
          }
          images.push_back(tmp0);
          file.close();
      }
    } else {
        // SERVER party reads directly from stdin
        for (int i = 0; i < im_batch_size; i++) {
            uint64_t* tmp0 = make_array<uint64_t>(1, 227, 227, 3);
            std::cerr << "Loading input from stdin..." << std::endl;
            for (uint64_t i0 = 0; i0 < 1; i0++) {
                for (uint64_t i1 = 0; i1 < 227; i1++) {
                    for (uint64_t i2 = 0; i2 < 227; i2++) {
                        for (uint64_t i3 = 0; i3 < 3; i3++) {
                            Arr4DIdxRowM(tmp0, 1, 227, 227, 3, i0, i1, i2, i3) = 0;
                        }
                    }
                }
            }
            images.push_back(tmp0);
        }
    }

    return images;
}