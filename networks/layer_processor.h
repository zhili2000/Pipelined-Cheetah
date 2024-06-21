#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>
#include <tuple>

// Class that handles the processing of a layer.
class LayerProcessor {
public:
    LayerProcessor(std::function<void(uint64_t*, uint64_t**, int)> layerFunc, size_t initialPendingTasks)
        : layerFunc(layerFunc), stop(false), pendingTasks(initialPendingTasks) {}

    void addTask(uint64_t* input, int task_number
  ) {
        uint64_t* output = nullptr;
        {
          std::unique_lock<std::mutex> lock(mtx);
          tasks.push(std::make_tuple(input, &output, task_number));
        }
        cv.notify_one();
    }

    void start() {
      worker = std::thread([this]() {
        while (true) {
          std::tuple<uint64_t*, uint64_t**, int> task;
          {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this]() { return !tasks.empty() || stop; });
            if (stop && tasks.empty()) {
              break;
            }
            task = tasks.front();
            tasks.pop();
          }
          layerFunc(std::get<0>(task), std::get<1>(task), std::get<2>(task));
          if (nextLayerProcessor) {
              // Pass current output as next input.
              nextLayerProcessor->addTask(*std::get<1>(task), std::get<2>(task));
          }
          {
            std::unique_lock<std::mutex> lock(mtx);
            pendingTasks -= 1;
            if (pendingTasks == 0) {
              cv.notify_all();
            }
          }
        }
      });
    }

    void setNextLayerProcessor(LayerProcessor* nextProcessor) {
      nextLayerProcessor = nextProcessor;
    }

    size_t getPendingTasks() const {
      return pendingTasks.load();
    }

    void stopProcessing() {
      if (stop == false) {
        {
          std::unique_lock<std::mutex> lock(mtx);
          stop = true;
          // Notify all threads can be stopped.
          cv.notify_all();
        }
      }

      if (worker.joinable()) {
        worker.join();
      }
    }

    std::mutex mtx;
    bool stop;

private:
    std::function<void(uint64_t*, uint64_t**, int)> layerFunc;
    std::queue<std::tuple<uint64_t*, uint64_t**, int>> tasks;
    std::thread worker;
    std::condition_variable cv;
    LayerProcessor* nextLayerProcessor = nullptr;
    std::atomic<size_t> pendingTasks;
};