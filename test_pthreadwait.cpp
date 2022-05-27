#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>

#define COUNT 1000000

double sum=0.0;
std::mutex m;
std::condition_variable cv;
std::atomic<bool> flag(true);
std::atomic<bool> finished(false);
// std::chrono::time_point<system_clock, std::chrono::nanoseconds >
std::chrono::high_resolution_clock::time_point start;

void waits() {
    
    while(!finished)
    {
        // std::cout << "finished" << std::endl;
        std::unique_lock<std::mutex> lk(m);
        flag = false;
        cv.wait(lk, []() {return flag == true; });
        // 这里获取时间 计算时间 累加耗时
        auto end = std::chrono::high_resolution_clock::now();
        sum += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
}

void signals() {
    while(flag) {}
    int count = COUNT;
    
    for(int i = 0; i < count; i++)
    {
        while(flag) {}
        std::unique_lock<std::mutex> lk(m);
        flag = true;
        lk.unlock();
        //这里获取时间
        start = std::chrono::high_resolution_clock::now();
        cv.notify_one();
    }

    finished = true;
    flag = true;
    cv.notify_one();
}

int main()
{
    std::thread t1(waits), t2(signals);
    t1.join();
    t2.join();
    std::cout << "sum: " << sum / COUNT << "ns" << std::endl;
}