#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

int main() {
  auto old = steady_clock::now();
  int i = 10000000;
  while (i) {
    i--;
  }
  auto now = steady_clock::now();

  auto dur = now - old;
  cout << duration_cast<milliseconds>(dur).count() << "ms" << endl;

  return 0;
}