#include <algorithm>
#include <iostream>
#include <random>
#include <ranges>
#include <vector>

int main() {
  std::vector<int> v(10);
  std::random_device rd;
  std::uniform_int_distribution<int> dist(1, 9);
  std::uniform_real_distribution<double> dist1(1.0, 9.0);

  std::cout << dist(rd) << std::endl;

  std::generate(v.begin(), v.end(), [&]() { return dist(rd); });

  for (auto i : v) std::cout << i << " ";

  //   std::ranges::shuffle(v, rd);

  return 0;
}