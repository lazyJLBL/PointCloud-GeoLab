#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

using Point = std::array<double, 3>;

struct Node {
  int index = -1;
  int axis = 0;
  std::unique_ptr<Node> left;
  std::unique_ptr<Node> right;
};

class KDTree {
 public:
  explicit KDTree(std::vector<Point> points) : points_(std::move(points)) {
    std::vector<int> indices(points_.size());
    std::iota(indices.begin(), indices.end(), 0);
    root_ = build(indices, 0);
  }

  std::pair<int, double> nearest(const Point& query) const {
    int best_index = -1;
    double best_dist2 = std::numeric_limits<double>::infinity();
    search(root_.get(), query, best_index, best_dist2);
    return {best_index, std::sqrt(best_dist2)};
  }

 private:
  std::unique_ptr<Node> build(std::vector<int>& indices, int depth) {
    if (indices.empty()) {
      return nullptr;
    }
    const int axis = depth % 3;
    const auto mid = indices.begin() + static_cast<long long>(indices.size() / 2);
    std::nth_element(indices.begin(), mid, indices.end(), [&](int a, int b) {
      return points_[a][axis] < points_[b][axis];
    });

    auto node = std::make_unique<Node>();
    node->index = *mid;
    node->axis = axis;

    std::vector<int> left(indices.begin(), mid);
    std::vector<int> right(mid + 1, indices.end());
    node->left = build(left, depth + 1);
    node->right = build(right, depth + 1);
    return node;
  }

  void search(const Node* node, const Point& query, int& best_index, double& best_dist2) const {
    if (node == nullptr) {
      return;
    }

    const double dist2 = squared_distance(points_[node->index], query);
    if (dist2 < best_dist2 || (dist2 == best_dist2 && node->index < best_index)) {
      best_dist2 = dist2;
      best_index = node->index;
    }

    const int axis = node->axis;
    const double delta = query[axis] - points_[node->index][axis];
    const Node* near_branch = delta <= 0.0 ? node->left.get() : node->right.get();
    const Node* far_branch = delta <= 0.0 ? node->right.get() : node->left.get();
    search(near_branch, query, best_index, best_dist2);
    if (delta * delta <= best_dist2) {
      search(far_branch, query, best_index, best_dist2);
    }
  }

  static double squared_distance(const Point& a, const Point& b) {
    double sum = 0.0;
    for (int i = 0; i < 3; ++i) {
      const double d = a[i] - b[i];
      sum += d * d;
    }
    return sum;
  }

  std::vector<Point> points_;
  std::unique_ptr<Node> root_;
};

int main() {
  std::vector<Point> points = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 2.0, 0.0},
      {0.0, 0.0, 3.0},
      {0.3, 0.2, 0.1},
  };
  const Point query = {0.25, 0.15, 0.05};
  const KDTree tree(points);
  const auto [index, distance] = tree.nearest(query);

  std::cout << "Nearest index: " << index << "\n";
  std::cout << "Distance: " << distance << "\n";
  return index == 4 ? 0 : 1;
}
