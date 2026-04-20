#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

static inline double sgn(double v) {
  if (v > 0.0) return 1.0;
  if (v < 0.0) return -1.0;
  return 0.0;
}

static std::vector<double> average_ranks(const std::vector<double>& values) {
  const int n = static_cast<int>(values.size());
  std::vector<int> ord(n);
  std::iota(ord.begin(), ord.end(), 0);
  std::sort(ord.begin(), ord.end(), [&](int a, int b) { return values[a] < values[b]; });

  std::vector<double> ranks(n, 0.0);
  int i = 0;
  while (i < n) {
    int j = i + 1;
    while (j < n && values[ord[j]] == values[ord[i]]) {
      ++j;
    }
    const double avg = 0.5 * (static_cast<double>(i + 1) + static_cast<double>(j));
    for (int k = i; k < j; ++k) {
      ranks[ord[k]] = avg;
    }
    i = j;
  }
  return ranks;
}

static std::vector<std::vector<int>> groups_by_value(const std::vector<double>& values) {
  std::vector<double> unique_vals = values;
  std::sort(unique_vals.begin(), unique_vals.end());
  unique_vals.erase(std::unique(unique_vals.begin(), unique_vals.end()), unique_vals.end());

  std::vector<std::vector<int>> groups(unique_vals.size());
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    int idx = static_cast<int>(
        std::lower_bound(unique_vals.begin(), unique_vals.end(), values[i]) - unique_vals.begin());
    groups[idx].push_back(i);
  }
  return groups;
}

class FenwickTreeI {
  std::vector<int> tree;
  int n;

 public:
  explicit FenwickTreeI(int n_) : tree(static_cast<size_t>(n_ + 1), 0), n(n_) {}

  void update(int i, int delta = 1) {
    for (; i <= n; i += i & (-i)) tree[static_cast<size_t>(i)] += delta;
  }

  int query(int i) const {
    int s = 0;
    for (; i > 0; i -= i & (-i)) s += tree[static_cast<size_t>(i)];
    return s;
  }
};

static std::vector<int> compress_vals(const std::vector<double>& vals, int& M) {
  std::vector<double> sorted_unique = vals;
  std::sort(sorted_unique.begin(), sorted_unique.end());
  sorted_unique.erase(std::unique(sorted_unique.begin(), sorted_unique.end()), sorted_unique.end());
  M = static_cast<int>(sorted_unique.size());
  std::vector<int> compressed(vals.size());
  for (size_t i = 0; i < vals.size(); ++i) {
    compressed[i] = static_cast<int>(
                        std::lower_bound(sorted_unique.begin(), sorted_unique.end(), vals[i]) -
                        sorted_unique.begin()) +
                    1;
  }
  return compressed;
}

static std::vector<std::vector<int>> group_by_sorted_keys(const std::vector<int>& keys, int M) {
  std::vector<std::vector<int>> groups(static_cast<size_t>(M));
  for (int i = 0; i < static_cast<int>(keys.size()); ++i) {
    groups[static_cast<size_t>(keys[static_cast<size_t>(i)] - 1)].push_back(i);
  }
  return groups;
}

struct JointCounts {
  std::vector<int> count_both_less;
  std::vector<int> count_x_less_y_eq;
  std::vector<int> count_x_eq_y_less;
  std::vector<int> count_both_eq;
};

static JointCounts compute_joint_counts(
    const std::vector<int>& Xc, int Mx, const std::vector<int>& Yc, int My, int n) {
  auto x_groups = group_by_sorted_keys(Xc, Mx);
  JointCounts out;
  out.count_both_less.assign(static_cast<size_t>(n), 0);
  out.count_x_less_y_eq.assign(static_cast<size_t>(n), 0);
  out.count_x_eq_y_less.assign(static_cast<size_t>(n), 0);
  out.count_both_eq.assign(static_cast<size_t>(n), 0);

  FenwickTreeI tree(My);
  for (int g = 0; g < Mx; ++g) {
    for (int idx : x_groups[static_cast<size_t>(g)]) {
      int yc = Yc[static_cast<size_t>(idx)];
      if (yc > 1) out.count_both_less[static_cast<size_t>(idx)] = tree.query(yc - 1);
      out.count_x_less_y_eq[static_cast<size_t>(idx)] = tree.query(yc) - tree.query(yc - 1);
    }
    for (int idx : x_groups[static_cast<size_t>(g)]) tree.update(Yc[static_cast<size_t>(idx)]);
  }

  for (int g = 0; g < Mx; ++g) {
    auto indices = x_groups[static_cast<size_t>(g)];
    if (indices.empty()) continue;
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
      return Yc[static_cast<size_t>(a)] < Yc[static_cast<size_t>(b)];
    });
    int cumcount = 0;
    int prev_yr = -1;
    int same_count = 0;
    for (int idx : indices) {
      int yr = Yc[static_cast<size_t>(idx)];
      if (yr == prev_yr) {
        same_count++;
      } else {
        cumcount += same_count;
        same_count = 1;
        prev_yr = yr;
      }
      out.count_x_eq_y_less[static_cast<size_t>(idx)] = cumcount;
    }
    std::map<int, int> pair_counts;
    for (int idx : x_groups[static_cast<size_t>(g)]) pair_counts[Yc[static_cast<size_t>(idx)]]++;
    for (int idx : x_groups[static_cast<size_t>(g)]) {
      out.count_both_eq[static_cast<size_t>(idx)] = pair_counts[Yc[static_cast<size_t>(idx)]];
    }
  }
  return out;
}

struct PairCounts {
  double concordant;
  double discordant;
};

static PairCounts count_concordant_discordant(
    py::array_t<double, py::array::c_style | py::array::forcecast> x_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> y_in) {
  const int n = static_cast<int>(x_in.shape(0));
  auto x = x_in.unchecked<1>();
  auto y = y_in.unchecked<1>();

  std::map<double, int> y_map;
  for (int i = 0; i < n; ++i) y_map[y(i)];
  int rank = 1;
  for (auto& p : y_map) p.second = rank++;
  const int M = static_cast<int>(y_map.size());
  std::vector<int> Yc(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) Yc[static_cast<size_t>(i)] = y_map[y(i)];

  std::vector<int> ord(static_cast<size_t>(n));
  std::iota(ord.begin(), ord.end(), 0);
  std::sort(ord.begin(), ord.end(), [&](int a, int b) {
    if (x(a) != x(b)) return x(a) < x(b);
    return y(a) < y(b);
  });

  FenwickTreeI tree(M);
  double concordant = 0.0;
  double discordant = 0.0;
  int i = 0;
  while (i < n) {
    int j = i;
    while (j < n && x(ord[static_cast<size_t>(j)]) == x(ord[static_cast<size_t>(i)])) ++j;
    for (int k = i; k < j; ++k) {
      int yc = Yc[static_cast<size_t>(ord[static_cast<size_t>(k)])];
      concordant += tree.query(yc - 1);
      discordant += tree.query(M) - tree.query(yc);
    }
    for (int k = i; k < j; ++k) tree.update(Yc[static_cast<size_t>(ord[static_cast<size_t>(k)])]);
    i = j;
  }
  return {concordant, discordant};
}

static double pair_tie_proportion(
    py::array_t<double, py::array::c_style | py::array::forcecast> v_in) {
  const int n = static_cast<int>(v_in.shape(0));
  auto v = v_in.unchecked<1>();
  std::map<double, int> freq;
  for (int i = 0; i < n; ++i) freq[v(i)]++;
  double n_tied_pairs = 0.0;
  for (const auto& p : freq) {
    double nk = static_cast<double>(p.second);
    n_tied_pairs += nk * (nk - 1.0);
  }
  return n_tied_pairs / (static_cast<double>(n) * (n - 1.0));
}

py::dict kendall_tau_sign_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> x_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> y_in) {
  if (x_in.ndim() != 1 || y_in.ndim() != 1) throw std::invalid_argument("x and y must be 1-D arrays.");
  if (x_in.shape(0) != y_in.shape(0)) throw std::invalid_argument("x and y must have same length.");
  const int n = static_cast<int>(x_in.shape(0));
  const double num_pairs = static_cast<double>(n) * (n - 1.0) / 2.0;
  PairCounts pc = count_concordant_discordant(x_in, y_in);
  const double sum_sign = pc.concordant - pc.discordant;
  const double expectation = sum_sign / num_pairs;
  const double p_tie_y = pair_tie_proportion(y_in);
  const double scale_factor = 1.0 - p_tie_y;
  const double tau = (scale_factor > 1e-10) ? expectation / scale_factor : 0.0;
  py::dict out;
  out["tau"] = tau;
  out["expectation"] = expectation;
  return out;
}

py::array_t<double> h_bar_vec_v2_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> x_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> y_in) {
  if (x_in.ndim() != 1 || y_in.ndim() != 1) throw std::invalid_argument("x and y must be 1-D arrays.");
  if (x_in.shape(0) != y_in.shape(0)) throw std::invalid_argument("x and y must have same length.");
  const int n = static_cast<int>(x_in.shape(0));
  auto x = x_in.unchecked<1>();
  auto y = y_in.unchecked<1>();
  std::vector<double> Xv(static_cast<size_t>(n)), Yv(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    Xv[static_cast<size_t>(i)] = x(i);
    Yv[static_cast<size_t>(i)] = y(i);
  }
  int Mx, My;
  std::vector<int> Xc = compress_vals(Xv, Mx);
  std::vector<int> Yc = compress_vals(Yv, My);
  JointCounts jc = compute_joint_counts(Xc, Mx, Yc, My, n);
  py::array_t<double> out(n);
  auto o = out.mutable_unchecked<1>();
  for (int i = 0; i < n; ++i) {
    o(i) = (jc.count_both_less[static_cast<size_t>(i)] +
            0.5 * jc.count_x_eq_y_less[static_cast<size_t>(i)] +
            0.5 * jc.count_x_less_y_eq[static_cast<size_t>(i)] +
            0.25 * jc.count_both_eq[static_cast<size_t>(i)]) /
           n;
  }
  return out;
}

py::array_t<double> kernel_agc_v2_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> x_rank_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> y_rank_in, double rho) {
  if (x_rank_in.ndim() != 1 || y_rank_in.ndim() != 1) {
    throw std::invalid_argument("x_rank and y_rank must be 1-D arrays.");
  }
  if (x_rank_in.shape(0) != y_rank_in.shape(0)) {
    throw std::invalid_argument("x_rank and y_rank must have same length.");
  }
  const int N = static_cast<int>(x_rank_in.shape(0));
  auto xr = x_rank_in.unchecked<1>();
  auto yr = y_rank_in.unchecked<1>();

  std::vector<double> Gx(static_cast<size_t>(N)), Gy(static_cast<size_t>(N));
  std::vector<double> Xv(static_cast<size_t>(N)), Yv(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) {
    Gx[static_cast<size_t>(i)] = (xr(i) - 0.5) / N;
    Gy[static_cast<size_t>(i)] = (yr(i) - 0.5) / N;
    Xv[static_cast<size_t>(i)] = xr(i);
    Yv[static_cast<size_t>(i)] = yr(i);
  }

  int Mx, My;
  std::vector<int> Xc = compress_vals(Xv, Mx);
  std::vector<int> Yc = compress_vals(Yv, My);
  JointCounts jc = compute_joint_counts(Xc, Mx, Yc, My, N);

  std::vector<double> Hbar(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) {
    Hbar[static_cast<size_t>(i)] =
        (jc.count_both_less[static_cast<size_t>(i)] +
         0.5 * jc.count_x_eq_y_less[static_cast<size_t>(i)] +
         0.5 * jc.count_x_less_y_eq[static_cast<size_t>(i)] +
         0.25 * jc.count_both_eq[static_cast<size_t>(i)]) /
        N;
  }

  std::vector<int> n_y(static_cast<size_t>(My), 0);
  for (int i = 0; i < N; ++i) n_y[static_cast<size_t>(Yc[static_cast<size_t>(i)] - 1)]++;
  std::vector<int> cum_n_y(static_cast<size_t>(My + 1), 0);
  for (int m = 0; m < My; ++m) cum_n_y[static_cast<size_t>(m + 1)] = cum_n_y[static_cast<size_t>(m)] + n_y[static_cast<size_t>(m)];

  std::vector<int> n_y_eq(static_cast<size_t>(N)), n_y_lt(static_cast<size_t>(N)), n_y_gt(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) {
    int yi = Yc[static_cast<size_t>(i)] - 1;
    n_y_eq[static_cast<size_t>(i)] = n_y[static_cast<size_t>(yi)];
    n_y_lt[static_cast<size_t>(i)] = cum_n_y[static_cast<size_t>(yi)];
    n_y_gt[static_cast<size_t>(i)] = N - n_y_lt[static_cast<size_t>(i)] - n_y_eq[static_cast<size_t>(i)];
  }

  std::vector<int> n_x(static_cast<size_t>(Mx), 0);
  for (int i = 0; i < N; ++i) n_x[static_cast<size_t>(Xc[static_cast<size_t>(i)] - 1)]++;
  std::vector<int> cum_n_x(static_cast<size_t>(Mx + 1), 0);
  for (int m = 0; m < Mx; ++m) cum_n_x[static_cast<size_t>(m + 1)] = cum_n_x[static_cast<size_t>(m)] + n_x[static_cast<size_t>(m)];
  std::vector<int> n_x_eq(static_cast<size_t>(N)), n_x_lt(static_cast<size_t>(N)), n_x_gt(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) {
    int xi = Xc[static_cast<size_t>(i)] - 1;
    n_x_eq[static_cast<size_t>(i)] = n_x[static_cast<size_t>(xi)];
    n_x_lt[static_cast<size_t>(i)] = cum_n_x[static_cast<size_t>(xi)];
    n_x_gt[static_cast<size_t>(i)] = N - n_x_lt[static_cast<size_t>(i)] - n_x_eq[static_cast<size_t>(i)];
  }

  std::vector<double> w_full(static_cast<size_t>(N)), w_half(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) {
    w_full[static_cast<size_t>(i)] = n_y_gt[static_cast<size_t>(i)] + 0.5 * n_y_eq[static_cast<size_t>(i)];
    w_half[static_cast<size_t>(i)] = 0.5 * n_y_gt[static_cast<size_t>(i)] + 0.25 * n_y_eq[static_cast<size_t>(i)];
  }

  auto X_groups = group_by_sorted_keys(Xc, Mx);
  std::vector<double> w_half_group_sum(static_cast<size_t>(Mx), 0.0);
  for (int g = 0; g < Mx; ++g)
    for (int idx : X_groups[static_cast<size_t>(g)])
      w_half_group_sum[static_cast<size_t>(g)] += w_half[static_cast<size_t>(idx)];

  std::vector<double> sum_xeq_whalf(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) sum_xeq_whalf[static_cast<size_t>(i)] = w_half_group_sum[static_cast<size_t>(Xc[static_cast<size_t>(i)] - 1)];

  std::vector<double> sum_xlt_wfull(static_cast<size_t>(N), 0.0);
  double running_sum = 0.0;
  for (int g = 0; g < Mx; ++g) {
    for (int idx : X_groups[static_cast<size_t>(g)]) sum_xlt_wfull[static_cast<size_t>(idx)] = running_sum;
    for (int idx : X_groups[static_cast<size_t>(g)]) running_sum += w_full[static_cast<size_t>(idx)];
  }

  std::vector<double> S_i(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) S_i[static_cast<size_t>(i)] = (sum_xlt_wfull[static_cast<size_t>(i)] + sum_xeq_whalf[static_cast<size_t>(i)]) / N;

  std::vector<double> sum_nxgt_by_ygroup(static_cast<size_t>(My), 0.0);
  std::vector<double> sum_nxeq_by_ygroup(static_cast<size_t>(My), 0.0);
  for (int i = 0; i < N; ++i) {
    int yi = Yc[static_cast<size_t>(i)] - 1;
    sum_nxgt_by_ygroup[static_cast<size_t>(yi)] += n_x_gt[static_cast<size_t>(i)];
    sum_nxeq_by_ygroup[static_cast<size_t>(yi)] += n_x_eq[static_cast<size_t>(i)];
  }
  std::vector<double> cum_nxgt(static_cast<size_t>(My + 1), 0.0);
  std::vector<double> cum_nxeq(static_cast<size_t>(My + 1), 0.0);
  for (int m = 0; m < My; ++m) {
    cum_nxgt[static_cast<size_t>(m + 1)] = cum_nxgt[static_cast<size_t>(m)] + sum_nxgt_by_ygroup[static_cast<size_t>(m)];
    cum_nxeq[static_cast<size_t>(m + 1)] = cum_nxeq[static_cast<size_t>(m)] + sum_nxeq_by_ygroup[static_cast<size_t>(m)];
  }
  std::vector<double> T_m(static_cast<size_t>(My));
  for (int m = 0; m < My; ++m) {
    T_m[static_cast<size_t>(m)] =
        (cum_nxgt[static_cast<size_t>(m)] + 0.5 * cum_nxeq[static_cast<size_t>(m)] +
         0.5 * sum_nxgt_by_ygroup[static_cast<size_t>(m)] +
         0.25 * sum_nxeq_by_ygroup[static_cast<size_t>(m)]) /
        N;
  }

  py::array_t<double> out(N);
  auto o = out.mutable_unchecked<1>();
  for (int i = 0; i < N; ++i) {
    double g1 = T_m[static_cast<size_t>(Yc[static_cast<size_t>(i)] - 1)] / N;
    double g2 = S_i[static_cast<size_t>(i)] / N;
    o(i) = 4.0 * (g1 + g2 + Gx[static_cast<size_t>(i)] * Gy[static_cast<size_t>(i)] -
                  Gy[static_cast<size_t>(i)] - Gx[static_cast<size_t>(i)]) +
           1.0 - rho;
  }
  return out;
}

py::dict akc_ij_cpp(py::array_t<double, py::array::c_style | py::array::forcecast> x_in,
                    py::array_t<double, py::array::c_style | py::array::forcecast> y_in) {
  if (x_in.ndim() != 1 || y_in.ndim() != 1) {
    throw std::invalid_argument("x and y must be 1-D arrays.");
  }
  if (x_in.shape(0) != y_in.shape(0)) {
    throw std::invalid_argument("x and y must have the same length.");
  }
  const ssize_t n = x_in.shape(0);
  if (n < 2) {
    throw std::invalid_argument("Need at least 2 observations.");
  }

  auto x = x_in.unchecked<1>();
  auto y = y_in.unchecked<1>();

  std::vector<double> xvals(static_cast<size_t>(n)), yvals(static_cast<size_t>(n));
  for (ssize_t i = 0; i < n; ++i) {
    xvals[static_cast<size_t>(i)] = x(i);
    yvals[static_cast<size_t>(i)] = y(i);
  }

  int My = 0;
  std::vector<int> Yc = compress_vals(yvals, My);

  std::vector<int> ord(static_cast<size_t>(n));
  std::iota(ord.begin(), ord.end(), 0);
  std::sort(ord.begin(), ord.end(), [&](int a, int b) {
    if (xvals[static_cast<size_t>(a)] != xvals[static_cast<size_t>(b)]) {
      return xvals[static_cast<size_t>(a)] < xvals[static_cast<size_t>(b)];
    }
    return yvals[static_cast<size_t>(a)] < yvals[static_cast<size_t>(b)];
  });

  std::vector<double> H_A(static_cast<size_t>(n), 0.0);
  std::vector<double> H_B(static_cast<size_t>(n), 0.0);

  // Forward pass: contributions from x_j < x_i.
  {
    FenwickTreeI tree(My);
    int i = 0;
    while (i < static_cast<int>(n)) {
      int j = i;
      while (j < static_cast<int>(n) &&
             xvals[static_cast<size_t>(ord[static_cast<size_t>(j)])] ==
                 xvals[static_cast<size_t>(ord[static_cast<size_t>(i)])]) {
        ++j;
      }
      int total_inserted = (i > 0) ? tree.query(My) : 0;
      for (int k = i; k < j; ++k) {
        int idx = ord[static_cast<size_t>(k)];
        int yc = Yc[static_cast<size_t>(idx)];
        int below = (yc > 1) ? tree.query(yc - 1) : 0;
        int above = total_inserted - tree.query(yc);
        H_A[static_cast<size_t>(idx)] += static_cast<double>(below - above);
        H_B[static_cast<size_t>(idx)] += static_cast<double>(below + above);
      }
      for (int k = i; k < j; ++k) tree.update(Yc[static_cast<size_t>(ord[static_cast<size_t>(k)])]);
      i = j;
    }
  }

  // Backward pass: contributions from x_j > x_i.
  {
    FenwickTreeI tree(My);
    int i = static_cast<int>(n);
    while (i > 0) {
      int j = i;
      while (j > 0 &&
             xvals[static_cast<size_t>(ord[static_cast<size_t>(j - 1)])] ==
                 xvals[static_cast<size_t>(ord[static_cast<size_t>(i - 1)])]) {
        --j;
      }
      int total_inserted = tree.query(My);
      for (int k = j; k < i; ++k) {
        int idx = ord[static_cast<size_t>(k)];
        int yc = Yc[static_cast<size_t>(idx)];
        int below = (yc > 1) ? tree.query(yc - 1) : 0;
        int above = total_inserted - tree.query(yc);
        H_A[static_cast<size_t>(idx)] += static_cast<double>(above - below);
        H_B[static_cast<size_t>(idx)] += static_cast<double>(below + above);
      }
      for (int k = j; k < i; ++k) tree.update(Yc[static_cast<size_t>(ord[static_cast<size_t>(k)])]);
      i = j;
    }
  }

  // Same-X contributions to H_B only: count j != i with y_j != y_i.
  {
    int Mx = 0;
    std::vector<int> Xc = compress_vals(xvals, Mx);
    auto x_groups = group_by_sorted_keys(Xc, Mx);
    for (int g = 0; g < Mx; ++g) {
      const auto& grp = x_groups[static_cast<size_t>(g)];
      int gsize = static_cast<int>(grp.size());
      if (gsize <= 1) continue;
      std::map<int, int> y_freq;
      for (int idx : grp) y_freq[Yc[static_cast<size_t>(idx)]]++;
      for (int idx : grp) {
        int same_y = y_freq[Yc[static_cast<size_t>(idx)]] - 1;
        int diff_y = (gsize - 1) - same_y;
        H_B[static_cast<size_t>(idx)] += static_cast<double>(diff_y);
      }
    }
  }

  const double num_pairs = static_cast<double>(n) * static_cast<double>(n - 1) / 2.0;

  double sum_HA = 0.0;
  double sum_HB = 0.0;
  for (ssize_t i = 0; i < n; ++i) {
    sum_HA += H_A[static_cast<size_t>(i)];
    sum_HB += H_B[static_cast<size_t>(i)];
  }

  const double A = sum_HA / (2.0 * num_pairs);
  const double B = sum_HB / (2.0 * num_pairs);
  if (B <= 1e-12) {
    throw std::invalid_argument("AKC IJ undefined because y is almost fully tied.");
  }
  const double akc = A / B;

  py::array_t<double> ic_arr(n), dA_arr(n), dB_arr(n);
  auto ic = ic_arr.mutable_unchecked<1>();
  auto dA = dA_arr.mutable_unchecked<1>();
  auto dB = dB_arr.mutable_unchecked<1>();
  double sum_ic2 = 0.0;
  for (ssize_t i = 0; i < n; ++i) {
    const double n_dA_i = 2.0 * (H_A[static_cast<size_t>(i)] / static_cast<double>(n - 1) - A);
    const double n_dB_i = 2.0 * (H_B[static_cast<size_t>(i)] / static_cast<double>(n - 1) - B);
    const double ic_i = n_dA_i / B - (A / (B * B)) * n_dB_i;
    dA(i) = n_dA_i;
    dB(i) = n_dB_i;
    ic(i) = ic_i;
    sum_ic2 += ic_i * ic_i;
  }

  const double var_ij = sum_ic2 / static_cast<double>(n);

  py::dict out;
  out["akc"] = akc;
  out["ic"] = ic_arr;
  out["var_ij"] = var_ij;
  out["dA"] = dA_arr;
  out["dB"] = dB_arr;
  return out;
}

py::dict agc_ij_cpp(py::array_t<double, py::array::c_style | py::array::forcecast> x_in,
                    py::array_t<double, py::array::c_style | py::array::forcecast> y_in) {
  if (x_in.ndim() != 1 || y_in.ndim() != 1) {
    throw std::invalid_argument("x and y must be 1-D arrays.");
  }
  if (x_in.shape(0) != y_in.shape(0)) {
    throw std::invalid_argument("x and y must have the same length.");
  }
  const int n = static_cast<int>(x_in.shape(0));
  if (n < 2) {
    throw std::invalid_argument("Need at least 2 observations.");
  }

  auto x = x_in.unchecked<1>();
  auto y = y_in.unchecked<1>();
  std::vector<double> xvals(n), yvals(n);
  for (int i = 0; i < n; ++i) {
    xvals[i] = x(i);
    yvals[i] = y(i);
  }

  const std::vector<double> R = average_ranks(yvals);
  const std::vector<double> Q = average_ranks(xvals);
  const double mean_rank = 0.5 * (static_cast<double>(n) + 1.0);

  std::vector<double> tR(n), tQ(n);
  for (int i = 0; i < n; ++i) {
    tR[i] = R[i] - mean_rank;
    tQ[i] = Q[i] - mean_rank;
  }

  double A = 0.0;
  double B = 0.0;
  for (int i = 0; i < n; ++i) {
    A += tQ[i] * tR[i];
    B += tR[i] * tR[i];
  }
  if (B <= 1e-18) {
    throw std::invalid_argument("AGC IJ undefined because y rank variance is near zero.");
  }
  const double agc = A / B;

  const double total_tR = std::accumulate(tR.begin(), tR.end(), 0.0);
  const double total_tQ = std::accumulate(tQ.begin(), tQ.end(), 0.0);

  std::vector<double> SRx(n, 0.0);
  auto x_groups = groups_by_value(xvals);
  double cum_tR = 0.0;
  for (const auto& grp : x_groups) {
    double group_tR = 0.0;
    for (int idx : grp) group_tR += tR[idx];
    for (int idx : grp) {
      const double sum_gt = total_tR - cum_tR - group_tR;
      const double sum_lt = cum_tR;
      SRx[idx] = 0.5 * (sum_gt - sum_lt + tR[idx]);
    }
    cum_tR += group_tR;
  }

  std::vector<double> SQy(n, 0.0);
  std::vector<double> SRy(n, 0.0);
  auto y_groups = groups_by_value(yvals);
  double cum_tQ = 0.0;
  double cum_tR_y = 0.0;
  for (const auto& grp : y_groups) {
    double group_tQ = 0.0;
    double group_tR = 0.0;
    for (int idx : grp) {
      group_tQ += tQ[idx];
      group_tR += tR[idx];
    }
    for (int idx : grp) {
      const double sum_gt_Q = total_tQ - cum_tQ - group_tQ;
      const double sum_lt_Q = cum_tQ;
      SQy[idx] = 0.5 * (sum_gt_Q - sum_lt_Q + tQ[idx]);

      const double sum_gt_R = total_tR - cum_tR_y - group_tR;
      const double sum_lt_R = cum_tR_y;
      SRy[idx] = 0.5 * (sum_gt_R - sum_lt_R + tR[idx]);
    }
    cum_tQ += group_tQ;
    cum_tR_y += group_tR;
  }

  py::array_t<double> dA_arr(n), dB_arr(n), ic_arr(n);
  auto dA = dA_arr.mutable_unchecked<1>();
  auto dB = dB_arr.mutable_unchecked<1>();
  auto ic = ic_arr.mutable_unchecked<1>();

  double sum_ic2 = 0.0;
  for (int i = 0; i < n; ++i) {
    const double dA_i = tQ[i] * tR[i] + SRx[i] + SQy[i];
    const double dB_i = tR[i] * tR[i] + 2.0 * SRy[i];
    const double raw_deriv = dA_i / B - (A / (B * B)) * dB_i;
    const double ic_i = static_cast<double>(n) * raw_deriv;
    dA(i) = dA_i;
    dB(i) = dB_i;
    ic(i) = ic_i;
    sum_ic2 += ic_i * ic_i;
  }

  const double var_ij = sum_ic2 / static_cast<double>(n);
  py::dict out;
  out["agc"] = agc;
  out["ic"] = ic_arr;
  out["var_ij"] = var_ij;
  out["dA"] = dA_arr;
  out["dB"] = dB_arr;
  return out;
}

PYBIND11_MODULE(_acor_cpp, m) {
  m.doc() = "Native kernels for acor";
  m.def("akc_ij_cpp", &akc_ij_cpp, "AKC IJ estimate and influence function",
        py::arg("x"), py::arg("y"));
  m.def("agc_ij_cpp", &agc_ij_cpp, "AGC IJ estimate and influence function",
        py::arg("x"), py::arg("y"));
  m.def("kendall_tau_sign_cpp", &kendall_tau_sign_cpp, "AKC Fenwick Kendall sign",
        py::arg("x"), py::arg("y"));
  m.def("h_bar_vec_v2_cpp", &h_bar_vec_v2_cpp, "Fenwick H_bar vector",
        py::arg("x"), py::arg("y"));
  m.def("kernel_agc_v2_cpp", &kernel_agc_v2_cpp, "Fenwick AGC kernel values",
        py::arg("x_rank"), py::arg("y_rank"), py::arg("rho"));
}
