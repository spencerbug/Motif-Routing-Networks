#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <cmath>

namespace py = pybind11;

constexpr double inv_sqrt2 = 0.7071067811865475;

/**
 * Performs the Haar wavelet transform on a 1D vector of doubles.
 * This function modifies the input vector in place.
 * * @param data The input vector containing the data to be transformed.
 * * @param n The number of elements in the input vector. It must be a power of 2.
 * * @param depth The number of levels of decomposition to perform. 
 * * If the depth exceeds log2(n), it will be capped to log2(n).
 * 
 */
inline void haar_wavelet_transform(std::vector<double>& data, int n, int depth) {
    std::vector<double> temp(n);
    int stages = 0;
    int N = n;
    while (N > 1) { N /= 2; ++stages; }
    int max_depth = std::min(depth, stages);
    N = n;
    for (int d = 0; d < max_depth; ++d) {
        N /= 2;
        for (int i = 0; i < N; ++i) {
            temp[i] = (data[2 * i] + data[2 * i + 1]) * inv_sqrt2;
            temp[N + i] = (data[2 * i] - data[2 * i + 1]) * inv_sqrt2;
        }
        std::copy(temp.begin(), temp.begin() + 2 * N, data.begin());
    }
}

/**
 * Performs the inverse Haar wavelet transform on a 1D vector of doubles.
 * This function modifies the input vector in place.
 * @param data The input vector containing the data to be transformed.
 * @param n The number of elements in the input vector. It must be a power of 2.
 * @param depth The number of levels of decomposition to perform.
 * If the depth exceeds log2(n), it will be capped to log2(n).
 */
inline void inverse_haar_wavelet_transform(std::vector<double>& data, int n, int depth) {
    std::vector<double> temp(n);
    int stages = 0;
    int N = n;
    while (N > 1) { N /= 2; ++stages; }
    int max_depth = std::min(depth, stages);
    // Compute the starting h for the given depth
    int h = n >> max_depth;
    while (h < n) {
        for (int i = 0; i < h; ++i) {
            temp[2 * i] = (data[i] + data[h + i]) * inv_sqrt2;
            temp[2 * i + 1] = (data[i] - data[h + i]) * inv_sqrt2;
        }
        std::copy(temp.begin(), temp.begin() + 2 * h, data.begin());
        h *= 2;
    }
}

// 3D DWT using Haar wavelet
py::array_t<double> dwt3d_cpp(py::array_t<double> input, int depth, const std::string& boundary = "periodic") {
    if (boundary != "periodic") throw std::runtime_error("Only 'periodic' boundary condition implemented.");

    auto buf = input.request();
    if (buf.ndim != 3) throw std::runtime_error("Input must be a 3D array");


    auto shape = buf.shape;
    int X = shape[0], Y = shape[1], Z = shape[2];

    
    // Ensure each dimension is a power of 2 and depth is valid
    auto is_power_of_2 = [](int n) { return n > 0 && (n & (n - 1)) == 0; };
    if (!is_power_of_2(X) || !is_power_of_2(Y) || !is_power_of_2(Z))
        throw std::runtime_error("All dimensions must be powers of 2.");

    double* ptr = static_cast<double*>(buf.ptr);

    std::vector<double> temp(std::max({X, Y, Z}));

    // Transform along X-axis
    for (int y = 0; y < Y; ++y)
        for (int z = 0; z < Z; ++z) {
            for (int x = 0; x < X; ++x)
                temp[x] = ptr[x * Y * Z + y * Z + z];
            haar_wavelet_transform(temp, X, depth);
            for (int x = 0; x < X; ++x)
                ptr[x * Y * Z + y * Z + z] = temp[x];
        }

    // Transform along Y-axis
    for (int x = 0; x < X; ++x)
        for (int z = 0; z < Z; ++z) {
            for (int y = 0; y < Y; ++y)
                temp[y] = ptr[x * Y * Z + y * Z + z];
            haar_wavelet_transform(temp, Y, depth);
            for (int y = 0; y < Y; ++y)
                ptr[x * Y * Z + y * Z + z] = temp[y];
        }

    // Transform along Z-axis
    for (int x = 0; x < X; ++x)
        for (int y = 0; y < Y; ++y) {
            for (int z = 0; z < Z; ++z)
                temp[z] = ptr[x * Y * Z + y * Z + z];
            haar_wavelet_transform(temp, Z, depth);
            for (int z = 0; z < Z; ++z)
                ptr[x * Y * Z + y * Z + z] = temp[z];
        }

    return input;
}

// 3D IDWT using Haar wavelet
py::array_t<double> idwt3d_cpp(py::array_t<double> coeffs, py::tuple shape, int depth, const std::string& boundary = "periodic") {
    if (boundary != "periodic") throw std::runtime_error("Only 'periodic' boundary condition implemented.");

    auto buf = coeffs.request();
    if (buf.ndim != 3 || shape.size() != 3) throw std::runtime_error("Coeffs and shape must be 3D.");

    int X = shape[0].cast<int>(), Y = shape[1].cast<int>(), Z = shape[2].cast<int>();

    // Ensure each dimension is a power of 2 and depth is valid
    auto is_power_of_2 = [](int n) { return n > 0 && (n & (n - 1)) == 0; };
    if (!is_power_of_2(X) || !is_power_of_2(Y) || !is_power_of_2(Z))
        throw std::runtime_error("All dimensions must be powers of 2.");

    double* ptr = static_cast<double*>(buf.ptr);

    std::vector<double> temp(std::max({X, Y, Z}));

    // Inverse transform along Z-axis
    for (int x = 0; x < X; ++x)
        for (int y = 0; y < Y; ++y) {
            for (int z = 0; z < Z; ++z)
                temp[z] = ptr[x * Y * Z + y * Z + z];
            inverse_haar_wavelet_transform(temp, Z, depth);
            for (int z = 0; z < Z; ++z)
                ptr[x * Y * Z + y * Z + z] = temp[z];
        }

    // Inverse transform along Y-axis
    for (int x = 0; x < X; ++x)
        for (int z = 0; z < Z; ++z) {
            for (int y = 0; y < Y; ++y)
                temp[y] = ptr[x * Y * Z + y * Z + z];
            inverse_haar_wavelet_transform(temp, Y, depth);
            for (int y = 0; y < Y; ++y)
                ptr[x * Y * Z + y * Z + z] = temp[y];
        }

    // Inverse transform along X-axis
    for (int y = 0; y < Y; ++y)
        for (int z = 0; z < Z; ++z) {
            for (int x = 0; x < X; ++x)
                temp[x] = ptr[x * Y * Z + y * Z + z];
            inverse_haar_wavelet_transform(temp, X, depth);
            for (int x = 0; x < X; ++x)
                ptr[x * Y * Z + y * Z + z] = temp[x];
        }

    return coeffs;
}

PYBIND11_MODULE(dwt3d_cpp, m) {
    m.def("dwt3d_cpp", &dwt3d_cpp, "3D Discrete Wavelet Transform (Haar only)",
          py::arg("input"), py::arg("depth"), py::arg("boundary"));
    m.def("idwt3d_cpp", &idwt3d_cpp, "Inverse 3D Discrete Wavelet Transform (Haar only)",
          py::arg("coeffs"), py::arg("shape"), py::arg("depth"), py::arg("boundary"));
}
