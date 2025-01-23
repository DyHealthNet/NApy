#include <matrix.hpp>

double DataMatrix::operator()(py::ssize_t i, py::ssize_t j) const {
    return m_data[(size_t) (i * m_cols + j)];
}

double& DataMatrix::operator()(py::ssize_t i, py::ssize_t j) {
    return m_data[(size_t) (i * m_cols + j)];
}
