#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/iostream.h>
#include <omp.h>
#include <boost/math/distributions/beta.hpp>
#include <utility>
#include <matrix.hpp>
#include <stats.hpp>

PYBIND11_MODULE(libnapy, m){
    
    //****** Add datamatrix class to module. *******//
    py::class_<DataMatrix>(m, "DataMatrix", py::buffer_protocol())
        .def(py::init<py::ssize_t, py::ssize_t>())
        /// Construct from a buffer
        .def(py::init([](const py::buffer &b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<double>::format() || info.ndim != 2) {
                throw std::runtime_error("Incompatible buffer format!");
            }

            auto *v = new DataMatrix(info.shape[0], info.shape[1]);
            memcpy(v->data(), info.ptr, sizeof(double) * (size_t) (v->rows() * v->cols()));
            return v;
        }))

        .def("rows", &DataMatrix::rows)
        .def("cols", &DataMatrix::cols)

        /// Bare bones interface
        .def("__getitem__",
             [](const DataMatrix &m, std::pair<py::ssize_t, py::ssize_t> i) {
                 if (i.first >= m.rows() || i.second >= m.cols()) {
                     throw py::index_error();
                 }
                 return m(i.first, i.second);
             })
        .def("__setitem__",
             [](DataMatrix &m, std::pair<py::ssize_t, py::ssize_t> i, double v) {
                 if (i.first >= m.rows() || i.second >= m.cols()) {
                     throw py::index_error();
                 }
                 m(i.first, i.second) = v;
             })
        /// Provide buffer access
        .def_buffer([](DataMatrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                          /* Pointer to buffer */
                {m.rows(), m.cols()},              /* Buffer dimensions */
                {sizeof(double) * size_t(m.cols()), /* Strides (in bytes) for each index */
                 sizeof(double)});
        });

    //********* Add correlation functionality. *********//
    m.def("pearson_with_nans", &statistics::pearson_with_nans, "Computes pairwise NAN-aware Pearson correlation of all rows in given DataMatrix.");
    m.def("spearman_with_nans", &statistics::spearman_with_nans, "Computes pairwise NAN-aware Spearman correlation of all rows in given DataMatrix.");
    m.def("chi_squared_with_nans", &statistics::chi_squared_with_nans, "Computes pairwise Chi-squared tests for all categorical rows in DataMatrix.");
    m.def("anova_with_nans", &statistics::anova_with_nans, "Computes pairwise ANOVA for all combinations of categorical and continuous data.");
    m.def("kruskal_wallis_with_nans", &statistics::kruskal_wallis_with_nans, "Compute pairwise Kruskal-Wallis tests for all combinations of categorical and continuous data.");
    m.def("t_test_with_nans", &statistics::ttest, "Compute pairwise t-tests for all combations for binary and continuous data.");
    m.def("mwu_with_nans", &statistics::mwu_with_nans, "Compute pairwise MWU tests for all combinations of binary and continuous data.");

    //********* Add OMP functionality. *********//
	m.def("get_max_threads", &omp_get_max_threads, "Returns max number of threads");
	m.def("get_num_threads", &omp_get_num_threads, "Returns number active threads");
    m.def("set_num_threads", &omp_set_num_threads, "Set number of threads");

    	
}
