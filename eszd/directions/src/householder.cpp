#include <torch/extension.h>
#include <c10/util/Exception.h>
#include <vector>

using namespace torch::indexing;
/*

at::Tensor orgqr_normalized_eye_forward(torch::Tensor param, torch::Tensor eye) {
    TORCH_CHECK(param.is_floating_point(), "\"param\" must be a floating point tensor");
    TORCH_CHECK(param.dim() == 3, "\"param\" must have 3 dimensions");
//    TORCH_CHECK(eye.dim() == 3, "\"eye\" must have 3 dimensions");
    TORCH_CHECK(!eye.requires_grad(), "\"eye\" must not require grad");
    auto d = param.size(0);
    auto l = param.size(1);
//    TORCH_CHECK(eye.size(0) == b && eye.size(1) == d && eye.size(2) == r, "\"eye\" dimensions must match parameters")
 //   TORCH_CHECK(d >= r, "\"param\" dim[1] must be greater or equal than dim[2]");
  //  TORCH_CHECK(param.device() == eye.device(), "Device mismatch");
    const auto options = at::TensorOptions()
        .dtype(param.dtype())
        .device(param.device())
        .requires_grad(false);
    auto m = eye;
    auto minus_tau_param = -2 * param;
    auto cx = torch::empty({1, l}, options);
    at::bmm_out(cx, param, m);
    return m;
/*    auto ax = torch::empty_like(m);
    auto bx = torch::empty_like(m);
    auto cx = torch::empty({b, 1, r}, options);
    for (int j=r-1; j>=0; j--) {
        if (j < r-1) {
            m = ax;
        }
        auto uT = param.index({Slice(), Slice(), j}).unsqueeze(1);
        auto mtu = minus_tau_param.index({Slice(), Slice(), j}).unsqueeze(2);
        at::bmm_out(cx, uT, m);
        at::baddbmm_out(bx, m, mtu, cx);
        std::swap(ax, bx);
    }
    return ax;
}
*/

at::Tensor householder_reflector(int d, int l) {
    torch::Tensor v = torch::randn({d, 1});
    auto mtau = -2 * v;
    auto m = torch::eye(d, l);
    auto ax = torch::empty({d, l});
    auto bx = torch::empty({d, l});
    auto cx = torch::empty({1, l});
    auto uT = v.index({l-1, Slice()}).unsqueeze(0);
    auto mtu = mtau.index({l-1, Slice()}).unsqueeze(1);
    at::bmm_out(cx, uT, m);
    std::cout << cx << std::endl;
    at::baddbmm_out(bx, m, mtu, cx);

    return bx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_reflector", &householder_reflector, "ORGQR normalized input forward");
}