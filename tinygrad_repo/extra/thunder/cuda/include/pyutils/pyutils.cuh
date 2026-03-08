#pragma once

#include "util.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for automatic Python list -> std::vector conversion

namespace kittens {
namespace py {

template<typename T> struct from_object {
    static T make(pybind11::object obj) {
        return obj.cast<T>();
    }
    static T unwrap(pybind11::object obj, int dev_idx) {
        return make(obj); // Scalars should be passed in as a scalar
    }
};
template<ducks::gl::all GL> struct from_object<GL> {
    static GL make(pybind11::object obj) {
        // Check if argument is a torch.Tensor
        if (pybind11::hasattr(obj, "__class__") && 
            obj.attr("__class__").attr("__name__").cast<std::string>() == "Tensor") {
        
            // Check if tensor is contiguous
            if (!obj.attr("is_contiguous")().cast<bool>()) {
                throw std::runtime_error("Tensor must be contiguous");
            }
            if (obj.attr("device").attr("type").cast<std::string>() == "cpu") {
                throw std::runtime_error("Tensor must be on CUDA device");
            }
            
            // Get shape, pad with 1s if needed
            std::array<int, 4> shape = {1, 1, 1, 1};
            auto py_shape = obj.attr("shape").cast<pybind11::tuple>();
            size_t dims = py_shape.size();
            if (dims > 4) {
                throw std::runtime_error("Expected Tensor.ndim <= 4");
            }
            for (size_t i = 0; i < dims; ++i) {
                shape[4 - dims + i] = pybind11::cast<int>(py_shape[i]);
            }
            
            // Get data pointer using data_ptr()
            uint64_t data_ptr = obj.attr("data_ptr")().cast<uint64_t>();
            
            // Create GL object using make_gl
            return make_gl<GL>(data_ptr, shape[0], shape[1], shape[2], shape[3]);
        }
        throw std::runtime_error("Expected a torch.Tensor");
    }
    static GL unwrap(pybind11::object obj, int dev_idx) {
        if (!pybind11::isinstance<pybind11::list>(obj))
            throw std::runtime_error("GL unwrap expected a Python list.");
        pybind11::list lst = pybind11::cast<pybind11::list>(obj);
        if (dev_idx >= lst.size())
            throw std::runtime_error("Device index out of bounds.");
        return *lst[dev_idx].cast<std::shared_ptr<GL>>();
    }
};
template<ducks::pgl::all PGL> struct from_object<PGL> {
    static PGL make(pybind11::object obj) {
        static_assert(!PGL::MULTICAST, "Multicast not yet supported on pyutils. Please initialize the multicast pointer manually.");
        if (!pybind11::isinstance<pybind11::list>(obj))
            throw std::runtime_error("PGL from_object expected a Python list.");
        pybind11::list tensors = pybind11::cast<pybind11::list>(obj);
        if (tensors.size() != PGL::num_devices)
            throw std::runtime_error("Expected a list of " + std::to_string(PGL::num_devices) + " tensors");
        std::array<int, 4> shape = {1, 1, 1, 1};
        uint64_t data_ptrs[PGL::num_devices];
        for (int i = 0; i < PGL::num_devices; i++) {
            auto tensor = tensors[i];
            if (!pybind11::hasattr(tensor, "__class__") || 
                tensor.attr("__class__").attr("__name__").cast<std::string>() != "Tensor")
                throw std::runtime_error("Expected a list of torch.Tensor");
            if (!tensor.attr("is_contiguous")().cast<bool>())
                throw std::runtime_error("Tensor must be contiguous");
            if (tensor.attr("device").attr("type").cast<std::string>() == "cpu")
                throw std::runtime_error("Tensor must be on CUDA device");
            auto py_shape = tensor.attr("shape").cast<pybind11::tuple>();
            size_t dims = py_shape.size();
            if (dims > 4)
                throw std::runtime_error("Expected Tensor.ndim <= 4");
            for (size_t j = 0; j < dims; ++j) {
                if (i == 0)
                    shape[4 - dims + j] = pybind11::cast<int>(py_shape[j]);
                else if (shape[4 - dims + j] != pybind11::cast<int>(py_shape[j]))
                    throw std::runtime_error("All tensors must have the same shape");
            }
            data_ptrs[i] = tensor.attr("data_ptr")().cast<uint64_t>();
        }
        return make_pgl<PGL>(data_ptrs, shape[0], shape[1], shape[2], shape[3]);
    }
    static PGL unwrap(pybind11::object obj, int dev_idx) {
        return *obj.cast<std::shared_ptr<PGL>>();
    }
};

static std::unordered_set<std::string> registered;
template<typename T> static void register_pyclass(pybind11::module &m) {
    if constexpr (ducks::gl::all<T> || ducks::pgl::all<T>) {
        std::string _typename = typeid(T).name();
        if (registered.find(_typename) == registered.end()) {
            pybind11::class_<T, std::shared_ptr<T>>(m, _typename.c_str());
            registered.insert(_typename);
        }
    }
}
template<typename T> static pybind11::object multigpu_make(pybind11::object obj) {
    if constexpr (ducks::gl::all<T>) {
        if (!pybind11::isinstance<pybind11::list>(obj))
            throw std::runtime_error("multigpu_make [GL] expected a Python list.");
        pybind11::list lst = pybind11::cast<pybind11::list>(obj);
        std::vector<std::shared_ptr<T>> gls;
        for (int i = 0; i < lst.size(); i++)
            gls.push_back(std::make_shared<T>(from_object<T>::make(lst[i])));
        return pybind11::cast(gls);
    } else if constexpr (ducks::pgl::all<T>) {
        return pybind11::cast(std::make_shared<T>(from_object<T>::make(obj)));
    } else {
        return pybind11::cast(from_object<T>::make(obj));
    }
}

template<typename T> concept has_dynamic_shared_memory = requires(T t) { { t.dynamic_shared_memory() } -> std::convertible_to<int>; };
template<typename T> concept is_multigpu_globals = requires { 
    { T::num_devices } -> std::convertible_to<std::size_t>;
    { T::dev_idx } -> std::convertible_to<std::size_t>;
} && T::num_devices >= 1;

template<typename> struct trait;
template<typename MT, typename T> struct trait<MT T::*> { using member_type = MT; using type = T; };
template<typename> using object = pybind11::object;
template<auto kernel, typename TGlobal> static void bind_kernel(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def(name, [](object<decltype(member_ptrs)>... args, pybind11::kwargs kwargs) {
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};
        cudaStream_t raw_stream = nullptr;
        if (kwargs.contains("stream")) {
            // Extract stream pointer
            uintptr_t stream_ptr = kwargs["stream"].attr("cuda_stream").cast<uintptr_t>();
            raw_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        }
        if constexpr (has_dynamic_shared_memory<TGlobal>) {
            int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
            kernel<<<__g__.grid(), __g__.block(), __dynamic_shared_memory__, raw_stream>>>(__g__);
        } else {
            kernel<<<__g__.grid(), __g__.block(), 0, raw_stream>>>(__g__);
        }
    });
}
template<auto function, typename TGlobal> static void bind_function(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def(name, [](object<decltype(member_ptrs)>... args) {
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};
        function(__g__);
    });
}
static void bind_multigpu_boilerplate(auto m) {
    m.def("enable_all_p2p_access", [](const std::vector<int>& device_ids) {
        int device_count;
        CUDACHECK(cudaGetDeviceCount(&device_count));
        if (device_count < device_ids.size())
            throw std::runtime_error("Not enough CUDA devices available");
        for (int i = 0; i < device_ids.size(); i++) {
            CUDACHECK(cudaSetDevice(device_ids[i]));
            for (int j = 0; j < device_ids.size(); j++) {
                if (i == j) continue;
                int can_access = 0;
                CUDACHECK(cudaDeviceCanAccessPeer(&can_access, device_ids[i], device_ids[j]));
                if (!can_access)
                    throw std::runtime_error("Device " + std::to_string(device_ids[i]) + " cannot access device " + std::to_string(device_ids[j]));
                cudaError_t res = cudaDeviceEnablePeerAccess(device_ids[j], 0);
                if (res != cudaSuccess && res != cudaErrorPeerAccessAlreadyEnabled) {
                    CUDACHECK(res);
                }
            }
        }
    });
    pybind11::class_<KittensClub, std::shared_ptr<KittensClub>>(m, "KittensClub")
        .def(pybind11::init([](const std::vector<int>& device_ids) {
            int device_count;
            CUDACHECK(cudaGetDeviceCount(&device_count));
            if (device_count < device_ids.size())
                throw std::runtime_error("Not enough CUDA devices available");
            auto club = std::make_shared<KittensClub>(device_ids.data(), device_ids.size());
            club->execute([&](int dev_idx, cudaStream_t stream) {}); // warmup
            return club;
        }), pybind11::arg("device_ids"))
        .def(pybind11::init([](const std::vector<int>& device_ids, const std::vector<pybind11::object>& streams) {
            int device_count;
            CUDACHECK(cudaGetDeviceCount(&device_count));
            if (device_count < device_ids.size())
                throw std::runtime_error("Not enough CUDA devices available");
            if (streams.size() != device_ids.size())
                throw std::runtime_error("Number of streams must match number of devices");
            
            std::vector<cudaStream_t> raw_streams(streams.size());
            for (size_t i = 0; i < streams.size(); ++i) {
                uintptr_t stream_ptr = streams[i].attr("cuda_stream").cast<uintptr_t>();
                raw_streams[i] = reinterpret_cast<cudaStream_t>(stream_ptr);
            }
            
            auto club = std::make_shared<KittensClub>(device_ids.data(), raw_streams.data(), device_ids.size());
            club->execute([&](int dev_idx, cudaStream_t stream) {}); // warmup
            return club;
        }), pybind11::arg("device_ids"), pybind11::arg("streams"));
}
template<auto kernel, typename TGlobal> static void bind_multigpu_kernel(auto m, auto name, auto TGlobal::*... member_ptrs) {
    static_assert(is_multigpu_globals<TGlobal>, "Multigpu globals must have a member num_devices >= 1 and dev_idx");
    (register_pyclass<typename trait<decltype(member_ptrs)>::member_type>(m), ...);
    m.def((std::string("make_globals_")+name).c_str(), [](object<decltype(member_ptrs)>... args) -> std::vector<pybind11::object> {
        return {multigpu_make<typename trait<decltype(member_ptrs)>::member_type>(args)...};
    });
    m.def(name, [](std::shared_ptr<KittensClub> club, object<decltype(member_ptrs)>... args) {
        std::vector<TGlobal> __g__;
        for (int i = 0; i < TGlobal::num_devices; i++) {
            __g__.emplace_back(from_object<typename trait<decltype(member_ptrs)>::member_type>::unwrap(args, i)...);
            __g__.back().dev_idx = i;
        }
        if constexpr (has_dynamic_shared_memory<TGlobal>) {
            club->execute([&](int dev_idx, cudaStream_t stream) {
                int __dynamic_shared_memory__ = (int)__g__[dev_idx].dynamic_shared_memory();
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
                kernel<<<__g__[dev_idx].grid(), __g__[dev_idx].block(), __dynamic_shared_memory__, stream>>>(__g__[dev_idx]);
            });
        } else {
            club->execute([&](int dev_idx, cudaStream_t stream) {
                kernel<<<__g__[dev_idx].grid(), __g__[dev_idx].block(), 0, stream>>>(__g__[dev_idx]);
            });
        }
    });
    // TODO: PGL destructor binding
}

} // namespace py
} // namespace kittens
