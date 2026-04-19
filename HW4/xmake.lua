set_optimize("fastest")
set_languages("cxx20")
set_warnings("all", "extra")
set_encodings("utf-8")
add_rules("plugin.compile_commands.autoupdate")

local function add_intel_opencl()
    add_includedirs("C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\include")
    add_linkdirs("C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\lib")
    add_links("OpenCL")
end

local function add_intel_sycl()
    add_intel_opencl()
    add_cxxflags("-fsycl")
    add_links("sycl")
end

target("matrix-mul")
    set_kind("binary")
    add_files("src/matrix-mul.cxx")
    add_intel_opencl()

target("sycl-matrix-mul")
    set_kind("binary")
    add_files("src/sycl-matrix-mul.cxx")
    add_intel_sycl()
