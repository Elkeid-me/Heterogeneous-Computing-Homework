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

target("matrix-mul")
    set_kind("binary")
    add_files("src/matrix-mul.cxx")
    add_intel_opencl()

target("cuda-matrix-mul")
    set_kind("binary")
    add_rules("cuda")
    add_files("src/cuda-matrix-mul.cu")
