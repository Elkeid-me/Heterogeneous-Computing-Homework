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

target("bmp-rotate-opencl")
    set_kind("binary")
    add_files("src/bmp-rotate-opencl.cxx")
    add_intel_opencl()

target("bmp-rotate-cpu-opencl")
    set_kind("binary")
    add_defines("USE_CPU")
    add_files("src/bmp-rotate-opencl.cxx")
    add_intel_opencl()