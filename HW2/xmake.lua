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

target("homework-1")
    set_kind("binary")
    add_files("src/HW-1.cxx")
    add_intel_opencl()

target("homework-2")
    set_kind("binary")
    add_files("src/HW-2.cxx")
    add_includedirs(path.join(os.projectdir(), "..", "base"))
    add_intel_opencl()
