set_optimize("fastest")
set_languages("cxx20")
set_warnings("all", "extra")
set_encodings("utf-8")
add_rules("plugin.compile_commands.autoupdate")

target("matrix-mul")
    set_kind("binary")
    includes("../xmake/opencl.lua")
    add_intel_opencl()
    add_includedirs("../base")
    add_files("src/matrix-mul.cxx")

if is_plat("linux") then
    target("cuda-matrix-mul")
        set_kind("binary")
        add_rules("cuda")
        add_files("src/cuda-matrix-mul.cu")
end
