set_optimize("fastest")
set_languages("cxx20")
set_warnings("all", "extra")
set_encodings("utf-8")
add_rules("plugin.compile_commands.autoupdate")

includes("../xmake/opencl.lua")
add_intel_opencl()
add_includedirs("../base")

target("matrix-mul")
    set_kind("binary")
    add_files("src/matrix-mul.cxx")

target("matrix-mul-optimized")
    set_kind("binary")
    add_files("src/matrix-mul-optimized.cxx")

