set_optimize("fastest")
set_languages("cxx20")
set_warnings("all", "extra")
set_encodings("utf-8")
add_rules("plugin.compile_commands.autoupdate")

includes("../xmake/opencl.lua")
add_intel_opencl()

target("homework-1")
    set_kind("binary")
    add_files("src/HW-1.cxx")

target("homework-2")
    set_kind("binary")
    add_files("src/HW-2.cxx", "../base/base.cxx")
    add_includedirs("../base")
