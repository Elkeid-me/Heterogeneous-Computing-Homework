set_optimize("fastest")
set_languages("cxx23")
set_warnings("all", "extra")
add_rules("plugin.compile_commands.autoupdate")

target("homework-2-1")
    set_kind("binary")
    add_files("src/HW-1.cxx")
    add_links("OpenCL")

    if is_plat("windows") then
        add_includedirs("C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\include")
        add_linkdirs("C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\lib")
        set_encodings("utf-8") -- MSVC特色
    end

target("homework-2-2")
    set_kind("binary")
    add_files("src/HW-2.cxx")
    add_links("OpenCL")

    if is_plat("windows") then
        add_includedirs("C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\include")
        add_linkdirs("C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\lib")
        set_encodings("utf-8")
    end
