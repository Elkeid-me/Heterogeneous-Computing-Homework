set_optimize("fastest")
set_languages("cxx23")
set_warnings("all", "extra")
set_encodings("utf-8")
add_rules("plugin.compile_commands.autoupdate")

target("lenet5")
    set_kind("binary")
    add_files("*.cxx")
