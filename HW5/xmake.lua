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

rule("rotate_image")
    on_build(function (target)
        local fs_path = path.join(os.projectdir(), "RotateImage",
                                  "Rotate.fsproj")

        os.runv("dotnet", {"build", fs_path, "--configuration", "Release"})
    end)

target("rotate_image")
    add_rules("rotate_image")
    after_build(function (target)
        local release_dir = path.join(os.projectdir(), "Release")
        local target_dir = path.join(os.projectdir(), "RotateImage",
                                     "bin", "Release", "net10.0")
        os.cp(path.join(target_dir, "*.exe"), release_dir)
        os.cp(path.join(target_dir, "*.dll"), release_dir)
        os.cp(path.join(target_dir, "Rotate.runtimeconfig.json"), release_dir)
    end)

target("rotate_image_opencl")
    set_kind("shared")
    add_defines("ROTATE_IMAGE_OPENCL_EXPORTS")
    add_files("src/rotate_image_opencl.cxx")
    add_includedirs(path.join(os.projectdir(), "..", "base"))
    add_intel_opencl()
    after_build(function (target)
        local release_dir = path.join(os.projectdir(), "Release")
        os.cp(target:targetfile(), release_dir)
    end)
