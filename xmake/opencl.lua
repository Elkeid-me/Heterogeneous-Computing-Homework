function add_intel_opencl()
    add_includedirs("C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\include")
    add_linkdirs("C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\lib")
    add_links("OpenCL")
end
