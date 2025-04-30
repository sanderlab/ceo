from conans import ConanFile, CMake
#from conan.tools.cmake import CMakeToolchain, CMakeDeps, CMake

class CEOClusteringConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = (("cnpy/cci.20180601"),
                ("eigen/3.4.0"),
                ("fmt/9.1.0"),
                ("boost/1.80.0"),
                ("zlib/1.2.13", "override"))
    generators = "cmake"

    #def generate(self):
    #    tc = CMakeToolchain(self)
    #    # This writes the "conan_toolchain.cmake"
    #    tc.generate()

    #    deps = CMakeDeps(self)
    #    # This writes all the config files (xxx-config.cmake)
    #    deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.definitions["CONAN_CXX_FLAGS"] += " -Ofast -march=ivybridge"
        cmake.configure()
        cmake.build()


