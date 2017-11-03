#include <iostream>
#include <boost/python.hpp>

char const* greet()
{
   return "hello, world";
}


BOOST_PYTHON_MODULE(libcabbage)
{
    using namespace boost::python;
    def("greet", greet);
}
